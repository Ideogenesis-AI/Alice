# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Alice project.
#
# Alice is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Alice is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Alice. If not, see <https://www.gnu.org/licenses/>.


"""Tests for the CBE complement module of 1-site-plus TgTRG.

Two properties carry the whole scheme and are checked throughout: the
expansion leaves `ρ` untouched, and the expanded environment reproduces the
exact effective Hamiltonian, so the 1-site effective energy in the expanded
basis still equals the one the unexpanded blocks give. A cooling run reads
`log Z` off `ρ` at every step, so an expansion that moved either would move
the free energy with it.

The state to expand
-------------------
Most tests run on `rank_limited_L6`, a Taylor state compacted well below its
own rank. That is the situation the scheme exists for — a bond narrower than
what cooling wants to put across it — and it is also the only situation in
which a complement exists at all.

A Taylor state left at full rank has none, and `saturated_L6` is that case:
`ρ(τ₀)` is exact on its own bonds, so there is nothing `H` can reach that the
bond does not already span, and the expansion must recognize it and leave the
state alone. The outer bonds are saturated outright, their dimension already
equal to `dim_outer · d²`.
"""

from __future__ import annotations

import pytest
import torch
from nicole import einsum, oplus

from alice.algorithm.tgtrg.complement import (
    _center_isometry_backward,
    _center_isometry_forward,
    _cheap_factors_backward,
    _cheap_factors_forward,
    _complement_vectors,
    _discarded_bulk,
    _fill_center_backward,
    _fill_center_forward,
    _has_complement,
    _left_half,
    _noise_floor,
    _project_complement_left,
    _project_complement_right,
    _regauge_left,
    _regauge_right,
    _right_half,
    expand_backward,
    expand_forward,
)
from alice.algorithm.tgtrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
    step_left_env,
    step_right_env,
)
from alice.algorithm.tgtrg.krylov import inner_product
from alice.algorithm.tgtrg.scheme_1s import matvec_1s
from alice.algorithm.tgtrg.scheme_2s import build_bulk, matvec_2s

from .conftest import build_spinless, thermal_state

# Center site of the forward tests; the backward tests work on bond
# (BWD_SITE - 1, BWD_SITE), the same interior bond read from the other side.
FWD_SITE = 2
BWD_SITE = 3

# Bond dimension the rank-limited fixture is compacted to, comfortably below
# both the exact rank of ρ(τ₀) and the saturation bound `dim_outer · d²`.
NKEEP = 6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def rank_limited_L6():
    """Spinless free-fermion chain, L=6, with `ρ(τ₀)` compacted to `NKEEP`.

    Returns a factory so each test gets its own density matrix: the
    expansion returns tensors rather than mutating, but canonicalization
    moves the state, and a shared fixture would leak that between tests.
    """
    mpo, spc = build_spinless(6)

    def make():
        rho = thermal_state(mpo, spc, 0.05)
        rho.compact({'nkeep': NKEEP})
        rho.canonical(0)
        return rho, mpo

    return make


@pytest.fixture(scope='module')
def saturated_L6():
    """Spinless free-fermion chain, L=6, with `ρ(τ₀)` at its own exact rank."""
    mpo, spc = build_spinless(6)

    def make():
        return thermal_state(mpo, spc, 0.05), mpo

    return make


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _envs_at(rho, mpo, center):
    """Canonicalize `rho` to `center` and return exact environment blocks.

    The right blocks are built first, while `ρ` is still right-canonical;
    moving the center to `center` afterwards only touches sites `0 … center`,
    so every `env_right[j]` with `j >= center` stays exact. The left blocks
    are then stepped up from the boundary over the now left-isometric tensors.
    """
    rho.canonical(0)
    env_right = Environment(rho.L)
    build_right_envs(rho, mpo, env_right)

    rho.canonical(center)
    env_left = Environment(rho.L)
    env_left[0] = left_env_boundary(rho, mpo)
    for j in range(center):
        env_left[j + 1] = step_left_env(env_left[j], rho[j], mpo[j])

    return env_left, env_right


def _isometry_error(T, left: bool) -> float:
    """Largest deviation of a site tensor's Gram matrix from the identity.

    With `left=True` the tensor is contracted over its left bond and *both*
    physical axes, testing `T† T = I`; otherwise over its right bond and both
    physical axes, testing `T T† = I`.
    """
    gram = (einsum('abrs,acrs->bc', T.conj(), T) if left
            else einsum('bcuv,dcuv->bd', T.conj(), T))
    err = 0.0
    for block in gram.data.values():
        mat = block.reshape(block.shape[0], block.shape[1])
        eye = torch.eye(mat.shape[0], dtype=mat.dtype)
        err = max(err, (mat - eye).abs().max().item())
    return err


def _local_energy(M, W, E_left, E_right) -> complex:
    """Return the 1-site effective energy of `M` in the given blocks."""
    return inner_product(M, matvec_1s(M, W, E_left, E_right))


def _forward_pieces(rho, mpo, site_i=FWD_SITE, alpha=None):
    """Assemble the intermediate quantities of a forward expansion.

    Returns `(env_left, env_right, theta, Q, lh, rh)` with the center at
    `site_i`, where `Q` is the center-side projector: the orthonormalized
    column space of the center tensor.
    """
    env_left, env_right = _envs_at(rho, mpo, site_i)
    theta = build_bulk(rho[site_i], rho[site_i + 1])
    A, R = _cheap_factors_forward(theta, alpha)
    lh = _left_half(env_left[site_i], A, mpo[site_i])
    rh = _right_half(env_right[site_i + 1], R, mpo[site_i + 1])
    return env_left, env_right, theta, _center_isometry_forward(rho[site_i]), lh, rh


def _backward_pieces(rho, mpo, site_i=BWD_SITE, alpha=None):
    """Assemble the intermediate quantities of a backward expansion.

    Returns `(env_left, env_right, theta, Q, lh, rh)` with the center at
    `site_i`, where `Q` is the center-side projector: the orthonormalized row
    space of the center tensor.
    """
    env_left, env_right = _envs_at(rho, mpo, site_i)
    theta = build_bulk(rho[site_i - 1], rho[site_i])
    L, B = _cheap_factors_backward(theta, alpha)
    lh = _left_half(env_left[site_i - 1], L, mpo[site_i - 1])
    rh = _right_half(env_right[site_i], B, mpo[site_i])
    return env_left, env_right, theta, _center_isometry_backward(rho[site_i]), lh, rh


# ---------------------------------------------------------------------------
# _cheap_factors_forward / _cheap_factors_backward
# ---------------------------------------------------------------------------

class TestCheapFactors:
    """The factored form of Θ, the cost saving the whole scheme rests on."""

    def test_forward_reconstructs_theta(self, rank_limited_L6):
        """Untruncated, the two factors multiply back to Θ."""
        rho, mpo = rank_limited_L6()
        rho.canonical(FWD_SITE)
        theta = build_bulk(rho[FWD_SITE], rho[FWD_SITE + 1])

        A, R = _cheap_factors_forward(theta, None)
        back = einsum('abxs,bdyv->adxsyv', A, R)
        assert (back - theta).norm() / theta.norm() < 1e-12

    def test_backward_reconstructs_theta(self, rank_limited_L6):
        """The mirror split is exact in the same way."""
        rho, mpo = rank_limited_L6()
        rho.canonical(BWD_SITE)
        theta = build_bulk(rho[BWD_SITE - 1], rho[BWD_SITE])

        L, B = _cheap_factors_backward(theta, None)
        back = einsum('abxs,bdyv->adxsyv', L, B)
        assert (back - theta).norm() / theta.norm() < 1e-12

    def test_forward_factor_is_left_isometric(self, rank_limited_L6):
        """`A` carries no singular values, so it is an isometry."""
        rho, mpo = rank_limited_L6()
        rho.canonical(FWD_SITE)
        theta = build_bulk(rho[FWD_SITE], rho[FWD_SITE + 1])

        A, _R = _cheap_factors_forward(theta, None)
        assert _isometry_error(A, left=True) < 1e-12

    def test_backward_factor_is_right_isometric(self, rank_limited_L6):
        """Mirror of the forward case: `B` is the isometric factor."""
        rho, mpo = rank_limited_L6()
        rho.canonical(BWD_SITE)
        theta = build_bulk(rho[BWD_SITE - 1], rho[BWD_SITE])

        _L, B = _cheap_factors_backward(theta, None)
        assert _isometry_error(B, left=False) < 1e-12

    def test_alpha_truncation_caps_the_internal_bond(self, rank_limited_L6):
        """`alpha` is what keeps the complement search below 2-site cost."""
        rho, mpo = rank_limited_L6()
        rho.canonical(FWD_SITE)
        theta = build_bulk(rho[FWD_SITE], rho[FWD_SITE + 1])

        A, R = _cheap_factors_forward(theta, 2)
        assert A.indices[1].dim <= 2
        assert R.indices[0].dim == A.indices[1].dim

    def test_both_physical_axes_stay_with_their_site(self, rank_limited_L6):
        """The bipartition is the 2-site one: `phys_out` does not migrate.

        `H` never touches `phys_out`, but the bond still has to span it — the
        split that pretends otherwise would cut Θ in the wrong place.
        """
        rho, mpo = rank_limited_L6()
        rho.canonical(FWD_SITE)
        theta = build_bulk(rho[FWD_SITE], rho[FWD_SITE + 1])

        A, R = _cheap_factors_forward(theta, None)
        assert len(A.indices) == 4
        assert len(R.indices) == 4
        assert A.indices[2].dim == theta.indices[2].dim
        assert A.indices[3].dim == theta.indices[3].dim


# ---------------------------------------------------------------------------
# _center_isometry_forward / _center_isometry_backward
# ---------------------------------------------------------------------------

class TestCenterIsometry:
    """The center tensor is not an isometry, so it gets orthonormalized."""

    def test_forward_is_left_isometric(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        rho.canonical(FWD_SITE)
        assert _isometry_error(_center_isometry_forward(rho[FWD_SITE]),
                               left=True) < 1e-12

    def test_backward_is_right_isometric(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        rho.canonical(BWD_SITE)
        assert _isometry_error(_center_isometry_backward(rho[BWD_SITE]),
                               left=False) < 1e-12

    def test_forward_spans_the_center_tensor(self, rank_limited_L6):
        """Projecting the center tensor onto its own span returns it."""
        rho, mpo = rank_limited_L6()
        rho.canonical(FWD_SITE)
        M = rho[FWD_SITE]
        Q = _center_isometry_forward(M)

        t1 = einsum('xcrs,xbrs->cb', M, Q.conj())
        proj = einsum('cb,abrs->acrs', t1, Q)
        assert (proj - M).norm() / M.norm() < 1e-12

    def test_backward_spans_the_center_tensor(self, rank_limited_L6):
        """Mirror of the forward case."""
        rho, mpo = rank_limited_L6()
        rho.canonical(BWD_SITE)
        M = rho[BWD_SITE]
        Q = _center_isometry_backward(M)

        t1 = einsum('axrs,bxrs->ab', M, Q.conj())
        proj = einsum('ab,bcrs->acrs', t1, Q)
        assert (proj - M).norm() / M.norm() < 1e-12

    def test_center_tensor_is_not_itself_a_projector(self, rank_limited_L6):
        """The reason the split is needed at all.

        `M M†` is a projector only for an isometry, and the center tensor
        carries the singular values. Projecting with it directly would not
        remove the kept space cleanly.
        """
        rho, mpo = rank_limited_L6()
        rho.canonical(FWD_SITE)
        assert _isometry_error(rho[FWD_SITE], left=True) > 1e-6


# ---------------------------------------------------------------------------
# _left_half / _right_half
# ---------------------------------------------------------------------------

class TestHalfContractions:
    """The factored half-applications of `H`, and what they reassemble to."""

    def test_left_half_rank(self, rank_limited_L6):
        """Five axes: the spectator `phys_out` rides along with the rest."""
        rho, mpo = rank_limited_L6()
        *_, lh, _rh = _forward_pieces(rho, mpo)
        assert len(lh.indices) == 5

    def test_right_half_rank(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        *_, _lh, rh = _forward_pieces(rho, mpo)
        assert len(rh.indices) == 5

    def test_halves_reconstruct_H_theta(self, rank_limited_L6):
        """Contracted over α and the MPO bond, the halves give `H·Θ`.

        This is the identity the whole construction rests on: whatever the
        projections then remove, they remove it from `H·Θ` and from nothing
        else.
        """
        rho, mpo = rank_limited_L6()
        env_left, env_right, theta, _Q, lh, rh = _forward_pieces(rho, mpo)

        joined = _discarded_bulk(lh, rh)
        ref = matvec_2s(theta, mpo[FWD_SITE], mpo[FWD_SITE + 1],
                        env_left[FWD_SITE], env_right[FWD_SITE + 1])
        assert (joined - ref).norm() / ref.norm() < 1e-12

    def test_backward_halves_reconstruct_H_theta(self, rank_limited_L6):
        """The same identity in the backward gauge."""
        rho, mpo = rank_limited_L6()
        env_left, env_right, theta, _Q, lh, rh = _backward_pieces(rho, mpo)

        joined = _discarded_bulk(lh, rh)
        ref = matvec_2s(theta, mpo[BWD_SITE - 1], mpo[BWD_SITE],
                        env_left[BWD_SITE - 1], env_right[BWD_SITE])
        assert (joined - ref).norm() / ref.norm() < 1e-12


# ---------------------------------------------------------------------------
# _project_complement_left / _project_complement_right
# ---------------------------------------------------------------------------

class TestProjectComplement:
    """Each end must remove its own kept space, and only that."""

    def test_left_discarded_is_orthogonal_to_the_kept_space(self,
                                                            rank_limited_L6):
        rho, mpo = rank_limited_L6()
        *_, Q, lh, _rh = _forward_pieces(rho, mpo)

        disc = _project_complement_left(lh, Q)
        overlap = einsum('xαpzw,xbzw->αpb', disc, Q.conj())
        assert overlap.norm() / lh.norm() < 1e-12

    def test_right_discarded_is_orthogonal_to_the_kept_space(self,
                                                             rank_limited_L6):
        rho, mpo = rank_limited_L6()
        rho_c = rho
        _el, _er, _theta, _Q, _lh, rh = _forward_pieces(rho_c, mpo)
        neighbor = rho_c[FWD_SITE + 1]

        disc = _project_complement_right(rh, neighbor)
        overlap = einsum('αpxzw,bxzw->αpb', disc, neighbor.conj())
        assert overlap.norm() / rh.norm() < 1e-12

    def test_left_projection_is_idempotent(self, rank_limited_L6):
        """Removing an orthogonal component twice removes it once."""
        rho, mpo = rank_limited_L6()
        *_, Q, lh, _rh = _forward_pieces(rho, mpo)

        once = _project_complement_left(lh, Q)
        twice = _project_complement_left(once, Q)
        assert (twice - once).norm() / lh.norm() < 1e-12

    def test_right_projection_is_idempotent(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        _el, _er, _theta, _Q, _lh, rh = _forward_pieces(rho, mpo)
        neighbor = rho[FWD_SITE + 1]

        once = _project_complement_right(rh, neighbor)
        twice = _project_complement_right(once, neighbor)
        assert (twice - once).norm() / rh.norm() < 1e-12

    def test_discarded_norms_are_bounded_by_the_originals(self,
                                                          rank_limited_L6):
        """A projection cannot add weight."""
        rho, mpo = rank_limited_L6()
        *_, Q, lh, rh = _forward_pieces(rho, mpo)
        neighbor = rho[FWD_SITE + 1]

        assert _project_complement_left(lh, Q).norm() <= lh.norm() * (1 + 1e-12)
        assert (_project_complement_right(rh, neighbor).norm()
                <= rh.norm() * (1 + 1e-12))


# ---------------------------------------------------------------------------
# _noise_floor / _has_complement
# ---------------------------------------------------------------------------

class TestHasComplement:
    """Telling a real complement apart from the noise of a cancellation."""

    def test_complement_detected_where_it_exists(self, rank_limited_L6):
        """A bond narrower than what cooling wants has somewhere to go."""
        rho, mpo = rank_limited_L6()
        *_, Q, lh, rh = _forward_pieces(rho, mpo)
        neighbor = rho[FWD_SITE + 1]

        theta_disc = _discarded_bulk(_project_complement_left(lh, Q),
                                     _project_complement_right(rh, neighbor))
        assert _has_complement(theta_disc, _noise_floor(lh, rh))

    def test_no_complement_where_rho_is_already_exact(self, saturated_L6):
        """`ρ(τ₀)` at its own rank has nothing `H` can reach beyond it.

        The outer bonds are saturated outright and the interior ones carry
        the exact rank of the Taylor state, so the doubly-discarded tensor
        collapses to roundoff at every one of them.
        """
        rho, mpo = saturated_L6()
        for site in range(rho.L - 1):
            fresh, mpo_i = saturated_L6()
            *_, Q, lh, rh = _forward_pieces(fresh, mpo_i, site_i=site)
            theta_disc = _discarded_bulk(
                _project_complement_left(lh, Q),
                _project_complement_right(rh, fresh[site + 1]),
            )
            assert not _has_complement(theta_disc, _noise_floor(lh, rh))

    def test_the_floor_scales_with_the_halves_not_the_difference(self,
                                                                 rank_limited_L6):
        """The threshold is absolute, keyed on what the subtraction can lose.

        Taken relative to Θ'_disc the test would shrink along with exactly
        the quantity whose smallness it exists to detect.
        """
        rho, mpo = rank_limited_L6()
        *_, lh, rh = _forward_pieces(rho, mpo)
        assert _noise_floor(lh, rh) == pytest.approx(1e-10 * lh.norm() * rh.norm())


# ---------------------------------------------------------------------------
# _complement_vectors
# ---------------------------------------------------------------------------

class TestComplementVectors:
    """The SVD of Θ'_disc, and the two isometries it yields."""

    @staticmethod
    def _disc_forward(rho, mpo):
        """Θ'_disc and its floor for the forward gauge."""
        *_, Q, lh, rh = _forward_pieces(rho, mpo)
        theta_disc = _discarded_bulk(_project_complement_left(lh, Q),
                                     _project_complement_right(rh,
                                                               rho[FWD_SITE + 1]))
        return theta_disc, _noise_floor(lh, rh)

    def test_ranks(self, rank_limited_L6):
        """N_L and N_R are site tensors, four axes each."""
        rho, mpo = rank_limited_L6()
        theta_disc, floor = self._disc_forward(rho, mpo)

        N_L, N_R = _complement_vectors(theta_disc, 3, '<<', floor)
        assert len(N_L.indices) == 4
        assert len(N_R.indices) == 4

    def test_k_is_bounded(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        theta_disc, floor = self._disc_forward(rho, mpo)

        _N_L, N_R = _complement_vectors(theta_disc, 2, '<<', floor)
        assert 0 < N_R.indices[0].dim <= 2

    def test_shared_k_dimension(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        theta_disc, floor = self._disc_forward(rho, mpo)

        N_L, N_R = _complement_vectors(theta_disc, 3, '<<', floor)
        assert N_L.indices[1].dim == N_R.indices[0].dim

    def test_NR_is_right_isometric(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        theta_disc, floor = self._disc_forward(rho, mpo)

        _N_L, N_R = _complement_vectors(theta_disc, 3, '<<', floor)
        assert _isometry_error(N_R, left=False) < 1e-12

    def test_NL_is_left_isometric(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        theta_disc, floor = self._disc_forward(rho, mpo)

        N_L, _N_R = _complement_vectors(theta_disc, 3, '<<', floor)
        assert _isometry_error(N_L, left=True) < 1e-12

    def test_NR_is_orthogonal_to_the_kept_space(self, rank_limited_L6):
        """What makes the `oplus` of the neighbor an isometry again."""
        rho, mpo = rank_limited_L6()
        theta_disc, floor = self._disc_forward(rho, mpo)

        _N_L, N_R = _complement_vectors(theta_disc, 3, '<<', floor)
        overlap = einsum('kcuv,bcuv->kb', N_R, rho[FWD_SITE + 1].conj())
        assert overlap.norm() < 1e-10


# ---------------------------------------------------------------------------
# _regauge_right / _regauge_left
# ---------------------------------------------------------------------------

class TestRegauge:
    """Recovering the isometry `oplus` leaves only approximately in place.

    Out of a near-identity `ρ(τ₀)` the complement comes back orthogonal to
    the kept space only to roughly `1e-10`. The projections are exact to
    roundoff; it is their contraction that cancels, Θ'_disc landing five or
    six orders below the discarded halves that produce it. The fixtures here
    sit in exactly that regime, so these tests fail without the
    re-decomposition and pass with it.
    """

    def test_stacked_neighbor_is_not_already_isometric(self, rank_limited_L6):
        """The defect the re-gauge exists to remove is real and measurable."""
        rho, mpo = rank_limited_L6()
        *_, Q, lh, rh = _forward_pieces(rho, mpo)
        theta_disc = _discarded_bulk(_project_complement_left(lh, Q),
                                     _project_complement_right(rh,
                                                               rho[FWD_SITE + 1]))
        _N_L, N_R = _complement_vectors(theta_disc, 3, '<<',
                                        _noise_floor(lh, rh))

        stacked = oplus(rho[FWD_SITE + 1], N_R, axes=[0])
        assert _isometry_error(stacked, left=False) > 1e-14

    def test_right_regauge_restores_the_isometry(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        *_, Q, lh, rh = _forward_pieces(rho, mpo)
        neighbor = rho[FWD_SITE + 1]
        theta_disc = _discarded_bulk(_project_complement_left(lh, Q),
                                     _project_complement_right(rh, neighbor))
        _N_L, N_R = _complement_vectors(theta_disc, 3, '<<',
                                        _noise_floor(lh, rh))

        regauged = _regauge_right(oplus(neighbor, N_R, axes=[0]),
                                  neighbor.itags[0])
        assert _isometry_error(regauged, left=False) < 1e-14

    def test_right_regauge_keeps_the_old_basis_in_its_span(self,
                                                           rank_limited_L6):
        """What makes the fill-in exact: the span can grow, never shrink."""
        rho, mpo = rank_limited_L6()
        *_, Q, lh, rh = _forward_pieces(rho, mpo)
        neighbor = rho[FWD_SITE + 1]
        theta_disc = _discarded_bulk(_project_complement_left(lh, Q),
                                     _project_complement_right(rh, neighbor))
        _N_L, N_R = _complement_vectors(theta_disc, 3, '<<',
                                        _noise_floor(lh, rh))

        regauged = _regauge_right(oplus(neighbor, N_R, axes=[0]),
                                  neighbor.itags[0])
        t1 = einsum('bxzw,kxzw->bk', neighbor, regauged.conj())
        proj = einsum('bk,kcuv->bcuv', t1, regauged)
        assert (proj - neighbor).norm() / neighbor.norm() < 1e-14

    def test_left_regauge_restores_the_isometry(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        *_, Q, lh, rh = _backward_pieces(rho, mpo)
        neighbor = rho[BWD_SITE - 1]
        theta_disc = _discarded_bulk(_project_complement_left(lh, neighbor),
                                     _project_complement_right(rh, Q))
        N_L, _N_R = _complement_vectors(theta_disc, 3, '>>',
                                        _noise_floor(lh, rh))

        regauged = _regauge_left(oplus(neighbor, N_L, axes=[1]),
                                 neighbor.itags[1])
        assert _isometry_error(regauged, left=True) < 1e-14


# ---------------------------------------------------------------------------
# expand_forward
# ---------------------------------------------------------------------------

class TestExpandForward:
    """The forward expansion as a whole."""

    @staticmethod
    def _expand(rho, mpo, k=3, alpha=None, site_i=FWD_SITE):
        """Run `expand_forward` at `site_i` and return everything around it."""
        env_left, env_right = _envs_at(rho, mpo, site_i)
        M_i, M_i1 = rho[site_i], rho[site_i + 1]
        out = expand_forward(M_i, M_i1, mpo[site_i], mpo[site_i + 1],
                             env_left[site_i], env_right[site_i + 1], k, alpha)
        return (M_i, M_i1, env_left, env_right) + out

    def test_bond_grows(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        M_i, _M_i1, _el, _er, M_i_exp, _exp, _E = self._expand(rho, mpo)
        assert M_i_exp.indices[1].dim > M_i.indices[1].dim

    def test_other_dimensions_are_untouched(self, rank_limited_L6):
        """Only the shared bond moves; the outer bonds and physics stay."""
        rho, mpo = rank_limited_L6()
        M_i, M_i1, _el, _er, M_i_exp, M_i1_exp, _E = self._expand(rho, mpo)

        assert M_i_exp.indices[0].dim == M_i.indices[0].dim
        assert M_i_exp.indices[2].dim == M_i.indices[2].dim
        assert M_i_exp.indices[3].dim == M_i.indices[3].dim
        assert M_i1_exp.indices[1].dim == M_i1.indices[1].dim
        assert M_i1_exp.indices[2].dim == M_i1.indices[2].dim
        assert M_i1_exp.indices[3].dim == M_i1.indices[3].dim

    def test_rho_is_preserved_exactly(self, rank_limited_L6):
        """The load-bearing property: expansion must not move the state."""
        rho, mpo = rank_limited_L6()
        M_i, M_i1, _el, _er, M_i_exp, M_i1_exp, _E = self._expand(rho, mpo)

        before = build_bulk(M_i, M_i1)
        after = build_bulk(M_i_exp, M_i1_exp)
        assert (after - before).norm() / before.norm() < 1e-12

    def test_neighbor_stays_right_isometric(self, rank_limited_L6):
        """Without this the gauge is gone and the fill-in is not exact."""
        rho, mpo = rank_limited_L6()
        *_, M_i1_exp, _E = self._expand(rho, mpo)
        assert _isometry_error(M_i1_exp, left=False) < 1e-12

    def test_effective_energy_is_unchanged(self, rank_limited_L6):
        """The expanded environment must reproduce the exact `H_eff`.

        An approximate block would perturb the propagator, and with it the
        cooling trajectory and every `log Z` read off it.
        """
        rho, mpo = rank_limited_L6()
        M_i, M_i1, env_left, env_right, M_i_exp, _M, E_right_exp = \
            self._expand(rho, mpo)

        plain = _local_energy(M_i, mpo[FWD_SITE], env_left[FWD_SITE],
                              step_right_env(env_right[FWD_SITE + 1], M_i1,
                                             mpo[FWD_SITE + 1]))
        expanded = _local_energy(M_i_exp, mpo[FWD_SITE], env_left[FWD_SITE],
                                 E_right_exp)
        assert expanded == pytest.approx(plain, rel=1e-10)

    def test_norm_is_unchanged(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        M_i, _M_i1, _el, _er, M_i_exp, _M, _E = self._expand(rho, mpo)
        assert M_i_exp.norm() == pytest.approx(M_i.norm(), rel=1e-12)

    def test_alpha_truncation_still_preserves_rho(self, rank_limited_L6):
        """A cheaper complement search is still an exact fill-in.

        `alpha` degrades which directions are found, never whether the state
        survives finding them.
        """
        rho, mpo = rank_limited_L6()
        M_i, M_i1, _el, _er, M_i_exp, M_i1_exp, _E = \
            self._expand(rho, mpo, alpha=2)

        before = build_bulk(M_i, M_i1)
        after = build_bulk(M_i_exp, M_i1_exp)
        assert (after - before).norm() / before.norm() < 1e-12

    def test_zero_k_leaves_the_bond_alone(self, rank_limited_L6):
        """`expand_k = 0` is the setting that reduces `'1sp'` to `'1s'`."""
        rho, mpo = rank_limited_L6()
        M_i, _M_i1, _el, _er, M_i_exp, _M, _E = self._expand(rho, mpo, k=0)
        assert M_i_exp.indices[1].dim == M_i.indices[1].dim

    def test_exact_rho_is_left_alone(self, saturated_L6):
        """No complement, no expansion, and the environment still valid."""
        rho, mpo = saturated_L6()
        M_i, M_i1, _el, env_right, M_i_exp, M_i1_exp, E_right_exp = \
            self._expand(rho, mpo)

        assert M_i_exp.indices[1].dim == M_i.indices[1].dim
        plain = step_right_env(env_right[FWD_SITE + 1], M_i1, mpo[FWD_SITE + 1])
        assert (E_right_exp - plain).norm() < 1e-12

    def test_fill_in_matches_a_zero_padding(self, rank_limited_L6):
        """Against a right-isometric neighbor the fill-in is a zero padding.

        Writing it as a projection instead keeps it exact where the
        complement is not perfectly orthogonal, but on a clean bond the two
        must agree.
        """
        rho, mpo = rank_limited_L6()
        M_i, M_i1, _el, _er, M_i_exp, M_i1_exp, _E = self._expand(rho, mpo)

        theta = build_bulk(M_i, M_i1)
        padded = _fill_center_forward(theta, M_i1_exp)
        assert (padded - M_i_exp).norm() / M_i_exp.norm() < 1e-12


# ---------------------------------------------------------------------------
# expand_backward
# ---------------------------------------------------------------------------

class TestExpandBackward:
    """The backward expansion, mirror of the forward one."""

    @staticmethod
    def _expand(rho, mpo, k=3, alpha=None, site_i=BWD_SITE):
        """Run `expand_backward` at `site_i` and return everything around it."""
        env_left, env_right = _envs_at(rho, mpo, site_i)
        M_i, M_im1 = rho[site_i], rho[site_i - 1]
        out = expand_backward(M_i, M_im1, mpo[site_i], mpo[site_i - 1],
                              env_left[site_i - 1], env_right[site_i], k, alpha)
        return (M_i, M_im1, env_left, env_right) + out

    def test_bond_grows(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        M_i, _M, _el, _er, M_i_exp, _E1, _E2 = self._expand(rho, mpo)
        assert M_i_exp.indices[0].dim > M_i.indices[0].dim

    def test_rho_is_preserved_exactly(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        M_i, M_im1, _el, _er, M_i_exp, M_im1_exp, _E = self._expand(rho, mpo)

        before = build_bulk(M_im1, M_i)
        after = build_bulk(M_im1_exp, M_i_exp)
        assert (after - before).norm() / before.norm() < 1e-12

    def test_neighbor_stays_left_isometric(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        *_, M_im1_exp, _E = self._expand(rho, mpo)
        assert _isometry_error(M_im1_exp, left=True) < 1e-12

    def test_effective_energy_is_unchanged(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        M_i, M_im1, env_left, env_right, M_i_exp, _M, E_left_exp = \
            self._expand(rho, mpo)

        plain = _local_energy(M_i, mpo[BWD_SITE],
                              step_left_env(env_left[BWD_SITE - 1], M_im1,
                                            mpo[BWD_SITE - 1]),
                              env_right[BWD_SITE])
        expanded = _local_energy(M_i_exp, mpo[BWD_SITE], E_left_exp,
                                 env_right[BWD_SITE])
        assert expanded == pytest.approx(plain, rel=1e-10)

    def test_alpha_truncation_still_preserves_rho(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        M_i, M_im1, _el, _er, M_i_exp, M_im1_exp, _E = \
            self._expand(rho, mpo, alpha=2)

        before = build_bulk(M_im1, M_i)
        after = build_bulk(M_im1_exp, M_i_exp)
        assert (after - before).norm() / before.norm() < 1e-12

    def test_exact_rho_is_left_alone(self, saturated_L6):
        rho, mpo = saturated_L6()
        M_i, M_im1, env_left, _er, M_i_exp, _M, E_left_exp = \
            self._expand(rho, mpo)

        assert M_i_exp.indices[0].dim == M_i.indices[0].dim
        plain = step_left_env(env_left[BWD_SITE - 1], M_im1, mpo[BWD_SITE - 1])
        assert (E_left_exp - plain).norm() < 1e-12

    def test_fill_in_matches_a_zero_padding(self, rank_limited_L6):
        rho, mpo = rank_limited_L6()
        M_i, M_im1, _el, _er, M_i_exp, M_im1_exp, _E = self._expand(rho, mpo)

        theta = build_bulk(M_im1, M_i)
        padded = _fill_center_backward(theta, M_im1_exp)
        assert (padded - M_i_exp).norm() / M_i_exp.norm() < 1e-12
