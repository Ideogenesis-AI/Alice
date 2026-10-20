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


"""Tests for the CBE complement module of 1-site-plus TDVP.

The two properties the TDVP expansion must have beyond DMRG's are checked
throughout: the expansion leaves the represented state untouched, and the
expanded environment reproduces the exact effective Hamiltonian, so the 1-site
effective energy in the expanded basis still equals `⟨ψ|H|ψ⟩`.

Most tests run on `heisenberg_L6`, whose bonds sit below `dim_outer · d` and
therefore have somewhere to expand into. The `L=4` fixture is used for the
opposite case: all of its bonds are saturated, so the expansion there must
recognize that there is no complement and leave the state alone.

Interior bonds of a random MPS are often saturated on one side, and since the
complement is the product of the two discarded halves, one saturated end is
enough to make it vanish. Which bonds are saturated depends on the gauge, so
the forward tests work on bond `(3, 4)` and the backward tests on bond `(1, 2)`
— the one interior bond that expands in each direction's own gauge.
"""

from __future__ import annotations

import pytest
import torch
from nicole import einsum, load_space

from alice.algorithm.tdvp.complement import (
    _center_isometry_backward,
    _center_isometry_forward,
    _cheap_factors_backward,
    _cheap_factors_forward,
    _complement_vectors,
    _fill_center_backward,
    _fill_center_forward,
    _has_complement,
    _left_half,
    _project_complement_left,
    _project_complement_right,
    _right_half,
    expand_backward,
    expand_forward,
)
from alice.algorithm.tdvp.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
    step_left_env,
)
from alice.algorithm.tdvp.krylov import inner_product
from alice.algorithm.tdvp.scheme_1s import matvec_1s
from alice.algorithm.tdvp.scheme_2s import build_bulk
from alice.network import build_hamiltonian, build_interaction, init_mps, observe

# Center site of the forward tests: bond (FWD_SITE, FWD_SITE + 1) of the L=6
# fixture is expandable at both ends once the center sits at FWD_SITE.
FWD_SITE = 3

# Center site of the backward tests: bond (BWD_SITE - 1, BWD_SITE) is the
# corresponding bond in the backward gauge.
BWD_SITE = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _envs_at(mps, mpo, center):
    """Canonicalize `mps` to `center` and return exact environment blocks.

    The right blocks are built first, while the state is still right-canonical;
    moving the center to `center` afterwards only touches sites `0 … center`, so
    every `env_right[j]` with `j >= center` stays exact. The left blocks are
    then stepped up from the boundary over the now left-isometric tensors.

    Returns
    -------
    tuple
        `(env_left, env_right)`, exact for a center at `center`.
    """
    mps.canonical(0)
    env_right = Environment(mps.L)
    build_right_envs(mps, mpo, env_right)

    mps.canonical(center)
    env_left = Environment(mps.L)
    env_left[0] = left_env_boundary(mps, mpo)
    for j in range(center):
        env_left[j + 1] = step_left_env(env_left[j], mps[j], mpo[j])

    return env_left, env_right


def _tensor_norm(T) -> float:
    """Frobenius norm of a Nicole `Tensor` over its block data."""
    total = sum(v.abs().pow(2).sum().item() for v in T.data.values())
    return total ** 0.5


def _isometry_error(T, left: bool) -> float:
    """Largest deviation of a site tensor's Gram matrix from the identity.

    With `left=True` the tensor is contracted over its left bond and physical
    index, testing `T† T = I`; otherwise over its right bond and physical index,
    testing `T T† = I`.
    """
    gram = (einsum('abr,acr->bc', T.conj(), T) if left
            else einsum('bcu,dcu->bd', T.conj(), T))
    err = 0.0
    for block in gram.data.values():
        mat = block.reshape(block.shape[0], block.shape[1])
        eye = torch.eye(mat.shape[0], dtype=mat.dtype)
        err = max(err, (mat - eye).abs().max().item())
    return err


def _local_energy(M, W, E_left, E_right) -> complex:
    """Return `⟨M|H_eff|M⟩` for the 1-site effective Hamiltonian."""
    return inner_product(M, matvec_1s(M, W, E_left, E_right))


def _forward_pieces(mps, mpo, site_i=FWD_SITE, alpha=None):
    """Assemble the intermediate quantities of a forward expansion at `site_i`.

    Returns `(env_left, env_right, theta, Q, lh, rh)` with the center at
    `site_i`, where `Q` is the center-side projector: the orthonormalized
    column space of the center tensor.
    """
    env_left, env_right = _envs_at(mps, mpo, site_i)
    theta = build_bulk(mps[site_i], mps[site_i + 1])
    A, R = _cheap_factors_forward(theta, alpha)
    lh = _left_half(env_left[site_i], A, mpo[site_i])
    rh = _right_half(env_right[site_i + 1], R, mpo[site_i + 1])
    return env_left, env_right, theta, _center_isometry_forward(mps[site_i]), lh, rh


def _backward_pieces(mps, mpo, site_i=BWD_SITE, alpha=None):
    """Assemble the intermediate quantities of a backward expansion at `site_i`.

    Returns `(env_left, env_right, theta, Q, lh, rh)` with the center at
    `site_i`, where `Q` is the center-side projector: the orthonormalized row
    space of the center tensor.
    """
    env_left, env_right = _envs_at(mps, mpo, site_i)
    theta = build_bulk(mps[site_i - 1], mps[site_i])
    L, B = _cheap_factors_backward(theta, alpha)
    lh = _left_half(env_left[site_i - 1], L, mpo[site_i - 1])
    rh = _right_half(env_right[site_i], B, mpo[site_i])
    return env_left, env_right, theta, _center_isometry_backward(mps[site_i]), lh, rh


@pytest.fixture
def heisenberg_su2_L4():
    """Heisenberg spin-1/2 chain, L=4, OBC, SU(2), with a random complex MPS.

    The one non-Abelian fixture here: `oplus` compresses linearly dependent CG
    components, so the expanded bond can come out smaller than `b + k`, and the
    expansion still has to preserve the state exactly.
    """
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': 4, 'bcx': 'OBC', 'n2x': True},
        'model': {
            'category': 'bosonic', 'label': 'Heisenberg',
            'symmetry': 'SU2', 'spin': 0.5, 'J': 1.0,
        },
    }
    interactions, spc, geo = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, geo.L, spc)
    mpo.to(dtype=torch.complex128)

    Spc, Op = load_space('Spin', 'SU2', {'J': 0.5})
    mps = init_mps(4, Spc, Op, bond_dim=2, dtype=torch.complex128)
    mps.canonical(0)
    mps.normalize()
    return mps, mpo


# ---------------------------------------------------------------------------
# _cheap_factors_forward / _cheap_factors_backward
# ---------------------------------------------------------------------------

class TestCheapFactors:
    """Tests for the SVD that feeds the two half-applications of H."""

    def test_ranks(self, heisenberg_L6):
        """Both factors are rank-3 site-shaped tensors, either way round."""
        mps, _ = heisenberg_L6
        theta = build_bulk(mps[FWD_SITE], mps[FWD_SITE + 1])
        for first, second in (_cheap_factors_forward(theta, alpha=None),
                              _cheap_factors_backward(theta, alpha=None)):
            assert len(first.indices) == 3
            assert len(second.indices) == 3

    def test_forward_reconstructs_theta(self, heisenberg_L6):
        """A ⊗ R reproduces Θ, so the cheap factors are a faithful split."""
        mps, _ = heisenberg_L6
        theta = build_bulk(mps[FWD_SITE], mps[FWD_SITE + 1])
        A, R = _cheap_factors_forward(theta, alpha=None)
        diff = build_bulk(A, R) - theta
        assert _tensor_norm(diff) < 1e-10 * max(_tensor_norm(theta), 1.0)

    def test_backward_reconstructs_theta(self, heisenberg_L6):
        """L ⊗ B likewise reproduces Θ, with the singular values on the left."""
        mps, _ = heisenberg_L6
        theta = build_bulk(mps[FWD_SITE], mps[FWD_SITE + 1])
        L, B = _cheap_factors_backward(theta, alpha=None)
        diff = build_bulk(L, B) - theta
        assert _tensor_norm(diff) < 1e-10 * max(_tensor_norm(theta), 1.0)

    def test_forward_factor_is_left_isometric(self, heisenberg_L6):
        """The forward sweep's factor facing the sweep is an isometry."""
        mps, _ = heisenberg_L6
        A, _ = _cheap_factors_forward(build_bulk(mps[FWD_SITE], mps[FWD_SITE + 1]), alpha=None)
        assert _isometry_error(A, left=True) < 1e-9

    def test_backward_factor_is_right_isometric(self, heisenberg_L6):
        """The backward sweep's factor facing the sweep is an isometry."""
        mps, _ = heisenberg_L6
        _, B = _cheap_factors_backward(build_bulk(mps[FWD_SITE], mps[FWD_SITE + 1]), alpha=None)
        assert _isometry_error(B, left=False) < 1e-9

    def test_the_factor_can_span_more_than_the_center_tensor(self, heisenberg_L6):
        """Untruncated, the isometric factor of Θ overreaches the kept space.

        `decomp` keeps every charge sector at full dimension whatever its rank,
        so the factor's internal bond can exceed the MPS bond. That is why it
        cannot serve as the center-side projector, however orthonormal it is:
        projecting with it would remove directions the complement needs.
        """
        mps, _ = heisenberg_L6
        theta = build_bulk(mps[FWD_SITE], mps[FWD_SITE + 1])
        A, _ = _cheap_factors_forward(theta, alpha=None)
        assert A.indices[1].dim > mps[FWD_SITE].indices[1].dim

    def test_alpha_truncation(self, heisenberg_L6):
        """The internal bond honors `alpha` when it is below the current bond."""
        mps, _ = heisenberg_L6
        A, _ = _cheap_factors_forward(build_bulk(mps[FWD_SITE], mps[FWD_SITE + 1]), alpha=1)
        assert A.indices[1].dim <= 1

    def test_no_truncation_keeps_bond(self, heisenberg_L6):
        """With `alpha=None` the internal bond is at most the attainable rank."""
        mps, _ = heisenberg_L6
        theta = build_bulk(mps[FWD_SITE], mps[FWD_SITE + 1])
        A, _ = _cheap_factors_forward(theta, alpha=None)
        assert A.indices[1].dim <= mps[FWD_SITE].indices[0].dim * mps[FWD_SITE].indices[2].dim


# ---------------------------------------------------------------------------
# _center_isometry_forward / _center_isometry_backward
# ---------------------------------------------------------------------------

class TestCenterIsometry:
    """Tests for the orthonormal basis that projects the center end."""

    def test_forward_is_left_isometric(self, heisenberg_L6):
        mps, mpo = heisenberg_L6
        _envs_at(mps, mpo, FWD_SITE)
        assert _isometry_error(_center_isometry_forward(mps[FWD_SITE]), left=True) < 1e-9

    def test_backward_is_right_isometric(self, heisenberg_L6):
        mps, mpo = heisenberg_L6
        _envs_at(mps, mpo, BWD_SITE)
        assert _isometry_error(_center_isometry_backward(mps[BWD_SITE]), left=False) < 1e-9

    def test_forward_spans_the_center_tensor(self, heisenberg_L6):
        """The basis covers the center tensor exactly — no more, no less.

        Both halves of that matter. Spanning less would leave kept directions
        in the complement, which `oplus` would then duplicate; spanning more
        would project away directions the complement is made of.
        """
        mps, mpo = heisenberg_L6
        _envs_at(mps, mpo, FWD_SITE)
        M, Q = mps[FWD_SITE], _center_isometry_forward(mps[FWD_SITE])
        remainder = M - einsum('xbz,xcz,acr->abr', M, Q.conj(), Q)
        assert _tensor_norm(remainder) < 1e-10 * max(_tensor_norm(M), 1.0)
        assert Q.indices[1].dim == M.indices[1].dim

    def test_backward_spans_the_center_tensor(self, heisenberg_L6):
        mps, mpo = heisenberg_L6
        _envs_at(mps, mpo, BWD_SITE)
        M, Q = mps[BWD_SITE], _center_isometry_backward(mps[BWD_SITE])
        remainder = M - einsum('axz,bxz,bcu->acu', M, Q.conj(), Q)
        assert _tensor_norm(remainder) < 1e-10 * max(_tensor_norm(M), 1.0)
        assert Q.indices[0].dim == M.indices[0].dim


# ---------------------------------------------------------------------------
# Half contractions
# ---------------------------------------------------------------------------

class TestHalfContractions:
    """Tests for the factored half-applications of H."""

    def test_left_half_rank(self, heisenberg_L6):
        """`_left_half` is rank-4: (bra_left, internal, mpo_mid, phys_bra)."""
        mps, mpo = heisenberg_L6
        *_, lh, _ = _forward_pieces(mps, mpo)
        assert len(lh.indices) == 4

    def test_right_half_rank(self, heisenberg_L6):
        """`_right_half` is rank-4: (internal, mpo_mid, bra_right, phys_bra)."""
        mps, mpo = heisenberg_L6
        *_, rh = _forward_pieces(mps, mpo)
        assert len(rh.indices) == 4

    def test_halves_reconstruct_H_theta(self, heisenberg_L6):
        """Joining the two halves reproduces H·Θ with the full internal bond.

        This is the identity the whole factorization rests on: the complement is
        read off a cheap stand-in for the 2-site gradient, which is exact when
        `alpha` keeps every singular value.
        """
        mps, mpo = heisenberg_L6
        env_left, env_right, theta, _, lh, rh = _forward_pieces(mps, mpo)
        joined = einsum('aαpr,αpcu->acru', lh, rh)
        direct = einsum('aob,bdsv,oprs,pquv,cqd->acru',
                        env_left[FWD_SITE], theta, mpo[FWD_SITE], mpo[FWD_SITE + 1],
                        env_right[FWD_SITE + 1])
        diff = joined - direct
        assert _tensor_norm(diff) < 1e-10 * max(_tensor_norm(direct), 1.0)


# ---------------------------------------------------------------------------
# Projectors
# ---------------------------------------------------------------------------

class TestProjectComplement:
    """Tests for the discarded-space projections."""

    def test_left_discarded_orthogonal(self, heisenberg_L6):
        """A† applied to the discarded left half vanishes, as it must for A† A = I."""
        mps, mpo = heisenberg_L6
        _, _, _, A, lh, _ = _forward_pieces(mps, mpo)
        lh_disc = _project_complement_left(lh, A)
        overlap = einsum('abr,aαpr->bαp', A.conj(), lh_disc)
        assert _tensor_norm(overlap) < 1e-9

    def test_right_discarded_orthogonal(self, heisenberg_L6):
        """The neighbor applied to the discarded right half vanishes."""
        mps, mpo = heisenberg_L6
        _, _, _, _, _, rh = _forward_pieces(mps, mpo)
        rh_disc = _project_complement_right(rh, mps[FWD_SITE + 1])
        overlap = einsum('bcu,αpcu->αpb', mps[FWD_SITE + 1].conj(), rh_disc)
        assert _tensor_norm(overlap) < 1e-9

    def test_left_projection_idempotent(self, heisenberg_L6):
        """Projecting twice equals projecting once — the hallmark of a projector."""
        mps, mpo = heisenberg_L6
        _, _, _, A, lh, _ = _forward_pieces(mps, mpo)
        once = _project_complement_left(lh, A)
        twice = _project_complement_left(once, A)
        assert _tensor_norm(twice - once) < 1e-9 * max(_tensor_norm(once), 1.0)

    def test_right_projection_idempotent(self, heisenberg_L6):
        """Same for the right projector, built from the isometric factor B."""
        mps, mpo = heisenberg_L6
        _, _, _, B, _, rh = _backward_pieces(mps, mpo)
        once = _project_complement_right(rh, B)
        twice = _project_complement_right(once, B)
        assert _tensor_norm(twice - once) < 1e-9 * max(_tensor_norm(once), 1.0)

    def test_center_tensor_is_not_a_projector(self, heisenberg_L6):
        """The orthogonality center fails the idempotence the isometric factor passes.

        This is why the center end projects against the SVD factor rather than
        against its own site tensor: in a forward sweep `mps[i]` is the center,
        and `M M†` is a projector only for an isometry.
        """
        mps, mpo = heisenberg_L6
        _, _, _, _, lh, _ = _forward_pieces(mps, mpo)
        once = _project_complement_left(lh, mps[FWD_SITE])
        twice = _project_complement_left(once, mps[FWD_SITE])
        assert _tensor_norm(twice - once) > 1e-6 * _tensor_norm(once)

    def test_discarded_norms_bounded(self, heisenberg_L6):
        """Projecting out the kept space cannot increase either half's norm."""
        mps, mpo = heisenberg_L6
        _, _, _, A, lh, rh = _forward_pieces(mps, mpo)
        assert _tensor_norm(_project_complement_left(lh, A)) <= _tensor_norm(lh) + 1e-9
        assert (_tensor_norm(_project_complement_right(rh, mps[FWD_SITE + 1]))
                <= _tensor_norm(rh) + 1e-9)

    def test_complement_detected_where_it_exists(self, heisenberg_L6):
        """An unsaturated bond leaves a complement well above roundoff."""
        mps, mpo = heisenberg_L6
        _, _, _, A, lh, rh = _forward_pieces(mps, mpo)
        assert _has_complement(lh, _project_complement_left(lh, A))
        assert _has_complement(rh, _project_complement_right(rh, mps[FWD_SITE + 1]))

    def test_no_complement_on_saturated_bond(self, heisenberg_L4):
        """A saturated bond leaves nothing but roundoff behind.

        Every bond of the L=4 fixture already has dimension `dim_outer · d` on
        one end, so the projection there removes the whole half.
        """
        mps, mpo = heisenberg_L4
        _, _, _, _, _, rh = _forward_pieces(mps, mpo, site_i=1)
        assert not _has_complement(rh, _project_complement_right(rh, mps[2]))


# ---------------------------------------------------------------------------
# Complement vectors
# ---------------------------------------------------------------------------

class TestComplementVectors:
    """Tests for the SVD of the doubly-discarded 2-site tensor."""

    def _vectors(self, mps, mpo, site_i=FWD_SITE, k_expand=2, flow='<<'):
        _, _, _, A, lh, rh = _forward_pieces(mps, mpo, site_i)
        lh_disc = _project_complement_left(lh, A)
        rh_disc = _project_complement_right(rh, mps[site_i + 1])
        return _complement_vectors(lh_disc, rh_disc, k_expand, flow=flow)

    def test_ranks(self, heisenberg_L6):
        """N_L is (bra_left, k, phys) and N_R is (k, bra_right, phys)."""
        mps, mpo = heisenberg_L6
        N_L, N_R = self._vectors(mps, mpo)
        assert len(N_L.indices) == 3
        assert len(N_R.indices) == 3

    def test_k_bounded(self, heisenberg_L6):
        """The complement bond never exceeds `k_expand`."""
        mps, mpo = heisenberg_L6
        N_L, N_R = self._vectors(mps, mpo, k_expand=2)
        assert N_L.indices[1].dim <= 2
        assert N_R.indices[0].dim <= 2

    def test_shared_k_dim(self, heisenberg_L6):
        """Both factors come from one SVD, so they share the complement bond."""
        mps, mpo = heisenberg_L6
        N_L, N_R = self._vectors(mps, mpo)
        assert N_L.indices[1].dim == N_R.indices[0].dim

    def test_NR_right_isometric(self, heisenberg_L6):
        """N_R is right-isometric, which is what keeps the expanded neighbor isometric."""
        mps, mpo = heisenberg_L6
        _, N_R = self._vectors(mps, mpo)
        assert _isometry_error(N_R, left=False) < 1e-9

    def test_NL_left_isometric(self, heisenberg_L6):
        """N_L is left-isometric, the mirror requirement for a backward sweep."""
        mps, mpo = heisenberg_L6
        N_L, _ = self._vectors(mps, mpo, flow='>>')
        assert _isometry_error(N_L, left=True) < 1e-9

    def test_NR_orthogonal_to_kept_space(self, heisenberg_L6):
        """N_R adds genuinely new directions rather than repeating the neighbor."""
        mps, mpo = heisenberg_L6
        _, N_R = self._vectors(mps, mpo)
        cross = einsum('bcu,dcu->bd', mps[FWD_SITE + 1].conj(), N_R)
        assert _tensor_norm(cross) < 1e-9

    def test_k_above_available_rank(self, heisenberg_L6):
        """Asking for more directions than the complement holds yields only real ones.

        The surplus would otherwise come back as roundoff vectors that are not
        orthogonal to the kept space, which is exactly what must not happen.
        """
        mps, mpo = heisenberg_L6
        _, N_R = self._vectors(mps, mpo, k_expand=256)
        assert _isometry_error(N_R, left=False) < 1e-9
        cross = einsum('bcu,dcu->bd', mps[FWD_SITE + 1].conj(), N_R)
        assert _tensor_norm(cross) < 1e-9


# ---------------------------------------------------------------------------
# expand_forward
# ---------------------------------------------------------------------------

class TestExpandForward:
    """Tests for the public forward expansion."""

    def _expand(self, mps, mpo, site_i=FWD_SITE, k_expand=2, alpha=None):
        env_left, env_right = _envs_at(mps, mpo, site_i)
        M_i_exp, M_i1_exp, E_right_exp = expand_forward(
            mps[site_i], mps[site_i + 1],
            mpo[site_i], mpo[site_i + 1],
            env_left[site_i], env_right[site_i + 1],
            k_expand, alpha,
        )
        return env_left, env_right, M_i_exp, M_i1_exp, E_right_exp

    def test_bond_grows(self, heisenberg_L6):
        """The shared bond grows, by at most `k_expand`."""
        mps, mpo = heisenberg_L6
        original = mps[FWD_SITE].indices[1].dim
        _, _, M_i_exp, M_i1_exp, _ = self._expand(mps, mpo, k_expand=2)
        assert M_i_exp.indices[1].dim == M_i1_exp.indices[0].dim
        assert original < M_i_exp.indices[1].dim <= original + 2

    def test_other_dims_preserved(self, heisenberg_L6):
        """Only the shared bond changes; outer bonds and physical indices do not."""
        mps, mpo = heisenberg_L6
        _, _, M_i_exp, M_i1_exp, _ = self._expand(mps, mpo)
        assert M_i_exp.indices[0].dim == mps[FWD_SITE].indices[0].dim
        assert M_i_exp.indices[2].dim == mps[FWD_SITE].indices[2].dim
        assert M_i1_exp.indices[1].dim == mps[FWD_SITE + 1].indices[1].dim
        assert M_i1_exp.indices[2].dim == mps[FWD_SITE + 1].indices[2].dim

    def test_state_preserved(self, heisenberg_L6):
        """The expansion leaves Θ untouched — the requirement DMRG does not have."""
        mps, mpo = heisenberg_L6
        _, _, M_i_exp, M_i1_exp, _ = self._expand(mps, mpo)
        theta = build_bulk(mps[FWD_SITE], mps[FWD_SITE + 1])
        diff = build_bulk(M_i_exp, M_i1_exp) - theta
        assert _tensor_norm(diff) < 1e-10 * max(_tensor_norm(theta), 1.0)

    def test_neighbor_stays_right_isometric(self, heisenberg_L6):
        """The expanded neighbor is still a right isometry, as the next step assumes."""
        mps, mpo = heisenberg_L6
        _, _, _, M_i1_exp, _ = self._expand(mps, mpo)
        assert _isometry_error(M_i1_exp, left=False) < 1e-9

    def test_effective_energy_unchanged(self, heisenberg_L6):
        """The expanded effective Hamiltonian still reports `⟨ψ|H|ψ⟩`.

        Both pieces are under test here: the exact fill-in of the center tensor
        and the expanded environment, which keeps the cross blocks between the
        neighbor and the complement.
        """
        mps, mpo = heisenberg_L6
        env_left, _, M_i_exp, _, E_right_exp = self._expand(mps, mpo)
        energy = observe(mps, mpo)
        expanded = _local_energy(M_i_exp, mpo[FWD_SITE], env_left[FWD_SITE], E_right_exp)
        assert abs(expanded - energy) < 1e-9 * max(abs(energy), 1.0)

    def test_norm_unchanged(self, heisenberg_L6):
        """The center tensor keeps its norm, since no weight moves into the complement."""
        mps, mpo = heisenberg_L6
        _, _, M_i_exp, _, _ = self._expand(mps, mpo)
        before = inner_product(mps[FWD_SITE], mps[FWD_SITE])
        after = inner_product(M_i_exp, M_i_exp)
        assert abs(after - before) < 1e-10 * max(abs(before), 1.0)

    def test_alpha_truncation_still_preserves_state(self, heisenberg_L6):
        """A cheaper `alpha` degrades the complement, never the represented state."""
        mps, mpo = heisenberg_L6
        _, _, M_i_exp, M_i1_exp, _ = self._expand(mps, mpo, alpha=1)
        theta = build_bulk(mps[FWD_SITE], mps[FWD_SITE + 1])
        diff = build_bulk(M_i_exp, M_i1_exp) - theta
        assert _tensor_norm(diff) < 1e-10 * max(_tensor_norm(theta), 1.0)

    def test_leftmost_bond(self, heisenberg_L6):
        """Bond (0, 1) works although site 0 has a trivial left bond."""
        mps, mpo = heisenberg_L6
        env_left, _, M_i_exp, _, E_right_exp = self._expand(mps, mpo, site_i=0,
                                                            k_expand=4)
        assert M_i_exp.indices[0].dim == mps[0].indices[0].dim
        energy = observe(mps, mpo)
        expanded = _local_energy(M_i_exp, mpo[0], env_left[0], E_right_exp)
        assert abs(expanded - energy) < 1e-9 * max(abs(energy), 1.0)

    def test_saturated_bond_is_left_alone(self, heisenberg_L4):
        """With no complement to add, the expansion returns the state untouched."""
        mps, mpo = heisenberg_L4
        before = mps[1].indices[1].dim
        env_left, _, M_i_exp, M_i1_exp, E_right_exp = self._expand(
            mps, mpo, site_i=1, k_expand=4,
        )
        assert M_i_exp.indices[1].dim == before
        theta = build_bulk(mps[1], mps[2])
        diff = build_bulk(M_i_exp, M_i1_exp) - theta
        assert _tensor_norm(diff) < 1e-10 * max(_tensor_norm(theta), 1.0)
        # The environment handed back is still the one the 1-site step needs.
        energy = observe(mps, mpo)
        expanded = _local_energy(M_i_exp, mpo[1], env_left[1], E_right_exp)
        assert abs(expanded - energy) < 1e-9 * max(abs(energy), 1.0)

    def test_fill_center_matches_zero_padding(self, heisenberg_L6):
        """The exact fill-in agrees with padding the center tensor by zeros.

        They coincide because the complement is orthogonal to the kept space;
        the fill-in is used instead because it stays exact without relying on
        that orthogonality, and does not depend on `oplus` keeping zero blocks.
        """
        mps, mpo = heisenberg_L6
        _, _, _, M_i1_exp, _ = self._expand(mps, mpo)
        theta = build_bulk(mps[FWD_SITE], mps[FWD_SITE + 1])
        filled = _fill_center_forward(theta, M_i1_exp)
        # Padding by zeros amounts to projecting Θ onto the *old* basis and
        # embedding the result, which is what contracting with the unexpanded
        # neighbor and re-expanding does.
        padded = einsum('acru,bcu,bdv,edv->aer',
                        theta, mps[FWD_SITE + 1].conj(), mps[FWD_SITE + 1], M_i1_exp.conj())
        assert _tensor_norm(filled - padded) < 1e-10 * max(_tensor_norm(filled), 1.0)


# ---------------------------------------------------------------------------
# expand_backward
# ---------------------------------------------------------------------------

class TestExpandBackward:
    """Tests for the public backward expansion."""

    def _expand(self, mps, mpo, site_i=BWD_SITE, k_expand=2, alpha=None):
        env_left, env_right = _envs_at(mps, mpo, site_i)
        M_i_exp, M_im1_exp, E_left_exp = expand_backward(
            mps[site_i], mps[site_i - 1],
            mpo[site_i], mpo[site_i - 1],
            env_left[site_i - 1], env_right[site_i],
            k_expand, alpha,
        )
        return env_left, env_right, M_i_exp, M_im1_exp, E_left_exp

    def test_bond_grows(self, heisenberg_L6):
        """The shared bond grows, by at most `k_expand`."""
        mps, mpo = heisenberg_L6
        original = mps[BWD_SITE].indices[0].dim
        _, _, M_i_exp, M_im1_exp, _ = self._expand(mps, mpo, k_expand=2)
        assert M_i_exp.indices[0].dim == M_im1_exp.indices[1].dim
        assert original < M_i_exp.indices[0].dim <= original + 2

    def test_state_preserved(self, heisenberg_L6):
        """Θ over bond (i-1, i) is unchanged by the expansion."""
        mps, mpo = heisenberg_L6
        _, _, M_i_exp, M_im1_exp, _ = self._expand(mps, mpo)
        theta = build_bulk(mps[BWD_SITE - 1], mps[BWD_SITE])
        diff = build_bulk(M_im1_exp, M_i_exp) - theta
        assert _tensor_norm(diff) < 1e-10 * max(_tensor_norm(theta), 1.0)

    def test_neighbor_stays_left_isometric(self, heisenberg_L6):
        """The expanded left neighbor is still a left isometry."""
        mps, mpo = heisenberg_L6
        _, _, _, M_im1_exp, _ = self._expand(mps, mpo)
        assert _isometry_error(M_im1_exp, left=True) < 1e-9

    def test_effective_energy_unchanged(self, heisenberg_L6):
        """The expanded left environment reproduces `⟨ψ|H|ψ⟩`."""
        mps, mpo = heisenberg_L6
        _, env_right, M_i_exp, _, E_left_exp = self._expand(mps, mpo)
        energy = observe(mps, mpo)
        expanded = _local_energy(M_i_exp, mpo[BWD_SITE], E_left_exp,
                                 env_right[BWD_SITE])
        assert abs(expanded - energy) < 1e-9 * max(abs(energy), 1.0)

    def test_fill_center_matches_zero_padding(self, heisenberg_L6):
        """The backward fill-in likewise agrees with zero padding."""
        mps, mpo = heisenberg_L6
        _, _, _, M_im1_exp, _ = self._expand(mps, mpo)
        theta = build_bulk(mps[BWD_SITE - 1], mps[BWD_SITE])
        filled = _fill_center_backward(theta, M_im1_exp)
        padded = einsum('acru,abr,dbs,des->ecu',
                        theta, mps[BWD_SITE - 1].conj(), mps[BWD_SITE - 1], M_im1_exp.conj())
        assert _tensor_norm(filled - padded) < 1e-10 * max(_tensor_norm(filled), 1.0)

    def test_rightmost_bond(self, heisenberg_L6):
        """Bond (L-2, L-1) works although site L-1 has a trivial right bond."""
        mps, mpo = heisenberg_L6
        L = mps.L
        _, env_right, M_i_exp, _, E_left_exp = self._expand(mps, mpo, site_i=L - 1)
        energy = observe(mps, mpo)
        expanded = _local_energy(M_i_exp, mpo[L - 1], E_left_exp, env_right[L - 1])
        assert abs(expanded - energy) < 1e-9 * max(abs(energy), 1.0)

    def test_saturated_bond_is_left_alone(self, heisenberg_L4):
        """With no complement to add, the expansion returns the state untouched."""
        mps, mpo = heisenberg_L4
        before = mps[2].indices[0].dim
        _, _, M_i_exp, M_im1_exp, _ = self._expand(mps, mpo, site_i=2, k_expand=4)
        assert M_i_exp.indices[0].dim == before
        theta = build_bulk(mps[1], mps[2])
        diff = build_bulk(M_im1_exp, M_i_exp) - theta
        assert _tensor_norm(diff) < 1e-10 * max(_tensor_norm(theta), 1.0)


# ---------------------------------------------------------------------------
# Generic symmetry group
# ---------------------------------------------------------------------------

class TestNonAbelian:
    """The expansion must hold up under a generic symmetry group."""

    def test_forward_state_and_energy_preserved(self, heisenberg_su2_L4):
        """Expanding an SU(2) state changes neither Θ nor the effective energy."""
        mps, mpo = heisenberg_su2_L4
        env_left, env_right = _envs_at(mps, mpo, 1)
        theta = build_bulk(mps[1], mps[2])
        M_i_exp, M_i1_exp, E_right_exp = expand_forward(
            mps[1], mps[2], mpo[1], mpo[2],
            env_left[1], env_right[2], k_expand=2, alpha=None,
        )
        diff = build_bulk(M_i_exp, M_i1_exp) - theta
        assert _tensor_norm(diff) < 1e-10 * max(_tensor_norm(theta), 1.0)

        energy = observe(mps, mpo)
        expanded = _local_energy(M_i_exp, mpo[1], env_left[1], E_right_exp)
        assert abs(expanded - energy) < 1e-9 * max(abs(energy), 1.0)

    def test_backward_state_preserved(self, heisenberg_su2_L4):
        """Same for the backward direction."""
        mps, mpo = heisenberg_su2_L4
        env_left, env_right = _envs_at(mps, mpo, 2)
        theta = build_bulk(mps[1], mps[2])
        M_i_exp, M_im1_exp, _ = expand_backward(
            mps[2], mps[1], mpo[2], mpo[1],
            env_left[1], env_right[2], k_expand=2, alpha=None,
        )
        diff = build_bulk(M_im1_exp, M_i_exp) - theta
        assert _tensor_norm(diff) < 1e-10 * max(_tensor_norm(theta), 1.0)
