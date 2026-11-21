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


"""Tests for alice.algorithm.tgtrg.scheme_2s.

The 2-site Hamiltonian is checked the same way as the 1-site one — symmetry,
and a closed contraction equal to `Tr[ρ̂Hρ̂]` — plus the two things only this
scheme has: splits that may truncate, and a bond that can therefore grow.

Bond growth is the reason the cooling driver defaults to this scheme, so it
is tested directly: an untruncated split of Θ leaves a wider bond than the
one it came from, and `nkeep` bounds that width.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch
from nicole import einsum

from alice.algorithm.tgtrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
)
from alice.algorithm.tgtrg.krylov import inner_product
from alice.algorithm.tgtrg.scheme_2s import (
    build_bulk,
    discarded_weight,
    evolve_2site,
    matvec_2s,
    split_backward,
    split_forward,
)

from .conftest import dense, energy_reference, thermal_state

KRYLOV = {'max_krylov': 40, 'tol': 1e-14}


@pytest.fixture
def pair_update(spinless_fermion_L4):
    """A 2-site update on the leftmost pair of the L=4 chain, with ρ(β=0.1)."""
    mpo, spc, _ = spinless_fermion_L4
    rho = thermal_state(mpo, spc, 0.1)
    env_right = Environment(rho.L)
    build_right_envs(rho, mpo, env_right)
    return (rho, mpo, build_bulk(rho[0], rho[1]), mpo[0], mpo[1],
            left_env_boundary(rho, mpo), env_right[1])


# ---------------------------------------------------------------------------
# build_bulk
# ---------------------------------------------------------------------------

class TestBuildBulk:
    """Tests for the two-site bond tensor."""

    def test_axis_layout(self, pair_update):
        """Θ carries the outer bonds and all four physical axes, in order."""
        rho, _, theta, *_ = pair_update
        assert len(theta.indices) == 6
        assert theta.itags == (rho[0].itags[0], rho[1].itags[1],
                               rho[0].itags[2], rho[0].itags[3],
                               rho[1].itags[2], rho[1].itags[3])

    def test_dimensions(self, pair_update):
        """The internal bond is gone; every other dimension is inherited."""
        rho, _, theta, *_ = pair_update
        dims = [i.dim for i in theta.indices]
        assert dims == [rho[0].indices[0].dim, rho[1].indices[1].dim,
                        rho[0].indices[2].dim, rho[0].indices[3].dim,
                        rho[1].indices[2].dim, rho[1].indices[3].dim]

    def test_is_bilinear(self, pair_update):
        """Scaling either site scales Θ."""
        rho, _, theta, *_ = pair_update
        assert (build_bulk(rho[0] * 2.0, rho[1]) - theta * 2.0).norm() < 1e-12
        assert (build_bulk(rho[0], rho[1] * 2.0) - theta * 2.0).norm() < 1e-12


# ---------------------------------------------------------------------------
# matvec_2s
# ---------------------------------------------------------------------------

class TestMatvec2s:
    """Tests for the 2-site effective Hamiltonian."""

    def test_preserves_the_index_structure(self, pair_update):
        """H_eff|Θ⟩ is interchangeable with |Θ⟩."""
        _, _, theta, W_i, W_i1, E_left, E_right = pair_update
        out = matvec_2s(theta, W_i, W_i1, E_left, E_right)
        assert out.itags == theta.itags
        assert [i.dim for i in out.indices] == [i.dim for i in theta.indices]
        assert [i.direction for i in out.indices] == [i.direction
                                                      for i in theta.indices]

    def test_is_symmetric(self, pair_update):
        """H_eff is symmetric, as its Lanczos call requires."""
        _, _, theta, W_i, W_i1, E_left, E_right = pair_update
        H = dense(partial(matvec_2s, W_i=W_i, W_i1=W_i1, E_left=E_left,
                          E_right=E_right), theta)
        assert np.abs(H - H.T).max() < 1e-11 * max(1.0, np.abs(H).max())

    def test_expectation_is_the_energy(self, pair_update):
        """⟨Θ|H_eff|Θ⟩ is Tr[ρ̂Hρ̂], as for the 1-site Hamiltonian."""
        rho, mpo, theta, W_i, W_i1, E_left, E_right = pair_update
        value = inner_product(theta, matvec_2s(theta, W_i, W_i1, E_left, E_right))
        assert value.real == pytest.approx(energy_reference(rho, mpo), rel=1e-9)

    def test_agrees_with_the_one_site_hamiltonian(self, spinless_fermion_L4):
        """The two schemes see the same operator on the same density matrix.

        A 2-site update spans sites 0 and 1 while a 1-site update at site 0
        spans only site 0, but both close over the same `H`, so the closed
        expectation value must coincide.
        """
        from alice.algorithm.tgtrg.scheme_1s import matvec_1s

        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        env_right = Environment(rho.L)
        build_right_envs(rho, mpo, env_right)
        E_left = left_env_boundary(rho, mpo)

        theta = build_bulk(rho[0], rho[1])
        two_site = inner_product(
            theta, matvec_2s(theta, mpo[0], mpo[1], E_left, env_right[1])).real
        one_site = inner_product(
            rho[0], matvec_1s(rho[0], mpo[0], E_left, env_right[0])).real
        assert two_site == pytest.approx(one_site, rel=1e-9)

    def test_holds_for_the_spinful_chain(self, spinful_fermion_L4):
        """Two symmetry groups change nothing structurally."""
        mpo, spc, _ = spinful_fermion_L4
        rho = thermal_state(mpo, spc, 0.1, order=3)
        env_right = Environment(rho.L)
        build_right_envs(rho, mpo, env_right)

        theta = build_bulk(rho[0], rho[1])
        value = inner_product(
            theta, matvec_2s(theta, mpo[0], mpo[1], left_env_boundary(rho, mpo),
                             env_right[1]))
        assert value.real == pytest.approx(energy_reference(rho, mpo), rel=1e-9)


# ---------------------------------------------------------------------------
# evolve_2site
# ---------------------------------------------------------------------------

class TestEvolve2Site:
    """Tests for the cooled bond tensor."""

    def test_zero_step_rebuilds_theta(self, pair_update):
        """exp(0) returns `build_bulk`'s own output untouched."""
        rho, mpo, theta, W_i, W_i1, E_left, E_right = pair_update
        out, log_scale, _ = evolve_2site(rho[0], rho[1], W_i, W_i1, E_left,
                                         E_right, 0.0, KRYLOV)
        assert log_scale == pytest.approx(0.0)
        assert (out - theta).norm() < 1e-12

    def test_first_order_in_a_small_step(self, pair_update):
        """For small tau the step agrees with `(1 - tau H_eff)Θ` to O(tau²)."""
        rho, _, theta, W_i, W_i1, E_left, E_right = pair_update
        HT = matvec_2s(theta, W_i, W_i1, E_left, E_right)

        deviations = []
        for tau in (1e-2, 1e-3):
            out, log_scale, _ = evolve_2site(rho[0], rho[1], W_i, W_i1, E_left,
                                             E_right, tau, KRYLOV)
            full = out * float(np.exp(log_scale))
            deviations.append((full - (theta - HT * tau)).norm() / theta.norm())
        assert deviations[0] / deviations[1] > 50.0

    def test_lowers_the_energy(self, pair_update):
        """Cooling moves weight toward the low end of the local spectrum."""
        rho, _, theta, W_i, W_i1, E_left, E_right = pair_update
        mv = partial(matvec_2s, W_i=W_i, W_i1=W_i1, E_left=E_left, E_right=E_right)

        def rayleigh(v):
            return inner_product(v, mv(v)).real / inner_product(v, v).real

        cooled, _, _ = evolve_2site(rho[0], rho[1], W_i, W_i1, E_left, E_right,
                                    0.5, KRYLOV)
        assert rayleigh(cooled) < rayleigh(theta)

    def test_cooling_then_heating_restores_theta(self, pair_update):
        """The step is invertible, H_eff being tau-independent."""
        rho, _, theta, W_i, W_i1, E_left, E_right = pair_update
        cooled, log_c, _ = evolve_2site(rho[0], rho[1], W_i, W_i1, E_left,
                                        E_right, 0.2, KRYLOV)
        # Split and rebuild so the reverse step starts from site tensors.
        M_i, M_i1 = split_forward(cooled, 'NEW', None)
        back, log_b, _ = evolve_2site(M_i, M_i1, W_i, W_i1, E_left, E_right,
                                      -0.2, KRYLOV)
        restored = back * float(np.exp(log_c + log_b))
        assert (restored - theta).norm() < 1e-9 * theta.norm()

    def test_stays_real(self, pair_update):
        """Nothing in the 2-site path promotes a real density matrix."""
        rho, _, _, W_i, W_i1, E_left, E_right = pair_update
        out, _, _ = evolve_2site(rho[0], rho[1], W_i, W_i1, E_left, E_right,
                                 0.3, KRYLOV)
        assert not out.dtype.is_complex


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def _is_identity(X) -> bool:
    """Check that a 2nd-order tensor is the identity on every surviving block."""
    for block in X.data.values():
        if block.shape[0] != block.shape[1]:
            return False
        if not torch.allclose(block, torch.eye(block.shape[0], dtype=block.dtype),
                              atol=1e-11):
            return False
    return True


class TestSplitForward:
    """Tests for the forward SVD split."""

    def test_isometry_condition(self, pair_update):
        """`M_i†M_i` is the identity on the new bond."""
        _, _, theta, *_ = pair_update
        M_i, _ = split_forward(theta, 'NEW', None)
        assert _is_identity(einsum('abrs,acrs->bc', M_i.conj(), M_i))

    def test_reconstructs_theta(self, pair_update):
        """An untruncated split loses nothing."""
        _, _, theta, *_ = pair_update
        M_i, M_i1 = split_forward(theta, 'NEW', None)
        assert (build_bulk(M_i, M_i1) - theta).norm() < 1e-11 * theta.norm()

    def test_axis_layout(self, pair_update):
        """Both factors come out in MPO axis order, tagged on the new bond."""
        rho, _, theta, *_ = pair_update
        M_i, M_i1 = split_forward(theta, 'NEW', None)
        assert M_i.itags == (rho[0].itags[0], 'NEW') + rho[0].itags[2:]
        assert M_i1.itags == ('NEW', rho[1].itags[1]) + rho[1].itags[2:]

    def test_bond_can_grow(self, spinless_fermion_L4):
        """The split is how a cooling run adds bond dimension.

        Starting from `ρ(τ₀)` at the Hamiltonian's bond dimension, growth is
        the entire reason to prefer this scheme over the 1-site one. The
        density matrix here is compressed first so the mid-chain bond has
        room to widen; at the ends it is capped by the physical dimension
        either way.
        """
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        rho.compact({'nkeep': 2})

        theta = build_bulk(rho[1], rho[2])
        M_i, _ = split_forward(theta, 'NEW', None)
        assert M_i.indices[1].dim > rho[1].indices[1].dim

    def test_nkeep_bounds_the_bond(self, pair_update):
        """`nkeep` caps the new bond dimension."""
        _, _, theta, *_ = pair_update
        M_i, _ = split_forward(theta, 'NEW', {'nkeep': 3})
        assert M_i.indices[1].dim <= 3

    def test_truncation_loses_weight(self, pair_update):
        """A truncated split no longer reconstructs Θ exactly."""
        _, _, theta, *_ = pair_update
        M_i, M_i1 = split_forward(theta, 'NEW', {'nkeep': 2})
        assert (build_bulk(M_i, M_i1) - theta).norm() > 1e-8 * theta.norm()


class TestSplitBackward:
    """Tests for the backward SVD split."""

    def test_isometry_condition(self, pair_update):
        """`M_{i+1}M_{i+1}†` is the identity on the new bond."""
        _, _, theta, *_ = pair_update
        _, M_i1 = split_backward(theta, 'NEW', None)
        assert _is_identity(einsum('abrs,cbrs->ac', M_i1.conj(), M_i1))

    def test_reconstructs_theta(self, pair_update):
        """An untruncated split loses nothing."""
        _, _, theta, *_ = pair_update
        M_i, M_i1 = split_backward(theta, 'NEW', None)
        assert (build_bulk(M_i, M_i1) - theta).norm() < 1e-11 * theta.norm()

    def test_axis_layout(self, pair_update):
        """Both factors come out in MPO axis order, tagged on the new bond."""
        rho, _, theta, *_ = pair_update
        M_i, M_i1 = split_backward(theta, 'NEW', None)
        assert M_i.itags == (rho[0].itags[0], 'NEW') + rho[0].itags[2:]
        assert M_i1.itags == ('NEW', rho[1].itags[1]) + rho[1].itags[2:]

    def test_agrees_with_the_forward_split_up_to_the_gauge(self, pair_update):
        """The two splits are the same decomposition, read from either side."""
        _, _, theta, *_ = pair_update
        fwd = build_bulk(*split_forward(theta, 'NEW', None))
        bwd = build_bulk(*split_backward(theta, 'NEW', None))
        assert (fwd - bwd).norm() < 1e-11 * theta.norm()


# ---------------------------------------------------------------------------
# discarded_weight
# ---------------------------------------------------------------------------

class TestDiscardedWeight:
    """Tests for the truncation diagnostic."""

    def test_zero_without_truncation(self, pair_update):
        """No `trunc` means no cut, reported without touching the SVD."""
        _, _, theta, *_ = pair_update
        assert discarded_weight(theta, None) == 0.0

    def test_zero_when_nothing_is_cut(self, pair_update):
        """A cap above the full rank discards nothing."""
        _, _, theta, *_ = pair_update
        assert discarded_weight(theta, {'nkeep': 1000}) < 1e-12

    def test_positive_when_the_bond_is_cut(self, pair_update):
        """A tight cap reports a loss."""
        _, _, theta, *_ = pair_update
        assert discarded_weight(theta, {'nkeep': 2}) > 0.0

    def test_grows_as_the_cap_tightens(self, pair_update):
        """Keeping fewer singular values cannot discard less."""
        _, _, theta, *_ = pair_update
        weights = [discarded_weight(theta, {'nkeep': n}) for n in (6, 4, 2)]
        assert weights == sorted(weights)

    def test_tracks_the_error_of_the_split(self, pair_update):
        """The reported weight matches the squared error of the actual split.

        For a normalized Θ the discarded weight is `1 - ‖kept‖²`, so it
        equals the relative squared norm of what the split left behind.
        """
        _, _, theta, *_ = pair_update
        trunc = {'nkeep': 3}
        M_i, M_i1 = split_forward(theta, 'NEW', trunc)
        lost = (build_bulk(M_i, M_i1) - theta).norm() ** 2 / theta.norm() ** 2
        assert discarded_weight(theta, trunc) == pytest.approx(lost, rel=1e-6)
