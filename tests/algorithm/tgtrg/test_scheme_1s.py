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


"""Tests for alice.algorithm.tgtrg.scheme_1s.

Three things are checked, each the premise of something downstream:

- The effective Hamiltonians are symmetric at every site, including the
  zero-site one, since the Lanczos recurrence has no meaning otherwise.
- The zero-site Hamiltonian is the *same* operator as the 1-site one seen
  through the split: `⟨C|H₀|C⟩` equals `⟨M|H_eff|M⟩` once the isometry is
  absorbed into the left environment. The back-evolution cancels the
  projector's double counting on the strength of that identity.
- The splits are exact and the absorbs invert them, so a step with `tau = 0`
  leaves the density matrix where it was.

The mid-chain cases are reached by walking the forward sweep by hand, which
exercises `split_forward`, `step_left_env`, and `absorb_forward` together on
the way to the site under test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from functools import partial
from nicole import einsum

from alice.algorithm.tgtrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
    step_left_env,
)
from alice.algorithm.tgtrg.krylov import inner_product
from alice.algorithm.tgtrg.scheme_1s import (
    absorb_backward,
    absorb_forward,
    evolve_1site,
    evolve_bond,
    matvec_0s,
    matvec_1s,
    split_backward,
    split_forward,
)

from .conftest import dense, energy_reference, thermal_state

KRYLOV = {'max_krylov': 40, 'tol': 1e-14}


def _bond_itag(rho, bond: int) -> str:
    """Return the itag ρ uses for the bond to the left of site `bond`."""
    return rho[bond].itags[0]


def _walk_to(rho, mpo, site):
    """Walk the forward sweep to `site`, returning the local update data.

    Returns
    -------
    tuple
        `(M, W, E_left, E_right)` for a 1-site update at `site`, with `M`
        carrying every bond tensor accumulated on the way.
    """
    env_right = Environment(rho.L)
    build_right_envs(rho, mpo, env_right)

    E_left = left_env_boundary(rho, mpo)
    M = rho[0]
    for i in range(site):
        A, C = split_forward(M, _bond_itag(rho, i + 1))
        E_left = step_left_env(E_left, A, mpo[i])
        M = absorb_forward(C, rho[i + 1])
    return M, mpo[site], E_left, env_right[site]


@pytest.fixture(params=[0, 1, 2, 3], ids=['site0', 'site1', 'site2', 'site3'])
def local_update(request, spinless_fermion_L4):
    """A 1-site update at each site of the L=4 chain, with ρ(β=0.1)."""
    mpo, spc, _ = spinless_fermion_L4
    rho = thermal_state(mpo, spc, 0.1)
    return rho, mpo, _walk_to(rho, mpo, request.param)


@pytest.fixture
def bond_update(spinless_fermion_L4):
    """A zero-site update on the first bond, with the 1-site data it came from."""
    mpo, spc, _ = spinless_fermion_L4
    rho = thermal_state(mpo, spc, 0.1)
    env_right = Environment(rho.L)
    build_right_envs(rho, mpo, env_right)

    E_left = left_env_boundary(rho, mpo)
    A, C = split_forward(rho[0], _bond_itag(rho, 1))
    return rho, mpo, C, step_left_env(E_left, A, mpo[0]), env_right[0]


# ---------------------------------------------------------------------------
# matvec_1s
# ---------------------------------------------------------------------------

class TestMatvec1s:
    """Tests for the 1-site effective Hamiltonian."""

    def test_preserves_the_index_structure(self, local_update):
        """H_eff|M⟩ is interchangeable with |M⟩, as the propagator needs."""
        _, _, (M, W, E_left, E_right) = local_update
        HM = matvec_1s(M, W, E_left, E_right)
        assert HM.itags == M.itags
        assert [i.dim for i in HM.indices] == [i.dim for i in M.indices]
        assert [i.direction for i in HM.indices] == [i.direction for i in M.indices]

    def test_is_symmetric(self, local_update):
        """H_eff is symmetric at every site, not only at the boundaries."""
        _, _, (M, W, E_left, E_right) = local_update
        H = dense(partial(matvec_1s, W=W, E_left=E_left, E_right=E_right), M)
        assert np.abs(H - H.T).max() < 1e-11 * max(1.0, np.abs(H).max())

    def test_is_linear(self, local_update):
        """The map is linear, which `dense` already assumes when rendering it."""
        _, _, (M, W, E_left, E_right) = local_update
        lhs = matvec_1s(M * 2.0 + M, W, E_left, E_right)
        rhs = matvec_1s(M, W, E_left, E_right) * 3.0
        assert (lhs - rhs).norm() < 1e-11 * max(1.0, lhs.norm())

    def test_expectation_is_the_energy(self, local_update):
        """⟨M|H_eff|M⟩ is Tr[ρ̂Hρ̂] at every site along the sweep.

        The walk to each site only moves norm between the isometry and the
        bond tensor, so the closed contraction is site-independent.
        """
        rho, mpo, (M, W, E_left, E_right) = local_update
        value = inner_product(M, matvec_1s(M, W, E_left, E_right))
        assert value.real == pytest.approx(energy_reference(rho, mpo), rel=1e-9)


# ---------------------------------------------------------------------------
# matvec_0s
# ---------------------------------------------------------------------------

class TestMatvec0s:
    """Tests for the zero-site effective Hamiltonian."""

    def test_preserves_the_index_structure(self, bond_update):
        """H₀|C⟩ is interchangeable with |C⟩."""
        _, _, C, E_left, E_right = bond_update
        HC = matvec_0s(C, E_left, E_right)
        assert HC.itags == C.itags
        assert [i.dim for i in HC.indices] == [i.dim for i in C.indices]
        assert [i.direction for i in HC.indices] == [i.direction for i in C.indices]

    def test_is_symmetric(self, bond_update):
        """H₀ is symmetric, as its own Lanczos call requires."""
        _, _, C, E_left, E_right = bond_update
        H = dense(partial(matvec_0s, E_left=E_left, E_right=E_right), C)
        assert np.abs(H - H.T).max() < 1e-11 * max(1.0, np.abs(H).max())

    def test_agrees_with_the_one_site_hamiltonian(self, bond_update):
        """⟨C|H₀|C⟩ equals ⟨M|H_eff|M⟩: the same operator through the split.

        On this identity rests the back-evolution's exact cancellation of
        what the tangent-space projector counted twice.
        """
        rho, mpo, C, E_left, E_right = bond_update
        value = inner_product(C, matvec_0s(C, E_left, E_right))
        assert value.real == pytest.approx(energy_reference(rho, mpo), rel=1e-9)


# ---------------------------------------------------------------------------
# Evolution steps
# ---------------------------------------------------------------------------

class TestEvolve1Site:
    """Tests for the cooled site tensor."""

    def test_zero_step_changes_nothing(self, local_update):
        """exp(0) is the identity, and removes no scale."""
        _, _, (M, W, E_left, E_right) = local_update
        out, log_scale, _ = evolve_1site(M, W, E_left, E_right, 0.0, KRYLOV)
        assert log_scale == pytest.approx(0.0)
        assert (out - M).norm() < 1e-12

    def test_first_order_in_a_small_step(self, local_update):
        """For small tau the step agrees with `(1 - tau H_eff)M` to O(tau²)."""
        _, _, (M, W, E_left, E_right) = local_update
        HM = matvec_1s(M, W, E_left, E_right)

        deviations = []
        for tau in (1e-2, 1e-3):
            out, log_scale, _ = evolve_1site(M, W, E_left, E_right, tau, KRYLOV)
            full = out * float(np.exp(log_scale))
            deviations.append((full - (M - HM * tau)).norm() / M.norm())
        # Shrinking tau tenfold must cut a second-order deviation by ~100.
        assert deviations[0] / deviations[1] > 50.0

    def test_cooling_then_heating_restores_the_tensor(self, local_update):
        """The step is invertible, since H_eff does not depend on tau."""
        _, _, (M, W, E_left, E_right) = local_update
        cooled, log_c, _ = evolve_1site(M, W, E_left, E_right, 0.2, KRYLOV)
        back, log_b, _ = evolve_1site(cooled, W, E_left, E_right, -0.2, KRYLOV)
        restored = back * float(np.exp(log_c + log_b))
        assert (restored - M).norm() < 1e-10 * M.norm()

    def test_lowers_the_energy(self, local_update):
        """Cooling pushes weight toward the low end of the local spectrum."""
        _, _, (M, W, E_left, E_right) = local_update
        mv = partial(matvec_1s, W=W, E_left=E_left, E_right=E_right)

        def rayleigh(v):
            return inner_product(v, mv(v)).real / inner_product(v, v).real

        cooled, _, _ = evolve_1site(M, W, E_left, E_right, 0.5, KRYLOV)
        assert rayleigh(cooled) < rayleigh(M)

    def test_reports_a_small_krylov_error(self, local_update):
        """A generous subspace converges on this chain."""
        _, _, (M, W, E_left, E_right) = local_update
        _, _, error = evolve_1site(M, W, E_left, E_right, 0.2, KRYLOV)
        assert error < 1e-8


class TestEvolveBond:
    """Tests for the evolved bond tensor."""

    def test_zero_step_changes_nothing(self, bond_update):
        """The same identity check, on the bond."""
        _, _, C, E_left, E_right = bond_update
        out, log_scale, _ = evolve_bond(C, E_left, E_right, 0.0, KRYLOV)
        assert log_scale == pytest.approx(0.0)
        assert (out - C).norm() < 1e-12

    def test_first_order_in_a_small_step(self, bond_update):
        """exp(-tau H₀) C agrees with `(1 - tau H₀)C` to O(tau²)."""
        _, _, C, E_left, E_right = bond_update
        HC = matvec_0s(C, E_left, E_right)

        deviations = []
        for tau in (1e-2, 1e-3):
            out, log_scale, _ = evolve_bond(C, E_left, E_right, tau, KRYLOV)
            full = out * float(np.exp(log_scale))
            deviations.append((full - (C - HC * tau)).norm() / C.norm())
        assert deviations[0] / deviations[1] > 50.0

    def test_negative_step_heats(self, bond_update):
        """A negated increment raises the local energy, as the splitting wants."""
        _, _, C, E_left, E_right = bond_update
        mv = partial(matvec_0s, E_left=E_left, E_right=E_right)

        def rayleigh(v):
            return inner_product(v, mv(v)).real / inner_product(v, v).real

        heated, _, _ = evolve_bond(C, E_left, E_right, -0.5, KRYLOV)
        assert rayleigh(heated) > rayleigh(C)


# ---------------------------------------------------------------------------
# Splits and absorbs
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
    """Tests for the left-isometric split."""

    def test_isometry_condition(self, local_update):
        """A†A is the identity on the new bond."""
        _, _, (M, _, _, _) = local_update
        A, _ = split_forward(M, 'NEW')
        assert _is_identity(einsum('abrs,acrs->bc', A.conj(), A))

    def test_reconstructs_the_site_tensor(self, local_update):
        """A·C is M exactly: the 1-site split discards nothing."""
        _, _, (M, _, _, _) = local_update
        A, C = split_forward(M, 'NEW')
        assert (einsum('abrs,bd->adrs', A, C) - M).norm() < 1e-11 * M.norm()

    def test_new_bond_itag_is_applied(self, local_update):
        """Both factors carry the requested itag on the shared bond."""
        _, _, (M, _, _, _) = local_update
        A, C = split_forward(M, 'NEW')
        assert A.itags[1] == 'NEW'
        assert C.itags[0] == 'NEW'

    def test_outer_itags_are_preserved(self, local_update):
        """Everything but the new bond keeps the layout the sweep expects."""
        _, _, (M, _, _, _) = local_update
        A, C = split_forward(M, 'NEW')
        assert A.itags[0] == M.itags[0]
        assert A.itags[2:] == M.itags[2:]
        assert C.itags[1] == M.itags[1]


class TestSplitBackward:
    """Tests for the right-isometric split."""

    def test_isometry_condition(self, local_update):
        """BB† is the identity on the new bond."""
        _, _, (M, _, _, _) = local_update
        _, B = split_backward(M, 'NEW')
        assert _is_identity(einsum('abrs,cbrs->ac', B.conj(), B))

    def test_reconstructs_the_site_tensor(self, local_update):
        """C·B is M exactly."""
        _, _, (M, _, _, _) = local_update
        C, B = split_backward(M, 'NEW')
        assert (einsum('ab,bdrs->adrs', C, B) - M).norm() < 1e-11 * M.norm()

    def test_new_bond_itag_is_applied(self, local_update):
        """Both factors carry the requested itag on the shared bond."""
        _, _, (M, _, _, _) = local_update
        C, B = split_backward(M, 'NEW')
        assert C.itags[1] == 'NEW'
        assert B.itags[0] == 'NEW'

    def test_outer_itags_are_preserved(self, local_update):
        """Everything but the new bond keeps its tag."""
        _, _, (M, _, _, _) = local_update
        C, B = split_backward(M, 'NEW')
        assert C.itags[0] == M.itags[0]
        assert B.itags[1] == M.itags[1]
        assert B.itags[2:] == M.itags[2:]


class TestAbsorb:
    """Tests for folding a bond tensor back into a site."""

    def test_forward_absorb_layout(self, spinless_fermion_L4):
        """The absorbed site keeps the MPO axis order and the new bond."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        _, C = split_forward(rho[0], 'NEW')
        M = absorb_forward(C, rho[1])
        assert M.itags == ('NEW',) + rho[1].itags[1:]

    def test_backward_absorb_layout(self, spinless_fermion_L4):
        """The mirror case, with the permutation back to MPO order."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        C, _ = split_backward(rho[1], 'NEW')
        M = absorb_backward(rho[0], C)
        assert M.itags == (rho[0].itags[0], 'NEW') + rho[0].itags[2:]

    def test_forward_absorb_is_linear_in_the_bond(self, spinless_fermion_L4):
        """Scaling the bond tensor scales the absorbed site."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        _, C = split_forward(rho[0], 'NEW')
        lhs = absorb_forward(C * 2.0, rho[1])
        rhs = absorb_forward(C, rho[1]) * 2.0
        assert (lhs - rhs).norm() < 1e-12 * max(1.0, lhs.norm())


# ---------------------------------------------------------------------------
# The identity the projector splitting rests on
# ---------------------------------------------------------------------------

class TestProjectorSplitting:
    """The zero-site term is exactly what the site terms count twice."""

    def test_bond_term_is_the_projection_of_the_site_term(self, spinless_fermion_L4):
        """`A·(H₀C)` equals `P_A(H_eff M)` for `M = A·C`, exactly.

        The generator of the bond step is the site generator projected onto
        the isometry that has just been frozen. That identity is the whole
        content of the splitting: the forward sweep applies `H_eff` on both
        sides of each bond, and `-H₀` on the bond subtracts the overlap once.
        A pair of steps on its own is therefore *not* `exp(-tau H)`; only the
        full sweep is, which is where that is tested.
        """
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        env_right = Environment(rho.L)
        build_right_envs(rho, mpo, env_right)
        E_left = left_env_boundary(rho, mpo)

        M = rho[0]
        A, C = split_forward(M, _bond_itag(rho, 1))
        E_left_new = step_left_env(E_left, A, mpo[0])

        bond_term = einsum('abrs,bd->adrs', A, matvec_0s(C, E_left_new,
                                                         env_right[0]))
        site_term = matvec_1s(M, mpo[0], E_left, env_right[0])
        projected = einsum('abrs,bd->adrs', A,
                           einsum('abrs,adrs->bd', A.conj(), site_term))

        assert (bond_term - projected).norm() < 1e-11 * site_term.norm()

    def test_bond_term_carries_the_same_energy(self, spinless_fermion_L4):
        """The projection keeps the full expectation value at the boundary.

        At site 0 the frozen isometry spans the whole left side, so the
        projector acts trivially on the energy and `⟨C|H₀|C⟩` is the total.
        """
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 0.1)
        env_right = Environment(rho.L)
        build_right_envs(rho, mpo, env_right)
        E_left = left_env_boundary(rho, mpo)

        A, C = split_forward(rho[0], _bond_itag(rho, 1))
        E_left_new = step_left_env(E_left, A, mpo[0])
        bond_energy = inner_product(C, matvec_0s(C, E_left_new, env_right[0])).real
        site_energy = inner_product(
            rho[0], matvec_1s(rho[0], mpo[0], E_left, env_right[0])).real

        assert bond_energy == pytest.approx(site_energy, rel=1e-10)
