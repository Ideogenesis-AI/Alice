# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Alice library.
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


"""Ground-state energy tests for the Heisenberg spin chain."""

from __future__ import annotations

import math

import pytest

from alice.network import MPS, MPO, observe

# Tolerance for comparing the observed energy against the iterative-diag reference.
# Both quantities are computed from the same exact arithmetic (no truncation in
# the MPS, no approximation in the MPO), so agreement to machine precision is expected.
_ATOL = 1e-10


@pytest.mark.slow
class TestHeisenberg:
    """MPS-MPO energy test for the spin-1/2 Heisenberg chain (U1 symmetry)."""

    def test_energy(self, spin_chain):
        """observe(mps, mpo) must reproduce the iterative-diag ground-state energy."""
        mps, mpo, E_gs = spin_chain
        E_obs = observe(mps, mpo)
        assert math.isclose(E_obs, E_gs, abs_tol=_ATOL), (
            f"Heisenberg energy mismatch: observe={E_obs:.12f}, ref={E_gs:.12f}, "
            f"diff={abs(E_obs - E_gs):.3e}"
        )

    def test_mps_normalized(self, spin_chain):
        """The ground-state MPS from iterative diagonalization must be unit-normalized."""
        mps, _, _ = spin_chain
        assert math.isclose(mps.norm(), 1.0, abs_tol=_ATOL), (
            f"Heisenberg MPS norm = {mps.norm():.12f}, expected 1"
        )

    def test_energy_invariant_under_mps_canonical(self, spin_chain):
        """Energy is unchanged through a canonical sweep of the MPS: center 0 → L-1."""
        mps, mpo, E_gs = spin_chain
        L = len(mps)
        mps_c = MPS([mps[i].clone() for i in range(L)], center=mps.center)
        mps_c.canonical(0)
        E0 = observe(mps_c, mpo)
        assert math.isclose(E0, E_gs, abs_tol=_ATOL), (
            f"Heisenberg energy changed after MPS.canonical(0): "
            f"observe={E0:.12f}, ref={E_gs:.12f}, diff={abs(E0 - E_gs):.3e}"
        )
        mps_c.canonical(L - 1)
        EN = observe(mps_c, mpo)
        assert math.isclose(EN, E_gs, abs_tol=_ATOL), (
            f"Heisenberg energy changed after MPS.canonical(L-1): "
            f"observe={EN:.12f}, ref={E_gs:.12f}, diff={abs(EN - E_gs):.3e}"
        )

    def test_energy_invariant_under_mpo_canonical(self, spin_chain):
        """Energy is unchanged through a canonical sweep of the MPO: center 0 → L-1."""
        mps, mpo, E_gs = spin_chain
        L = len(mpo)
        mpo_c = MPO([mpo[i].clone() for i in range(L)], center=mpo.center)
        mpo_c.canonical(0)
        E0 = observe(mps, mpo_c)
        assert math.isclose(E0, E_gs, abs_tol=_ATOL), (
            f"Heisenberg energy changed after MPO.canonical(0): "
            f"observe={E0:.12f}, ref={E_gs:.12f}, diff={abs(E0 - E_gs):.3e}"
        )
        mpo_c.canonical(L - 1)
        EN = observe(mps, mpo_c)
        assert math.isclose(EN, E_gs, abs_tol=_ATOL), (
            f"Heisenberg energy changed after MPO.canonical(L-1): "
            f"observe={EN:.12f}, ref={E_gs:.12f}, diff={abs(EN - E_gs):.3e}"
        )


@pytest.mark.slow
class TestHeisenbergSU2:
    """MPS-MPO energy test for the spin-1/2 Heisenberg chain (SU2 symmetry)."""

    def test_energy(self, spin_chain_su2):
        """observe(mps, mpo) must reproduce the iterative-diag ground-state energy."""
        mps, mpo, E_gs = spin_chain_su2
        E_obs = observe(mps, mpo)
        assert math.isclose(E_obs, E_gs, abs_tol=_ATOL), (
            f"Heisenberg (SU2) energy mismatch: observe={E_obs:.12f}, ref={E_gs:.12f}, "
            f"diff={abs(E_obs - E_gs):.3e}"
        )

    def test_mps_normalized(self, spin_chain_su2):
        """The ground-state MPS from iterative diagonalization must be unit-normalized."""
        mps, _, _ = spin_chain_su2
        assert math.isclose(mps.norm(), 1.0, abs_tol=_ATOL), (
            f"Heisenberg (SU2) MPS norm = {mps.norm():.12f}, expected 1"
        )

    def test_energy_invariant_under_mps_canonical(self, spin_chain_su2):
        """Energy is unchanged through a canonical sweep of the MPS: center 0 → L-1."""
        mps, mpo, E_gs = spin_chain_su2
        L = len(mps)
        mps_c = MPS([mps[i].clone() for i in range(L)], center=mps.center)
        mps_c.canonical(0)
        E0 = observe(mps_c, mpo)
        assert math.isclose(E0, E_gs, abs_tol=_ATOL), (
            f"Heisenberg (SU2) energy changed after MPS.canonical(0): "
            f"observe={E0:.12f}, ref={E_gs:.12f}, diff={abs(E0 - E_gs):.3e}"
        )
        mps_c.canonical(L - 1)
        EN = observe(mps_c, mpo)
        assert math.isclose(EN, E_gs, abs_tol=_ATOL), (
            f"Heisenberg (SU2) energy changed after MPS.canonical(L-1): "
            f"observe={EN:.12f}, ref={E_gs:.12f}, diff={abs(EN - E_gs):.3e}"
        )

    def test_energy_invariant_under_mpo_canonical(self, spin_chain_su2):
        """Energy is unchanged through a canonical sweep of the MPO: center 0 → L-1."""
        mps, mpo, E_gs = spin_chain_su2
        L = len(mpo)
        mpo_c = MPO([mpo[i].clone() for i in range(L)], center=mpo.center)
        mpo_c.canonical(0)
        E0 = observe(mps, mpo_c)
        assert math.isclose(E0, E_gs, abs_tol=_ATOL), (
            f"Heisenberg (SU2) energy changed after MPO.canonical(0): "
            f"observe={E0:.12f}, ref={E_gs:.12f}, diff={abs(E0 - E_gs):.3e}"
        )
        mpo_c.canonical(L - 1)
        EN = observe(mps, mpo_c)
        assert math.isclose(EN, E_gs, abs_tol=_ATOL), (
            f"Heisenberg (SU2) energy changed after MPO.canonical(L-1): "
            f"observe={EN:.12f}, ref={E_gs:.12f}, diff={abs(EN - E_gs):.3e}"
        )
