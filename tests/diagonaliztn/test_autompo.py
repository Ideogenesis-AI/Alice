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


"""AutoMPO structural and correctness tests."""

from __future__ import annotations

import math

import pytest

from nicole import load_space

from alice.network import Interaction2Site, MPO, observe, build_hamiltonian

from .assign import assign_heisenberg, assign_freefermion, assign_conductor

# Tolerance for energy comparisons.
_ATOL = 1e-10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_interactions(L: int, cpl: float = 1.0) -> list[Interaction2Site]:
    """Return a list of nearest-neighbor `Interaction2Site` objects."""
    return [
        Interaction2Site(cpl=cpl, label=['NN'], leading_site=i, terminal_site=i + 1)
        for i in range(L - 1)
    ]


# ---------------------------------------------------------------------------
# Basic structural tests  (not slow — small L=6 chain)
# ---------------------------------------------------------------------------

class TestAutoMPOBasic:
    """Structural sanity checks on a short L=6 Heisenberg chain (U1)."""

    @pytest.fixture(scope='class')
    def heisenberg_mpo(self):
        """Build and return an AutoMPO Heisenberg MPO for L=6."""
        L = 6
        spc, ops = load_space('Spin', 'U1', {'J': 0.5})
        interactions = _make_interactions(L)
        assign_heisenberg(interactions, spc, ops, symmetry='U1', J=1.0)
        return build_hamiltonian(interactions, L, spc)

    def test_length(self, heisenberg_mpo):
        """MPO must contain exactly L site tensors."""
        assert len(heisenberg_mpo) == 6

    def test_boundary_bond_dims(self, heisenberg_mpo):
        """Open-boundary tensors must have dim-1 on their outer bonds."""
        L = len(heisenberg_mpo)
        left_dim  = heisenberg_mpo[0].indices[0].dim
        right_dim = heisenberg_mpo[L - 1].indices[1].dim
        assert left_dim  == 1, f"Left boundary bond should be 1, got {left_dim}"
        assert right_dim == 1, f"Right boundary bond should be 1, got {right_dim}"

    def test_interior_bond_dims(self, heisenberg_mpo):
        """Interior bond dimensions must be greater than 1 (non-trivial MPO)."""
        dims = heisenberg_mpo.bond_dims
        for k, d in enumerate(dims):
            assert d > 1, (
                f"Bond ({k},{k+1}) has dim {d}; expected > 1 for a "
                f"non-trivial Heisenberg MPO"
            )

    def test_tensor_rank(self, heisenberg_mpo):
        """Every site tensor must have exactly 4 axes."""
        for i in range(len(heisenberg_mpo)):
            assert len(heisenberg_mpo[i].indices) == 4, (
                f"Site {i}: expected 4 axes, got {len(heisenberg_mpo[i].indices)}"
            )

    def test_bond_itags(self, heisenberg_mpo):
        """Bond axes must carry the expected `W{i:02d}` itags."""
        L = len(heisenberg_mpo)
        for i in range(L):
            t = heisenberg_mpo[i]
            left_itag  = t.itags[0]
            right_itag = t.itags[1]
            assert left_itag  == f'W{i:02d}', (
                f"Site {i} left bond itag: expected 'W{i:02d}', got '{left_itag}'"
            )
            assert right_itag == f'W{i+1:02d}', (
                f"Site {i} right bond itag: expected 'W{i+1:02d}', got '{right_itag}'"
            )

    def test_physical_itags(self, heisenberg_mpo):
        """Physical axes must carry the expected `s{i:02d}` itags."""
        L = len(heisenberg_mpo)
        for i in range(L):
            t = heisenberg_mpo[i]
            bra_itag = t.itags[2]
            ket_itag = t.itags[3]
            assert bra_itag == f's{i:02d}', (
                f"Site {i} bra itag: expected 's{i:02d}', got '{bra_itag}'"
            )
            assert ket_itag == f's{i:02d}', (
                f"Site {i} ket itag: expected 's{i:02d}', got '{ket_itag}'"
            )


# ---------------------------------------------------------------------------
# Correctness tests  (slow — L=20 chains using iterative-diag fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestAutoMPOHeisenberg:
    """AutoMPO Heisenberg correctness: energy must match iterative diagonalization."""

    def test_energy(self, spin_chain):
        """AutoMPO Heisenberg energy (U1) must match the iterative-diag reference."""
        mps, _, E_gs = spin_chain
        L = len(mps)
        spc, ops = load_space('Spin', 'U1', {'J': 0.5})
        interactions = _make_interactions(L)
        assign_heisenberg(interactions, spc, ops, symmetry='U1', J=1.0)
        mpo_auto = build_hamiltonian(interactions, L, spc)
        E_obs = observe(mps, mpo_auto)
        assert math.isclose(E_obs, E_gs, abs_tol=_ATOL), (
            f"Heisenberg (U1) energy mismatch: observe={E_obs:.12f}, "
            f"ref={E_gs:.12f}, diff={abs(E_obs - E_gs):.3e}"
        )

    def test_energy_su2(self, spin_chain_su2):
        """AutoMPO Heisenberg energy (SU2) must match the iterative-diag reference."""
        mps, _, E_gs = spin_chain_su2
        L = len(mps)
        spc, ops = load_space('Spin', 'SU2', {'J': 0.5})
        interactions = _make_interactions(L)
        assign_heisenberg(interactions, spc, ops, symmetry='SU2', J=1.0)
        mpo_auto = build_hamiltonian(interactions, L, spc)
        E_obs = observe(mps, mpo_auto)
        assert math.isclose(E_obs, E_gs, abs_tol=_ATOL), (
            f"Heisenberg (SU2) energy mismatch: observe={E_obs:.12f}, "
            f"ref={E_gs:.12f}, diff={abs(E_obs - E_gs):.3e}"
        )


@pytest.mark.slow
class TestAutoMPOFreeFermion:
    """AutoMPO free-fermion correctness: energy must match iterative diagonalization."""

    def test_energy(self, ferm_chain):
        """AutoMPO free-fermion energy must match the iterative-diag reference."""
        mps, _, E_gs = ferm_chain
        L = len(mps)
        spc, ops = load_space('Ferm', 'U1')
        interactions = _make_interactions(L)
        assign_freefermion(interactions, spc, ops, t=1.0)
        mpo_auto = build_hamiltonian(interactions, L, spc)
        E_obs = observe(mps, mpo_auto)
        assert math.isclose(E_obs, E_gs, abs_tol=_ATOL), (
            f"FreeFermion energy mismatch: observe={E_obs:.12f}, "
            f"ref={E_gs:.12f}, diff={abs(E_obs - E_gs):.3e}"
        )


@pytest.mark.slow
class TestAutoMPOConductor:
    """AutoMPO conductor correctness: energy must match iterative diagonalization."""

    def test_energy(self, band_chain):
        """AutoMPO conductor energy (U1,U1) must match the iterative-diag reference."""
        mps, _, E_gs = band_chain
        L = len(mps)
        spc, ops = load_space('Band', 'U1,U1')
        interactions = _make_interactions(L)
        assign_conductor(interactions, spc, ops, symmetry='U1,U1', t=1.0)
        mpo_auto = build_hamiltonian(interactions, L, spc)
        E_obs = observe(mps, mpo_auto)
        assert math.isclose(E_obs, E_gs, abs_tol=_ATOL), (
            f"Conductor (U1,U1) energy mismatch: observe={E_obs:.12f}, "
            f"ref={E_gs:.12f}, diff={abs(E_obs - E_gs):.3e}"
        )

    def test_energy_su2(self, band_chain_su2):
        """AutoMPO conductor energy (U1,SU2) must match the iterative-diag reference."""
        mps, _, E_gs = band_chain_su2
        L = len(mps)
        spc, ops = load_space('Band', 'U1,SU2')
        interactions = _make_interactions(L)
        assign_conductor(interactions, spc, ops, symmetry='U1,SU2', t=1.0)
        mpo_auto = build_hamiltonian(interactions, L, spc)
        E_obs = observe(mps, mpo_auto)
        assert math.isclose(E_obs, E_gs, abs_tol=_ATOL), (
            f"Conductor (U1,SU2) energy mismatch: observe={E_obs:.12f}, "
            f"ref={E_gs:.12f}, diff={abs(E_obs - E_gs):.3e}"
        )
