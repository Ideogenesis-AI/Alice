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


"""AutoMPO structural and correctness tests."""

from __future__ import annotations

import math

import pytest

from alice.network import observe, build_hamiltonian, build_interaction


# Tolerance for energy comparisons.
_ATOL = 1e-10


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _chain_config(L: int, model: str, category: str, **model_kwargs) -> dict:
    """Build a full build_interaction config dict for a 1D chain of length L."""
    model_cfg = {'category': category, 'label': model, **model_kwargs}
    return {
        'geometry': {
            'lattice':  'square',
            'traverse': 'sequential',
            'lx': L,
            'ly': 1,
            'bcx': 'OBC',
            'bcy': 'OBC',
            'n2x': True,
            'n2y': False,
            'n3d': False,
            'n3o': False,
        },
        'model': model_cfg,
    }


def _square_config(Lx: int, Ly: int, model: str, category: str, **model_kwargs) -> dict:
    """Build a full build_interaction config dict for an Lx×Ly square lattice."""
    model_cfg = {'category': category, 'label': model, **model_kwargs}
    return {
        'geometry': {
            'lattice':  'square',
            'traverse': 'sequential',
            'lx': Lx,
            'ly': Ly,
            'bcx': 'OBC',
            'bcy': 'OBC',
            'n2x': True,
            'n2y': True,
            'n3d': False,
            'n3o': False,
        },
        'model': model_cfg,
    }


# ---------------------------------------------------------------------------
# Basic structural tests  (not slow — small L=6 chain)
# ---------------------------------------------------------------------------

class TestAutoMPOBasic:
    """Structural sanity checks on a short L=6 Heisenberg chain (U1)."""

    @pytest.fixture(scope='class')
    def heisenberg_mpo(self):
        """Build and return an AutoMPO Heisenberg MPO for L=6."""
        L = 6
        config = _chain_config(L, 'Heisenberg', 'bosonic',
                               symmetry='U1', spin=0.5, J=1.0)
        interactions, spc, geo = build_interaction(config)
        assert geo.L == L
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
        config = _chain_config(L, 'Heisenberg', 'bosonic',
                               symmetry='U1', spin=0.5, J=1.0)
        interactions, spc, _ = build_interaction(config)
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
        config = _chain_config(L, 'Heisenberg', 'bosonic',
                               symmetry='SU2', spin=0.5, J=1.0)
        interactions, spc, _ = build_interaction(config)
        mpo_auto = build_hamiltonian(interactions, L, spc)
        E_obs = observe(mps, mpo_auto)
        assert math.isclose(E_obs, E_gs, abs_tol=_ATOL), (
            f"Heisenberg (SU2) energy mismatch: observe={E_obs:.12f}, "
            f"ref={E_gs:.12f}, diff={abs(E_obs - E_gs):.3e}"
        )

    def test_energy_compact_every(self, spin_chain):
        """Heisenberg (U1) energy with compact_every=3 must match the reference."""
        mps, _, E_gs = spin_chain
        L = len(mps)
        config = _chain_config(L, 'Heisenberg', 'bosonic',
                               symmetry='U1', spin=0.5, J=1.0)
        interactions, spc, _ = build_interaction(config)
        mpo_auto = build_hamiltonian(interactions, L, spc, compact_every=3)
        E_obs = observe(mps, mpo_auto)
        assert math.isclose(E_obs, E_gs, abs_tol=_ATOL), (
            f"Heisenberg (U1, compact_every=3) energy mismatch: "
            f"observe={E_obs:.12f}, ref={E_gs:.12f}, diff={abs(E_obs - E_gs):.3e}"
        )


@pytest.mark.slow
class TestAutoMPOFreeFermion:
    """AutoMPO free-fermion correctness: energy must match iterative diagonalization."""

    def test_energy(self, ferm_chain):
        """AutoMPO free-fermion energy must match the iterative-diag reference."""
        mps, _, E_gs = ferm_chain
        L = len(mps)
        config = _chain_config(L, 'FreeFermion', 'fermionic',
                               symmetry='U1', t=1.0)
        interactions, spc, _ = build_interaction(config)
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
        config = _chain_config(L, 'Hubbard', 'conductor',
                               symmetry='U1,U1', t=1.0, U=0.0)
        interactions, spc, _ = build_interaction(config)
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
        config = _chain_config(L, 'Hubbard', 'conductor',
                               symmetry='U1,SU2', t=1.0, U=0.0)
        interactions, spc, _ = build_interaction(config)
        mpo_auto = build_hamiltonian(interactions, L, spc)
        E_obs = observe(mps, mpo_auto)
        assert math.isclose(E_obs, E_gs, abs_tol=_ATOL), (
            f"Conductor (U1,SU2) energy mismatch: observe={E_obs:.12f}, "
            f"ref={E_gs:.12f}, diff={abs(E_obs - E_gs):.3e}"
        )


# ---------------------------------------------------------------------------
# compact_every parameter tests  (not slow — L=10 chain)
# ---------------------------------------------------------------------------

class TestCompactEvery:
    """Verify that periodic intermediate compaction produces bond dims, norm,
    and structural validity identical to compaction only at the end."""

    @pytest.mark.parametrize('every', [1, 2, 3, 5, 9])
    def test_compact_every(self, every):
        """compact_every=N must yield the same bond dims, norm, and valid MPO."""
        L = 10
        cfg = _chain_config(L, 'Heisenberg', 'bosonic', symmetry='U1', spin=0.5, J=1.0)
        interactions, spc, _ = build_interaction(cfg)
        ref = build_hamiltonian(interactions, L, spc, compact_every=0)
        mpo = build_hamiltonian(interactions, L, spc, compact_every=every)
        assert mpo.bond_dims == ref.bond_dims, (
            f"compact_every={every}: bond dims {mpo.bond_dims} != "
            f"reference {ref.bond_dims}"
        )
        assert math.isclose(mpo.norm(), ref.norm(), rel_tol=1e-10), (
            f"compact_every={every}: norm {mpo.norm():.12f} != "
            f"reference {ref.norm():.12f}"
        )
        mpo._validate()


# ---------------------------------------------------------------------------
# compact() bond-dimension and physics tests (not slow)
# ---------------------------------------------------------------------------

class TestCompactPhysics:
    """Verify that MPO.compact() produces correct exact bond dimensions and
    preserves the operator norm for several standard lattice models.

    For finite-state-machine MPOs built from NN (and limited longer-range)
    interactions, the SVD threshold 1e-14 retains all significant singular
    values, so the compressed bond dimension equals the minimal exact value.

    Known exact bulk bond dimensions (middle of a long chain):
    - Heisenberg NN (U1):          5  = {I, S+, S-, Sz, H_done}
    - Heisenberg NN (SU2):         3  = {I, S, H_done}  (spin multiplet)
    - Tight-binding / band (U1,U1): 6  = {I, c↑†, c↑, c↓†, c↓, H_done}
    - Tight-binding / band (U1,SU2):4  = {I, c†, c, H_done}  (SU2 doublet)
    - Heisenberg 4×2 square (U1):  8  (two open vertical bonds across the cut)
    - Heisenberg 4×3 square (U1): 11  (three open vertical bonds across the cut)
    """

    # ------------------------------------------------------------------
    # Class-scoped MPO fixtures (built once per test class)
    # ------------------------------------------------------------------

    @pytest.fixture(scope='class')
    def heisenberg_u1_mpo(self):
        """L=10 Heisenberg chain, U1 symmetry."""
        L = 10
        cfg = _chain_config(L, 'Heisenberg', 'bosonic', symmetry='U1', spin=0.5, J=1.0)
        interactions, spc, _ = build_interaction(cfg)
        return build_hamiltonian(interactions, L, spc)

    @pytest.fixture(scope='class')
    def heisenberg_su2_mpo(self):
        """L=10 Heisenberg chain, SU2 symmetry."""
        L = 10
        cfg = _chain_config(L, 'Heisenberg', 'bosonic', symmetry='SU2', spin=0.5, J=1.0)
        interactions, spc, _ = build_interaction(cfg)
        return build_hamiltonian(interactions, L, spc)

    @pytest.fixture(scope='class')
    def hubbard_u1u1_mpo(self):
        """L=8 Hubbard chain with U=0 (tight-binding), U1×U1 symmetry."""
        L = 8
        cfg = _chain_config(L, 'Hubbard', 'conductor', symmetry='U1,U1', t=1.0, U=0.0, mu=0.0)
        interactions, spc, _ = build_interaction(cfg)
        return build_hamiltonian(interactions, L, spc)

    @pytest.fixture(scope='class')
    def hubbard_u1su2_mpo(self):
        """L=8 Hubbard chain with U=0 (tight-binding), U1×SU2 symmetry."""
        L = 8
        cfg = _chain_config(L, 'Hubbard', 'conductor', symmetry='U1,SU2', t=1.0, U=0.0, mu=0.0)
        interactions, spc, _ = build_interaction(cfg)
        return build_hamiltonian(interactions, L, spc)

    @pytest.fixture(scope='class')
    def heisenberg_4x2_mpo(self):
        """4×2 Heisenberg square lattice (sequential ordering), U1 symmetry."""
        cfg = _square_config(4, 2, 'Heisenberg', 'bosonic', symmetry='U1', spin=0.5, J=1.0)
        interactions, spc, geo = build_interaction(cfg)
        return build_hamiltonian(interactions, geo.L, spc)

    @pytest.fixture(scope='class')
    def heisenberg_4x3_mpo(self):
        """4×3 Heisenberg square lattice (sequential ordering), U1 symmetry."""
        cfg = _square_config(4, 3, 'Heisenberg', 'bosonic', symmetry='U1', spin=0.5, J=1.0)
        interactions, spc, geo = build_interaction(cfg)
        return build_hamiltonian(interactions, geo.L, spc)

    # ------------------------------------------------------------------
    # Heisenberg chain (U1): exact bulk bond dimension = 5
    # ------------------------------------------------------------------

    def test_heisenberg_u1_bond_dims(self, heisenberg_u1_mpo):
        """Heisenberg (U1) MPO must have exact bond dims [4,5,…,5,4]."""
        assert heisenberg_u1_mpo.bond_dims == [4, 5, 5, 5, 5, 5, 5, 5, 4]

    def test_heisenberg_u1_norm_finite(self, heisenberg_u1_mpo):
        assert math.isfinite(heisenberg_u1_mpo.norm()) and heisenberg_u1_mpo.norm() > 0

    def test_heisenberg_u1_idempotent(self, heisenberg_u1_mpo):
        """A second compact() must not change bond dims or norm."""
        dims_before = heisenberg_u1_mpo.bond_dims[:]
        n_before = heisenberg_u1_mpo.norm()
        heisenberg_u1_mpo.compact()
        assert heisenberg_u1_mpo.bond_dims == dims_before
        assert math.isclose(heisenberg_u1_mpo.norm(), n_before, rel_tol=1e-10)

    def test_heisenberg_u1_validates(self, heisenberg_u1_mpo):
        heisenberg_u1_mpo._validate()

    # ------------------------------------------------------------------
    # Heisenberg chain (SU2): exact bulk bond dimension = 3
    # ------------------------------------------------------------------

    def test_heisenberg_su2_bond_dims(self, heisenberg_su2_mpo):
        """Heisenberg (SU2) MPO must have exact bond dims [2,3,…,3,2]."""
        assert heisenberg_su2_mpo.bond_dims == [2, 3, 3, 3, 3, 3, 3, 3, 2]

    def test_heisenberg_su2_norm_finite(self, heisenberg_su2_mpo):
        assert math.isfinite(heisenberg_su2_mpo.norm()) and heisenberg_su2_mpo.norm() > 0

    def test_heisenberg_su2_idempotent(self, heisenberg_su2_mpo):
        """A second compact() must not change bond dims or norm."""
        dims_before = heisenberg_su2_mpo.bond_dims[:]
        n_before = heisenberg_su2_mpo.norm()
        heisenberg_su2_mpo.compact()
        assert heisenberg_su2_mpo.bond_dims == dims_before
        assert math.isclose(heisenberg_su2_mpo.norm(), n_before, rel_tol=1e-10)

    # ------------------------------------------------------------------
    # Hubbard chain U=0 (tight-binding), U1×U1: exact bulk bond dim = 6
    # ------------------------------------------------------------------

    def test_hubbard_u1u1_bond_dims(self, hubbard_u1u1_mpo):
        """Hubbard (U1×U1, U=0) MPO must have exact bond dims [5,6,…,6,5]."""
        assert hubbard_u1u1_mpo.bond_dims == [5, 6, 6, 6, 6, 6, 5]

    def test_hubbard_u1u1_norm_finite(self, hubbard_u1u1_mpo):
        assert math.isfinite(hubbard_u1u1_mpo.norm()) and hubbard_u1u1_mpo.norm() > 0

    def test_hubbard_u1u1_idempotent(self, hubbard_u1u1_mpo):
        """A second compact() must not change bond dims or norm."""
        dims_before = hubbard_u1u1_mpo.bond_dims[:]
        n_before = hubbard_u1u1_mpo.norm()
        hubbard_u1u1_mpo.compact()
        assert hubbard_u1u1_mpo.bond_dims == dims_before
        assert math.isclose(hubbard_u1u1_mpo.norm(), n_before, rel_tol=1e-10)

    # ------------------------------------------------------------------
    # Hubbard chain U=0 (tight-binding), U1×SU2: exact bulk bond dim = 4
    # ------------------------------------------------------------------

    def test_hubbard_u1su2_bond_dims(self, hubbard_u1su2_mpo):
        """Hubbard (U1×SU2, U=0) MPO must have exact bond dims [3,4,…,4,3]."""
        assert hubbard_u1su2_mpo.bond_dims == [3, 4, 4, 4, 4, 4, 3]

    def test_hubbard_u1su2_norm_finite(self, hubbard_u1su2_mpo):
        assert math.isfinite(hubbard_u1su2_mpo.norm()) and hubbard_u1su2_mpo.norm() > 0

    def test_hubbard_u1su2_idempotent(self, hubbard_u1su2_mpo):
        """A second compact() must not change bond dims or norm."""
        dims_before = hubbard_u1su2_mpo.bond_dims[:]
        n_before = hubbard_u1su2_mpo.norm()
        hubbard_u1su2_mpo.compact()
        assert hubbard_u1su2_mpo.bond_dims == dims_before
        assert math.isclose(hubbard_u1su2_mpo.norm(), n_before, rel_tol=1e-10)

    # ------------------------------------------------------------------
    # 2D Heisenberg 4×2 square (U1): max bulk bond dimension = 8
    # ------------------------------------------------------------------

    def test_heisenberg_4x2_bond_dims(self, heisenberg_4x2_mpo):
        """4×2 Heisenberg (U1) MPO must have exact bond dims [4,8,…,8,4]."""
        assert heisenberg_4x2_mpo.bond_dims == [4, 8, 8, 8, 8, 8, 4]

    def test_heisenberg_4x2_norm_finite(self, heisenberg_4x2_mpo):
        assert math.isfinite(heisenberg_4x2_mpo.norm()) and heisenberg_4x2_mpo.norm() > 0

    def test_heisenberg_4x2_idempotent(self, heisenberg_4x2_mpo):
        """A second compact() must not change bond dims or norm."""
        dims_before = heisenberg_4x2_mpo.bond_dims[:]
        n_before = heisenberg_4x2_mpo.norm()
        heisenberg_4x2_mpo.compact()
        assert heisenberg_4x2_mpo.bond_dims == dims_before
        assert math.isclose(heisenberg_4x2_mpo.norm(), n_before, rel_tol=1e-10)

    def test_heisenberg_4x2_validates(self, heisenberg_4x2_mpo):
        heisenberg_4x2_mpo._validate()

    # ------------------------------------------------------------------
    # 2D Heisenberg 4×3 square (U1): max bulk bond dimension = 11
    # ------------------------------------------------------------------

    def test_heisenberg_4x3_bond_dims(self, heisenberg_4x3_mpo):
        """4×3 Heisenberg (U1) MPO must have exact bond dims [4,8,11,…,11,8,4]."""
        assert heisenberg_4x3_mpo.bond_dims == [4, 8, 11, 11, 11, 11, 11, 11, 11, 8, 4]

    def test_heisenberg_4x3_norm_finite(self, heisenberg_4x3_mpo):
        assert math.isfinite(heisenberg_4x3_mpo.norm()) and heisenberg_4x3_mpo.norm() > 0

    def test_heisenberg_4x3_idempotent(self, heisenberg_4x3_mpo):
        """A second compact() must not change bond dims or norm."""
        dims_before = heisenberg_4x3_mpo.bond_dims[:]
        n_before = heisenberg_4x3_mpo.norm()
        heisenberg_4x3_mpo.compact()
        assert heisenberg_4x3_mpo.bond_dims == dims_before
        assert math.isclose(heisenberg_4x3_mpo.norm(), n_before, rel_tol=1e-10)

    def test_heisenberg_4x3_validates(self, heisenberg_4x3_mpo):
        heisenberg_4x3_mpo._validate()
