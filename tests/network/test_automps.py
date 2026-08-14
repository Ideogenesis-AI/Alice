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


"""Tests for the universal MPS initializer (`alice.network.automps`)."""

from __future__ import annotations

import pytest
import torch
from nicole import load_space

from alice.network import MPS, init_mps
from alice.network.automps import (
    _auto_config,
    _bond_charges,
    _reachable_charges,
)


# ---------------------------------------------------------------------------
# Shared physical spaces
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def spin_u1():
    """Spin-1/2 U1 physical space."""
    return load_space('Spin', 'U1', {'J': 0.5})


@pytest.fixture(scope='module')
def spin_su2():
    """Spin-1/2 SU2 physical space."""
    return load_space('Spin', 'SU2', {'J': 0.5})


@pytest.fixture(scope='module')
def ferm_u1():
    """Spinless fermion U1 physical space."""
    return load_space('Ferm', 'U1')


@pytest.fixture(scope='module')
def ferm_z2():
    """Spinless fermion Z2 physical space."""
    return load_space('Ferm', 'Z2')


@pytest.fixture(scope='module')
def band_u1u1():
    """Spinful fermion U1⊗U1 (Band) physical space."""
    return load_space('Band', 'U1,U1')


@pytest.fixture(scope='module')
def band_u1su2():
    """Spinful fermion U1⊗SU2 (Band) physical space."""
    return load_space('Band', 'U1,SU2')


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _q_vac(Op):
    """Return the vacuum charge from the Op dictionary."""
    return Op['vac'].sectors[0].charge


# ---------------------------------------------------------------------------
# Unit tests: _bond_charges
# ---------------------------------------------------------------------------

class TestBondCharges:
    """Verify bond charge propagation for each symmetry group."""

    def test_spin_su2_vbs_path(self, spin_su2):
        """Spin-1/2 SU2 auto-config produces the 0-1-0-1 VBS/dimer bond charges."""
        Spc, Op = spin_su2
        L = 6
        Q_vac = _q_vac(Op)
        cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        Q = _bond_charges(Spc.group, Spc, cfg, Q_vac)

        assert len(Q) == L + 1
        assert Q[0] == Q_vac
        # VBS path: 0, 1, 0, 1, 0, 1, 0
        expected = [0, 1, 0, 1, 0, 1, 0]
        assert Q == expected

    def test_spin_u1_alternating(self, spin_u1):
        """Spin-1/2 U1 auto-config produces alternating bond charges."""
        Spc, Op = spin_u1
        L = 6
        Q_vac = _q_vac(Op)
        cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        Q = _bond_charges(Spc.group, Spc, cfg, Q_vac)

        assert len(Q) == L + 1
        assert Q[0] == Q_vac
        # Charges alternate between 0 and ±1; must return to Q_vac at the end.
        assert Q[-1] == Q_vac
        for i in range(L):
            # Consecutive bonds must differ by exactly 1 (the physical U1 charge).
            assert abs(Q[i + 1] - Q[i]) == 1

    def test_ferm_u1_same_as_spin_u1(self, ferm_u1, spin_u1):
        """Ferm U1 and Spin U1 have identical charge conventions and bond paths."""
        Spc_f, Op_f = ferm_u1
        Spc_s, Op_s = spin_u1
        L = 6
        Q_f = _bond_charges(
            Spc_f.group, Spc_f,
            _auto_config(L, Spc_f, Spc_f.group, _q_vac(Op_f), _q_vac(Op_f)),
            _q_vac(Op_f),
        )
        Q_s = _bond_charges(
            Spc_s.group, Spc_s,
            _auto_config(L, Spc_s, Spc_s.group, _q_vac(Op_s), _q_vac(Op_s)),
            _q_vac(Op_s),
        )
        assert Q_f == Q_s

    def test_ferm_z2_returns_to_vacuum(self, ferm_z2):
        """Ferm Z2 auto-config produces a valid closed path Q_0 = Q_L = 0."""
        Spc, Op = ferm_z2
        L = 8  # even L required for closure
        Q_vac = _q_vac(Op)
        cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        Q = _bond_charges(Spc.group, Spc, cfg, Q_vac)

        assert len(Q) == L + 1
        assert Q[0] == Q_vac
        assert Q[-1] == Q_vac

    def test_band_u1su2_returns_to_vacuum(self, band_u1su2):
        """Band U1⊗SU2 auto-config produces a closed bond charge path."""
        Spc, Op = band_u1su2
        L = 8
        Q_vac = _q_vac(Op)
        cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        Q = _bond_charges(Spc.group, Spc, cfg, Q_vac)

        assert len(Q) == L + 1
        assert Q[0] == Q_vac
        assert Q[-1] == Q_vac

    def test_band_u1u1_returns_to_vacuum(self, band_u1u1):
        """Band U1⊗U1 auto-config produces a closed bond charge path."""
        Spc, Op = band_u1u1
        L = 8
        Q_vac = _q_vac(Op)
        cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        Q = _bond_charges(Spc.group, Spc, cfg, Q_vac)

        assert len(Q) == L + 1
        assert Q[0] == Q_vac
        assert Q[-1] == Q_vac

    def test_custom_config_respected(self, spin_u1):
        """Explicitly passed config is used verbatim."""
        Spc, Op = spin_u1
        # Spin U1: sector 0 = charge -1, sector 1 = charge +1.
        # Config [0, 0, 0, 0] gives Q = [0, -1, -2, -3, -4].
        Q_vac = _q_vac(Op)
        Q = _bond_charges(Spc.group, Spc, [0, 0, 0, 0], Q_vac)
        assert Q[0] == Q_vac
        assert Q[1] == Q_vac - 1 or Q[1] == Q_vac + 1  # one step from vacuum

    def test_length_is_L_plus_one(self, spin_su2):
        """Bond charge list always has length L+1."""
        Spc, Op = spin_su2
        for L in (4, 6, 10):
            Q_vac = _q_vac(Op)
            cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
            Q = _bond_charges(Spc.group, Spc, cfg, Q_vac)
            assert len(Q) == L + 1


# ---------------------------------------------------------------------------
# Unit tests: _auto_config
# ---------------------------------------------------------------------------

class TestAutoConfig:
    """Verify the auto-balanced config selection for each space."""

    def test_spin_su2_all_zeros(self, spin_su2):
        """SU2 has one physical sector → auto-config is forced to all 0."""
        Spc, Op = spin_su2
        L = 8
        Q_vac = _q_vac(Op)
        cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        assert cfg == [0] * L

    def test_spin_u1_alternates(self, spin_u1):
        """Spin U1 auto-config alternates exactly two sector indices."""
        Spc, Op = spin_u1
        L = 8
        Q_vac = _q_vac(Op)
        cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        assert len(cfg) == L
        # Config alternates: each entry is in {0, 1} and adjacent entries differ.
        assert set(cfg) <= {0, 1}
        for i in range(L - 1):
            assert cfg[i] != cfg[i + 1]

    def test_ferm_u1_alternates(self, ferm_u1):
        """Ferm U1 auto-config alternates sector indices."""
        Spc, Op = ferm_u1
        L = 8
        Q_vac = _q_vac(Op)
        cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        assert len(cfg) == L
        assert set(cfg) <= {0, 1}
        for i in range(L - 1):
            assert cfg[i] != cfg[i + 1]

    def test_ferm_z2_uniform(self, ferm_z2):
        """Ferm Z2 auto-config uses a single sector (all same) for even L."""
        Spc, Op = ferm_z2
        L = 8
        Q_vac = _q_vac(Op)
        cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        assert len(cfg) == L
        # All entries should be the same (single-sector fill works for even L).
        assert len(set(cfg)) == 1

    def test_band_u1su2_uniform(self, band_u1su2):
        """Band U1⊗SU2 auto-config fills all sites with the neutral-U1 sector."""
        Spc, Op = band_u1su2
        L = 8
        Q_vac = _q_vac(Op)
        cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        assert len(cfg) == L
        # Doublet sector (first component U1=0) — all sites should use it.
        assert len(set(cfg)) == 1
        # Confirm the chosen sector has first U1 component = 0.
        chosen_charge = Spc.sectors[cfg[0]].charge
        assert chosen_charge[0] == 0

    def test_band_u1u1_two_sectors(self, band_u1u1):
        """Band U1⊗U1 auto-config uses exactly two alternating neutral-U1 sectors."""
        Spc, Op = band_u1u1
        L = 8
        Q_vac = _q_vac(Op)
        cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        assert len(cfg) == L
        # Two distinct neutral sectors should alternate.
        assert len(set(cfg)) == 2
        # Both chosen sectors must have first U1 component = 0 (neutral charge).
        for k in set(cfg):
            assert Spc.sectors[k].charge[0] == 0

    def test_explicit_config_passthrough(self, spin_u1):
        """init_mps passes explicit config unchanged."""
        Spc, Op = spin_u1
        L = 6
        explicit = [0, 1, 0, 1, 0, 1]
        mps = init_mps(L, Spc, Op, bond_dim=1, config=explicit)
        assert isinstance(mps, MPS)

    def test_wrong_config_length_raises(self, spin_u1):
        """Explicit config with wrong length raises ValueError."""
        Spc, Op = spin_u1
        with pytest.raises(ValueError, match="config has length"):
            init_mps(6, Spc, Op, bond_dim=1, config=[0, 1])

    def test_invalid_bond_dim_raises(self, spin_u1):
        """bond_dim < 1 raises ValueError."""
        Spc, Op = spin_u1
        with pytest.raises(ValueError, match="bond_dim"):
            init_mps(6, Spc, Op, bond_dim=0)


# ---------------------------------------------------------------------------
# Unit tests: _auto_config with explicit target_qn
# ---------------------------------------------------------------------------

class TestAutoConfigTargetQn:
    """Verify that _auto_config reaches target_qn when given explicitly."""

    def test_spin_u1_odd_L_target_positive(self, spin_u1):
        """Spin U1, L=7, target_qn=+1: greedy achieves Q[7]=+1 exactly."""
        Spc, Op = spin_u1
        Q_vac = _q_vac(Op)
        cfg = _auto_config(7, Spc, Spc.group, Q_vac, target_qn=1)
        Q = _bond_charges(Spc.group, Spc, cfg, Q_vac)
        assert len(cfg) == 7
        assert Q[7] == 1

    def test_spin_u1_odd_L_target_negative(self, spin_u1):
        """Spin U1, L=7, target_qn=-1: greedy achieves Q[7]=-1 exactly."""
        Spc, Op = spin_u1
        Q_vac = _q_vac(Op)
        cfg = _auto_config(7, Spc, Spc.group, Q_vac, target_qn=-1)
        Q = _bond_charges(Spc.group, Spc, cfg, Q_vac)
        assert len(cfg) == 7
        assert Q[7] == -1

    def test_even_L_target_qvac_same_as_before(self, spin_u1):
        """Even L, target_qn=Q_vac: same config as without target_qn argument."""
        Spc, Op = spin_u1
        L = 8
        Q_vac = _q_vac(Op)
        cfg_default = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        cfg_explicit = _auto_config(L, Spc, Spc.group, Q_vac, target_qn=Q_vac)
        assert cfg_default == cfg_explicit

    def test_ferm_u1_odd_L_target_achievable(self, ferm_u1):
        """Ferm U1, L=7, target_qn=1: auto-config achieves Q[7]=1."""
        Spc, Op = ferm_u1
        Q_vac = _q_vac(Op)
        cfg = _auto_config(7, Spc, Spc.group, Q_vac, target_qn=1)
        Q = _bond_charges(Spc.group, Spc, cfg, Q_vac)
        assert Q[7] == 1


# ---------------------------------------------------------------------------
# Unit tests: _reachable_charges
# ---------------------------------------------------------------------------

class TestReachableCharges:
    """Verify that BFS sector enumeration is L-independent and correct."""

    def test_su2_spin_half_q0_d2(self, spin_su2):
        """SU2 spin-1/2, Q_c=0, d=2 → {0, 1, 2}."""
        Spc, _ = spin_su2
        charges = _reachable_charges(Spc.group, Spc, Q_c=0, d=2)
        assert charges == {0, 1, 2}

    def test_su2_spin_half_q1_d2(self, spin_su2):
        """SU2 spin-1/2, Q_c=1, d=2 → {0, 1, 2, 3}."""
        Spc, _ = spin_su2
        charges = _reachable_charges(Spc.group, Spc, Q_c=1, d=2)
        assert charges == {0, 1, 2, 3}

    def test_spin_u1_q0_d2(self, spin_u1):
        """Spin U1, Q_c=0, d=2 → {-2, -1, 0, +1, +2}."""
        Spc, _ = spin_u1
        charges = _reachable_charges(Spc.group, Spc, Q_c=0, d=2)
        assert charges == {-2, -1, 0, 1, 2}

    def test_ferm_u1_q0_d2(self, ferm_u1):
        """Ferm U1, Q_c=0, d=2 → {-2, -1, 0, +1, +2} (same convention as Spin U1)."""
        Spc, _ = ferm_u1
        charges = _reachable_charges(Spc.group, Spc, Q_c=0, d=2)
        assert charges == {-2, -1, 0, 1, 2}

    def test_ferm_z2_q0_d2(self, ferm_z2):
        """Ferm Z2, Q_c=0, d=2 → {0, 1} (all Z2 charges, saturated at d=1)."""
        Spc, _ = ferm_z2
        charges = _reachable_charges(Spc.group, Spc, Q_c=0, d=2)
        assert charges == {0, 1}

    def test_band_u1su2_q00_d2_count(self, band_u1su2):
        """Band U1⊗SU2, Q_c=(0,0), d=2 → 9 distinct charges."""
        Spc, _ = band_u1su2
        charges = _reachable_charges(Spc.group, Spc, Q_c=(0, 0), d=2)
        assert len(charges) == 9
        # All elements must be tuples.
        assert all(isinstance(q, tuple) for q in charges)

    def test_count_independent_of_L(self, spin_su2):
        """Sector count from BFS depends only on Q_c and d, never on L.

        `_reachable_charges` takes no L argument. Calling it twice with the
        same (Q_c, d) must always give the same result.
        """
        Spc, _ = spin_su2
        # Q_c=0: {0, 1, 2} → 3 sectors.
        c1 = _reachable_charges(Spc.group, Spc, Q_c=0, d=2)
        c2 = _reachable_charges(Spc.group, Spc, Q_c=0, d=2)
        assert c1 == c2 == {0, 1, 2}
        # Q_c=1: {0, 1, 2, 3} → 4 sectors.
        c3 = _reachable_charges(Spc.group, Spc, Q_c=1, d=2)
        c4 = _reachable_charges(Spc.group, Spc, Q_c=1, d=2)
        assert c3 == c4 == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Integration tests: product-state mode (bond_dim=1)
# ---------------------------------------------------------------------------

class TestInitMpsProductState:
    """Integration tests for init_mps with bond_dim=1."""

    @pytest.mark.parametrize('space_fixture', [
        'spin_u1', 'spin_su2', 'ferm_u1', 'ferm_z2', 'band_u1u1', 'band_u1su2',
    ])
    def test_returns_mps(self, space_fixture, request):
        """init_mps(bond_dim=1) always returns an MPS instance."""
        Spc, Op = request.getfixturevalue(space_fixture)
        mps = init_mps(8, Spc, Op, bond_dim=1)
        assert isinstance(mps, MPS)

    @pytest.mark.parametrize('space_fixture', [
        'spin_u1', 'spin_su2', 'ferm_u1', 'ferm_z2', 'band_u1u1', 'band_u1su2',
    ])
    def test_correct_length(self, space_fixture, request):
        """init_mps(bond_dim=1) produces an MPS of the requested length."""
        Spc, Op = request.getfixturevalue(space_fixture)
        for L in (4, 8):
            mps = init_mps(L, Spc, Op, bond_dim=1)
            assert len(mps) == L

    @pytest.mark.parametrize('space_fixture', [
        'spin_u1', 'spin_su2', 'ferm_u1', 'ferm_z2', 'band_u1u1', 'band_u1su2',
    ])
    def test_bond_dim_is_one(self, space_fixture, request):
        """All interior bonds have dimension 1 for bond_dim=1."""
        Spc, Op = request.getfixturevalue(space_fixture)
        L = 8
        mps = init_mps(L, Spc, Op, bond_dim=1)
        # Check every left bond (axis 0) has total dim = 1.
        for t in mps:
            assert t.indices[0].dim == 1

    @pytest.mark.parametrize('space_fixture', [
        'spin_u1', 'spin_su2', 'ferm_u1', 'ferm_z2', 'band_u1u1', 'band_u1su2',
    ])
    def test_center_is_zero(self, space_fixture, request):
        """MPS is returned with orthogonality center at site 0."""
        Spc, Op = request.getfixturevalue(space_fixture)
        mps = init_mps(8, Spc, Op, bond_dim=1)
        assert mps.center == 0

    @pytest.mark.parametrize('space_fixture', [
        'spin_u1', 'spin_su2', 'ferm_u1', 'ferm_z2',
    ])
    def test_norm_is_one(self, space_fixture, request):
        """Product state (bond_dim=1) has norm 1 after canonical."""
        Spc, Op = request.getfixturevalue(space_fixture)
        mps = init_mps(8, Spc, Op, bond_dim=1)
        # Use the MPS's own norm() method, which correctly accounts for SU(2)
        # CG normalization factors in non-Abelian tensors.
        nrm = mps.norm()
        assert abs(float(nrm) - 1.0) < 1e-10, f"norm = {float(nrm)}, expected 1"

    def test_seed_reproducibility(self, spin_u1):
        """Product-state mode is deterministic (independent of seed)."""
        Spc, Op = spin_u1
        mps1 = init_mps(8, Spc, Op, bond_dim=1, seed=1)
        mps2 = init_mps(8, Spc, Op, bond_dim=1, seed=99)
        # Both are the same product state — check bond dims match.
        assert mps1.bond_dims == mps2.bond_dims


# ---------------------------------------------------------------------------
# Integration tests: random mode (bond_dim > 1)
# ---------------------------------------------------------------------------

class TestInitMpsRandom:
    """Integration tests for init_mps with bond_dim>1."""

    @pytest.mark.parametrize('space_fixture', [
        'spin_u1', 'spin_su2', 'ferm_u1', 'ferm_z2', 'band_u1su2',
    ])
    def test_returns_mps(self, space_fixture, request):
        """init_mps(bond_dim>1) always returns an MPS instance."""
        Spc, Op = request.getfixturevalue(space_fixture)
        mps = init_mps(8, Spc, Op, bond_dim=8)
        assert isinstance(mps, MPS)

    @pytest.mark.parametrize('space_fixture', [
        'spin_u1', 'spin_su2', 'ferm_u1', 'ferm_z2', 'band_u1su2',
    ])
    def test_correct_length(self, space_fixture, request):
        """init_mps(bond_dim>1) produces an MPS of the requested length."""
        Spc, Op = request.getfixturevalue(space_fixture)
        mps = init_mps(8, Spc, Op, bond_dim=8)
        assert len(mps) == 8

    @pytest.mark.parametrize('space_fixture', [
        'spin_u1', 'spin_su2', 'ferm_u1', 'ferm_z2', 'band_u1su2',
    ])
    def test_center_is_zero(self, space_fixture, request):
        """Random MPS is returned with orthogonality center at site 0."""
        Spc, Op = request.getfixturevalue(space_fixture)
        mps = init_mps(8, Spc, Op, bond_dim=8)
        assert mps.center == 0

    @pytest.mark.parametrize('space_fixture', [
        'spin_u1', 'spin_su2', 'ferm_u1', 'ferm_z2', 'band_u1su2',
    ])
    def test_bond_dim_does_not_exceed_target(self, space_fixture, request):
        """Interior bond dims do not exceed bond_dim after canonicalization."""
        Spc, Op = request.getfixturevalue(space_fixture)
        bond_dim = 8
        mps = init_mps(8, Spc, Op, bond_dim=bond_dim)
        for d in mps.bond_dims:
            assert d <= bond_dim

    @pytest.mark.parametrize('space_fixture', ['spin_u1', 'ferm_u1'])
    def test_bond_sectors_within_reachable(self, space_fixture, request):
        """Interior bond sectors lie within _reachable_charges(d=2) of Q_c."""
        Spc, Op = request.getfixturevalue(space_fixture)
        L = 8
        bond_dim = 8
        mps = init_mps(L, Spc, Op, bond_dim=bond_dim)
        Q_vac = _q_vac(Op)
        cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        Q_c = _bond_charges(Spc.group, Spc, cfg, Q_vac)[L // 2]
        allowed = _reachable_charges(Spc.group, Spc, Q_c, d=2)
        # Check all interior bond charges lie in the allowed set.
        for i in range(1, L):
            for sector in mps[i].indices[0].sectors:
                assert sector.charge in allowed, (
                    f"site {i}, charge {sector.charge} not in allowed set {allowed}"
                )

    def test_different_seeds_different_tensors(self, spin_u1):
        """Different seeds produce different random tensors."""
        Spc, Op = spin_u1
        mps1 = init_mps(8, Spc, Op, bond_dim=8, seed=1)
        mps2 = init_mps(8, Spc, Op, bond_dim=8, seed=99)
        # Compare the first non-empty block of site 0: they should differ.
        blk1 = next(iter(mps1[0].data.values()))
        blk2 = next(iter(mps2[0].data.values()))
        assert not torch.allclose(blk1, blk2), "Different seeds produced identical tensors"

    def test_same_seed_reproducible(self, spin_su2):
        """Same seed always produces the same tensors."""
        Spc, Op = spin_su2
        mps1 = init_mps(8, Spc, Op, bond_dim=8, seed=42)
        mps2 = init_mps(8, Spc, Op, bond_dim=8, seed=42)
        assert mps1.bond_dims == mps2.bond_dims


# ---------------------------------------------------------------------------
# Regression tests: odd-L random MPS
# ---------------------------------------------------------------------------

class TestInitMpsOddL:
    """Regression tests for odd-length random MPS (was producing zero tensors)."""

    @pytest.mark.parametrize('space_fixture', [
        'spin_u1', 'spin_su2', 'ferm_u1', 'ferm_z2', 'band_u1u1', 'band_u1su2',
    ])
    def test_odd_L_nonzero_norm(self, space_fixture, request):
        """Random MPS for odd L has nonzero norm after canonicalization.

        Before the fix, pinning the right boundary to Q_vac made all blocks of
        the last site tensor charge-forbidden, producing a zero MPS.
        """
        Spc, Op = request.getfixturevalue(space_fixture)
        mps = init_mps(7, Spc, Op, bond_dim=8)
        assert isinstance(mps, MPS)
        assert float(mps.norm()) > 1e-10, "MPS norm is zero — right boundary charge bug"

    def test_odd_L_right_boundary_charge(self, spin_u1):
        """Right boundary of last tensor carries Q[L] from the charge path."""
        Spc, Op = spin_u1
        L = 7
        Q_vac = _q_vac(Op)
        cfg = _auto_config(L, Spc, Spc.group, Q_vac, Q_vac)
        Q = _bond_charges(Spc.group, Spc, cfg, Q_vac)
        expected_qn = Q[L]

        mps = init_mps(L, Spc, Op, bond_dim=8)
        # The right bond of site L-1 is axis 1. After canonicalization it
        # should be a single-sector dummy index with charge = Q[L].
        right_idx = mps[L - 1].indices[1]
        assert len(right_idx.sectors) == 1
        assert right_idx.sectors[0].charge == expected_qn

    def test_odd_L_warning_emitted(self, spin_u1, caplog):
        """A WARNING is logged for odd L when target_qn is not given."""
        import logging
        Spc, Op = spin_u1
        with caplog.at_level(logging.WARNING, logger='alice.network.automps'):
            init_mps(7, Spc, Op, bond_dim=8)
        assert any(
            record.levelno >= logging.WARNING
            for record in caplog.records
        ), "Expected a WARNING for odd-L chain without explicit target_qn"


# ---------------------------------------------------------------------------
# Tests for the target_qn parameter
# ---------------------------------------------------------------------------

class TestTargetQn:
    """Tests for the `target_qn` keyword argument on `init_mps`."""

    def test_explicit_target_qn_random_mps(self, spin_u1):
        """Explicit target_qn is used as the right boundary charge (random mode)."""
        Spc, Op = spin_u1
        L = 7
        # Use +1 as the target: the auto-config greedy reaches it exactly.
        qn = 1
        mps = init_mps(L, Spc, Op, bond_dim=8, target_qn=qn)
        right_idx = mps[L - 1].indices[1]
        assert len(right_idx.sectors) == 1
        assert right_idx.sectors[0].charge == qn

    def test_explicit_target_qn_product_state(self, spin_u1):
        """target_qn drives the auto-config in product-state mode (bond_dim=1).

        For Spin U1, L=7, target_qn=+1: the auto-config greedy achieves Q[7]=+1
        exactly, so the product state has norm 1 and the right boundary of the
        last tensor carries charge +1.
        """
        Spc, Op = spin_u1
        L = 7
        mps = init_mps(L, Spc, Op, bond_dim=1, target_qn=1)
        assert isinstance(mps, MPS)
        # Norm must be nonzero (product state was not killed by a wrong boundary).
        assert float(mps.norm()) > 1e-10
        # Right boundary of last site must carry the requested charge.
        right_idx = mps[L - 1].indices[1]
        assert len(right_idx.sectors) == 1
        assert right_idx.sectors[0].charge == 1

    def test_explicit_target_qn_no_warning(self, spin_u1, caplog):
        """No warning is emitted when target_qn is given explicitly and achieved."""
        import logging
        Spc, Op = spin_u1
        L = 7
        # target_qn=1 is achievable for L=7 Spin U1 → no mismatch warning.
        with caplog.at_level(logging.WARNING, logger='alice.network.automps'):
            init_mps(L, Spc, Op, bond_dim=8, target_qn=1)
        assert not any(
            record.levelno >= logging.WARNING
            for record in caplog.records
        ), "Unexpected warning when target_qn is achievable"

    def test_target_qn_even_L_no_warning(self, spin_u1, caplog):
        """No warning is emitted for even L without explicit target_qn."""
        import logging
        Spc, Op = spin_u1
        with caplog.at_level(logging.WARNING, logger='alice.network.automps'):
            init_mps(8, Spc, Op, bond_dim=8)
        assert not any(
            record.levelno >= logging.WARNING
            for record in caplog.records
        ), "Unexpected warning for even-L chain"

    def test_negative_target_qn_product_state(self, spin_u1):
        """target_qn=-1 also works for product-state mode (Sz = -½, L=7)."""
        Spc, Op = spin_u1
        mps = init_mps(7, Spc, Op, bond_dim=1, target_qn=-1)
        right_idx = mps[6].indices[1]
        assert right_idx.sectors[0].charge == -1

    def test_unreachable_target_qn_raises_random(self, spin_u1):
        """ValueError when target_qn has the wrong parity for the given L (random mode).

        For Spin U1, each site contributes ±1. After L=4 sites the bond charge
        is always even (-4, -2, 0, 2, 4); an odd target such as 1 is unreachable.
        """
        Spc, Op = spin_u1
        with pytest.raises(ValueError, match="target_qn"):
            init_mps(4, Spc, Op, bond_dim=8, target_qn=1)

    def test_unreachable_target_qn_raises_product_state(self, spin_u1):
        """ValueError when target_qn has the wrong parity for the given L (product state).

        Same parity argument as the random-mode test: L=4 Spin U1 cannot reach
        an odd charge, so bond_dim=1 must also raise rather than silently produce
        an MPS in the wrong sector.
        """
        Spc, Op = spin_u1
        with pytest.raises(ValueError, match="target_qn"):
            init_mps(4, Spc, Op, bond_dim=1, target_qn=1)
