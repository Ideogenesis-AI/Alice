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


"""Tests for the universal iMPS initializer (`alice.network.autoimps`)."""

from __future__ import annotations

import pytest
import torch
from nicole import load_space

from alice.network import iMPS, init_imps
from alice.network.autoimps import (
    _auto_config,
    _bond_charges,
    _even_neighborhood,
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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _q_vac(Op):
    """Return the vacuum charge from the Op dictionary."""
    return Op['vac'].sectors[0].charge


# ---------------------------------------------------------------------------
# Unit tests: _bond_charges and _auto_config
# ---------------------------------------------------------------------------

class TestCellConfiguration:
    """Verify the configuration chosen for a cell closes on itself."""

    @pytest.mark.parametrize('space_fixture', ['spin_u1', 'spin_su2', 'ferm_u1'])
    @pytest.mark.parametrize('N', [2, 4])
    def test_auto_config_tiles(self, space_fixture, N, request):
        """The automatic configuration returns the bond charge to its start."""
        Spc, Op = request.getfixturevalue(space_fixture)
        q_vac = _q_vac(Op)
        config = _auto_config(N, Spc, Spc.group, q_vac)
        charges = _bond_charges(Spc.group, Spc, config, q_vac)

        assert len(config) == N
        assert len(charges) == N + 1
        assert charges[0] == q_vac
        assert charges[-1] == charges[0]

    def test_spin_u1_cell_alternates(self, spin_u1):
        """A spin-1/2 U1 cell is the Neel pattern, one up and one down."""
        Spc, Op = spin_u1
        q_vac = _q_vac(Op)
        charges = _bond_charges(
            Spc.group, Spc, _auto_config(4, Spc, Spc.group, q_vac), q_vac)

        assert charges == [0, 1, 0, 1, 0]

    def test_ferm_z2_fills_one_sector(self, ferm_z2):
        """A Z2 space has a neutral sector, so even an odd cell tiles."""
        Spc, Op = ferm_z2
        q_vac = _q_vac(Op)
        config = _auto_config(3, Spc, Spc.group, q_vac)
        charges = _bond_charges(Spc.group, Spc, config, q_vac)

        assert len(set(config)) == 1
        assert charges == [q_vac] * 4

    def test_odd_spin_cell_is_refused(self, spin_u1):
        """No spin-1/2 configuration is neutral over an odd cell."""
        Spc, Op = spin_u1
        with pytest.raises(ValueError, match='no charge-neutral configuration'):
            _auto_config(3, Spc, Spc.group, _q_vac(Op))


# ---------------------------------------------------------------------------
# Unit tests: _even_neighborhood
# ---------------------------------------------------------------------------

class TestEvenNeighborhood:
    """Verify the charges a bond is allowed to carry at its own position."""

    def test_contains_the_charge_itself(self, spin_u1):
        """The charge a bond already holds is always reachable."""
        Spc, _ = spin_u1
        assert 0 in _even_neighborhood(Spc.group, Spc, 0)

    def test_spin_u1_keeps_the_parity(self, spin_u1):
        """Two U1 steps of +-1 shift the charge by an even amount."""
        Spc, _ = spin_u1
        assert _even_neighborhood(Spc.group, Spc, 0) == {-2, 0, 2}
        assert _even_neighborhood(Spc.group, Spc, 1) == {-1, 1, 3}

    def test_more_steps_reach_further(self, spin_u1):
        """Each double step widens the set it returns."""
        Spc, _ = spin_u1
        near = _even_neighborhood(Spc.group, Spc, 0, steps=1)
        far = _even_neighborhood(Spc.group, Spc, 0, steps=2)
        assert near < far


# ---------------------------------------------------------------------------
# Integration tests: product mode (bond_dim = 1)
# ---------------------------------------------------------------------------

class TestProductCell:
    """Integration tests for init_imps with bond_dim=1."""

    @pytest.mark.parametrize('space_fixture', ['spin_u1', 'spin_su2', 'ferm_u1'])
    def test_returns_gauge_fixed_cell(self, space_fixture, request):
        """init_imps(bond_dim=1) returns a cell gauge fixed at site 0."""
        Spc, Op = request.getfixturevalue(space_fixture)
        imps = init_imps(2, Spc, Op)

        assert isinstance(imps, iMPS)
        assert imps.N == 2
        assert imps.center == 0

    @pytest.mark.parametrize('N', [2, 4])
    def test_every_bond_is_one_wide(self, N, spin_u1):
        """A product cell carries a single sector of width one on each bond."""
        Spc, Op = spin_u1
        imps = init_imps(N, Spc, Op)

        assert imps.bond_dims == [1] * N

    def test_schmidt_weights_are_normalized(self, spin_u1):
        """The bond matrix of a product cell is the unit weight."""
        Spc, Op = spin_u1
        imps = init_imps(2, Spc, Op)

        for bond in range(imps.N):
            weights = imps.schmidt_values(bond)
            assert abs(sum(w * w for w in weights) - 1.0) < 1e-10

    def test_seed_does_not_matter(self, spin_u1):
        """Product mode is deterministic, so the seed is ignored."""
        Spc, Op = spin_u1
        first = init_imps(2, Spc, Op, seed=1)
        second = init_imps(2, Spc, Op, seed=99)

        assert first.bond_dims == second.bond_dims

    def test_complex_dtype(self, spin_u1):
        """dtype=complex128 produces a complex cell."""
        Spc, Op = spin_u1
        imps = init_imps(2, Spc, Op, dtype=torch.complex128)

        assert all(t.dtype == torch.complex128 for t in imps)

    def test_explicit_config_sets_the_pattern(self, spin_u1):
        """An explicit neutral config is followed site by site."""
        Spc, Op = spin_u1
        imps = init_imps(4, Spc, Op, config=[1, 1, 0, 0])

        assert imps.N == 4
        assert imps.bond_dims == [1] * 4


# ---------------------------------------------------------------------------
# Integration tests: random mode (bond_dim > 1)
# ---------------------------------------------------------------------------

class TestRandomCell:
    """Integration tests for init_imps with bond_dim > 1."""

    @pytest.mark.parametrize('space_fixture', ['spin_u1', 'spin_su2', 'ferm_u1'])
    def test_returns_gauge_fixed_cell(self, space_fixture, request):
        """A random cell comes back gauge fixed and wider than one."""
        Spc, Op = request.getfixturevalue(space_fixture)
        imps = init_imps(2, Spc, Op, bond_dim=8)

        assert isinstance(imps, iMPS)
        assert imps.center == 0
        assert min(imps.bond_dims) > 1

    @pytest.mark.parametrize('N', [2, 4])
    def test_bonds_are_uniform(self, N, spin_u1):
        """Every bond of the cell comes out the same width."""
        Spc, Op = spin_u1
        imps = init_imps(N, Spc, Op, bond_dim=8)

        assert len(set(imps.bond_dims)) == 1

    def test_same_seed_repeats(self, spin_u1):
        """Two cells drawn with the same seed hold the same state."""
        Spc, Op = spin_u1
        first = init_imps(2, Spc, Op, bond_dim=8, seed=7)
        second = init_imps(2, Spc, Op, bond_dim=8, seed=7)

        assert first.schmidt_values(0) == pytest.approx(second.schmidt_values(0))

    def test_another_seed_draws_another_cell(self, spin_u1):
        """A different seed lands on a different state."""
        Spc, Op = spin_u1
        first = init_imps(2, Spc, Op, bond_dim=8, seed=7)
        second = init_imps(2, Spc, Op, bond_dim=8, seed=8)

        assert first.schmidt_values(0) != pytest.approx(second.schmidt_values(0))


# ---------------------------------------------------------------------------
# What init_imps refuses
# ---------------------------------------------------------------------------

class TestRefusals:
    """Verify the arguments and cells init_imps turns down."""

    def test_empty_cell(self, spin_u1):
        """A cell needs at least one site."""
        Spc, Op = spin_u1
        with pytest.raises(ValueError, match='N must be >= 1'):
            init_imps(0, Spc, Op)

    def test_empty_bond(self, spin_u1):
        """A bond needs at least one state."""
        Spc, Op = spin_u1
        with pytest.raises(ValueError, match='bond_dim must be >= 1'):
            init_imps(2, Spc, Op, bond_dim=0)

    def test_config_of_the_wrong_length(self, spin_u1):
        """An explicit config covers every site of the cell, and no more."""
        Spc, Op = spin_u1
        with pytest.raises(ValueError, match='config has length 3, expected 2'):
            init_imps(2, Spc, Op, config=[0, 1, 0])

    def test_config_that_does_not_tile(self, spin_u1):
        """A config carrying net charge over the cell is turned down."""
        Spc, Op = spin_u1
        with pytest.raises(ValueError, match='does not tile'):
            init_imps(2, Spc, Op, config=[0, 0])

    def test_cell_length_that_cannot_be_neutral(self, spin_u1):
        """An odd spin-1/2 cell is refused rather than built."""
        Spc, Op = spin_u1
        with pytest.raises(ValueError, match='no charge-neutral configuration'):
            init_imps(3, Spc, Op)
