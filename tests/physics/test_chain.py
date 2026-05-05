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


"""Tests for `intrcmap_1dchain` — the 1D chain geometry builder."""

from __future__ import annotations

import logging

import pytest

from alice.physics.geometry import intrcmap_1dchain


# Suppress INFO-level geometry logs during tests.
@pytest.fixture(autouse=True)
def _quiet_geometry(caplog):
    with caplog.at_level(logging.WARNING, logger='alice.physics.geometry'):
        yield


def _geo(lx, *, bcx='OBC', n2x=True):
    """Convenience factory for a 1D chain geometry sub-dict."""
    return {'lx': lx, 'bcx': bcx, 'n2x': n2x}


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

class TestChainTopology:
    """Bond counts and label checks for OBC and PBC chains."""

    @pytest.mark.parametrize("lx", [2, 5, 10, 20])
    def test_obc_bond_count(self, lx):
        """OBC chain of lx sites produces exactly lx-1 NN bonds."""
        interactions = intrcmap_1dchain(_geo(lx=lx))
        assert len(interactions) == lx - 1

    @pytest.mark.parametrize("lx", [2, 5, 10, 20])
    def test_pbc_bond_count(self, lx):
        """PBC chain of lx sites produces exactly lx bonds (lx-1 OBC + 1 PBC)."""
        interactions = intrcmap_1dchain(_geo(lx=lx, bcx='PBC'))
        assert len(interactions) == lx

    def test_n2x_false_produces_no_bonds(self):
        """n2x=False yields an empty interaction list."""
        assert intrcmap_1dchain(_geo(lx=10, n2x=False)) == []

    def test_obc_labels(self):
        """Every OBC bond carries exactly {'NN', 'N2X'}."""
        for intr in intrcmap_1dchain(_geo(lx=8)):
            assert set(intr.label) == {'NN', 'N2X'}

    def test_pbc_interior_labels(self):
        """Interior bonds in a PBC chain carry {'NN', 'N2X'}."""
        intrs = intrcmap_1dchain(_geo(lx=8, bcx='PBC'))
        interior = [i for i in intrs if 'PBC' not in i.label]
        for intr in interior:
            assert set(intr.label) == {'NN', 'N2X'}

    def test_pbc_bond_label(self):
        """The PBC closing bond carries exactly {'NN', 'PBC', 'N2X'}."""
        intrs = intrcmap_1dchain(_geo(lx=8, bcx='PBC'))
        pbc = [i for i in intrs if 'PBC' in i.label]
        assert len(pbc) == 1
        assert set(pbc[0].label) == {'NN', 'PBC', 'N2X'}

    def test_cpl_always_zero(self):
        """Geometry stage must never set a non-zero coupling."""
        for intr in intrcmap_1dchain(_geo(lx=10, bcx='PBC')):
            assert intr.cpl == 0.0

    def test_tensors_are_none(self):
        """Geometry stage must not populate any tensor fields."""
        for intr in intrcmap_1dchain(_geo(lx=6)):
            assert intr.leading_tnsr  is None
            assert intr.terminal_tnsr is None
            assert intr.intermid_tnsr is None

    def test_sorted_by_leading_site(self):
        """Interactions are returned sorted by leading_site."""
        intrs = intrcmap_1dchain(_geo(lx=10, bcx='PBC'))
        for i in range(len(intrs) - 1):
            assert intrs[i].leading_site <= intrs[i + 1].leading_site

    def test_leading_less_than_terminal(self):
        """Every bond satisfies leading_site < terminal_site."""
        for intr in intrcmap_1dchain(_geo(lx=10, bcx='PBC')):
            assert intr.leading_site < intr.terminal_site


# ---------------------------------------------------------------------------
# Exact bonds
# ---------------------------------------------------------------------------

class TestChainExactBonds:
    """Exact (leading, terminal) pairs for representative chain lengths."""

    def test_obc_10_sites(self):
        """10-site OBC chain: bonds are exactly (0,1), (1,2), …, (8,9)."""
        intrs = intrcmap_1dchain(_geo(lx=10))
        bonds = {(i.leading_site, i.terminal_site) for i in intrs}
        assert bonds == {(k, k + 1) for k in range(9)}

    def test_pbc_10_sites(self):
        """10-site PBC chain: OBC bonds plus the closing bond (0, 9)."""
        intrs = intrcmap_1dchain(_geo(lx=10, bcx='PBC'))
        bonds = {(i.leading_site, i.terminal_site) for i in intrs}
        expected = {(k, k + 1) for k in range(9)} | {(0, 9)}
        assert bonds == expected

    def test_pbc_closing_bond_sites(self):
        """The PBC bond connects site 0 to site lx-1."""
        lx = 7
        intrs = intrcmap_1dchain(_geo(lx=lx, bcx='PBC'))
        pbc = [i for i in intrs if 'PBC' in i.label]
        assert len(pbc) == 1
        assert pbc[0].leading_site  == 0
        assert pbc[0].terminal_site == lx - 1
