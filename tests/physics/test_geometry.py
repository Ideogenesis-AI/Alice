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


"""Tests for `build_geometry` dispatcher and generic traversal utilities."""

from __future__ import annotations

import logging

import pytest

from alice.physics.geometry import generate_snake_order, build_geometry


# Suppress INFO-level geometry logs during tests.
@pytest.fixture(autouse=True)
def _quiet_geometry(caplog):
    with caplog.at_level(logging.WARNING, logger='alice.physics.geometry'):
        yield


# ---------------------------------------------------------------------------
# Snake order
# ---------------------------------------------------------------------------

class TestSnakeOrder:
    """Tests for `generate_snake_order`."""

    def test_2x3_lattice(self):
        """Snake order for a 2×3 lattice."""
        ord_map, latt = generate_snake_order(lx=2, ly=3)

        # Expected (0-based):
        # 00. . .05
        # |      |
        # 01. . .04
        # |      |
        # 02-----03
        expected_ord = [[0, 5], [1, 4], [2, 3]]
        assert ord_map == expected_ord
        assert latt[0] == (0, 0)
        assert latt[3] == (2, 1)
        assert latt[5] == (0, 1)

    @pytest.mark.parametrize("lx,ly,expected_sites", [
        (2, 2, 4),
        (3, 2, 6),
        (4, 3, 12),
        (1, 5, 5),
    ])
    def test_total_sites(self, lx, ly, expected_sites):
        """Total number of sites equals lx * ly; all indices are unique."""
        ord_map, latt = generate_snake_order(lx, ly)
        assert len(latt) == expected_sites
        flat = [s for row in ord_map for s in row]
        assert len(set(flat)) == expected_sites

    def test_1d_chain(self):
        """1D chain (ly=1) produces sequential snake order."""
        ord_map, latt = generate_snake_order(lx=5, ly=1)
        assert ord_map == [[0, 1, 2, 3, 4]]
        assert latt == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


# ---------------------------------------------------------------------------
# build_geometry dispatcher
# ---------------------------------------------------------------------------

class TestBuildGeometry:
    """Tests for the `build_geometry` public dispatcher."""

    def test_square_snake_dispatches(self):
        """build_geometry with lattice='square', traverse='snake' works."""
        geo = {
            'lx': 4, 'ly': 3,
            'bcx': 'OBC', 'bcy': 'OBC',
            'n2x': True, 'n2y': True, 'n3d': False, 'n3o': False,
            'lattice': 'square',
            'traverse': 'snake',
        }
        interactions = build_geometry(geo)
        assert len(interactions) > 0
        assert all(intr.cpl == 0.0 for intr in interactions)

    def test_default_traverse(self):
        """Missing traverse key defaults to 'snake'."""
        geo = {
            'lx': 3, 'ly': 2,
            'bcx': 'OBC', 'bcy': 'OBC',
            'n2x': True, 'n2y': True, 'n3d': False, 'n3o': False,
            'lattice': 'square',
        }
        interactions = build_geometry(geo)
        assert len(interactions) == 7  # 3×2 OBC: 4 x-bonds + 3 y-bonds

    def test_unknown_lattice_raises(self):
        """Unknown lattice name raises ValueError."""
        geo = {
            'lx': 2, 'ly': 2,
            'bcx': 'OBC', 'bcy': 'OBC',
            'n2x': True, 'n2y': True, 'n3d': False, 'n3o': False,
            'lattice': 'honeycomb',
        }
        with pytest.raises(ValueError, match="Unknown lattice"):
            build_geometry(geo)

    def test_unknown_traverse_raises(self):
        """Unknown traverse name raises ValueError."""
        geo = {
            'lx': 2, 'ly': 2,
            'bcx': 'OBC', 'bcy': 'OBC',
            'n2x': True, 'n2y': True, 'n3d': False, 'n3o': False,
            'lattice': 'square',
            'traverse': 'hilbert',
        }
        with pytest.raises(ValueError, match="Unknown traversal"):
            build_geometry(geo)
