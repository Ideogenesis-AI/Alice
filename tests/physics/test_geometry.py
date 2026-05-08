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


"""Tests for `build_geometry`, `build_intrcmap`, and traversal utilities."""

from __future__ import annotations

import logging

import pytest

from alice.physics.geometry import Geometry, build_geometry, build_intrcmap
from alice.physics.square import generate_snake_order, generate_zigzag_order


# Suppress INFO-level geometry logs during tests.
@pytest.fixture(autouse=True)
def _quiet_geometry(caplog):
    with caplog.at_level(logging.WARNING, logger='alice.physics'):
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
# Zigzag order
# ---------------------------------------------------------------------------

class TestZigzagOrder:
    """Tests for `generate_zigzag_order`."""

    def test_2x3_lattice(self):
        """Zigzag order for a 2×3 lattice."""
        ord_map, latt = generate_zigzag_order(lx=2, ly=3)

        # Expected (0-based): all columns top→bottom, no reversals.
        # 00. . .03
        # |      |
        # 01. . .04
        # |      |
        # 02. . .05
        expected_ord = [[0, 3], [1, 4], [2, 5]]
        assert ord_map == expected_ord
        assert latt[0] == (0, 0)
        assert latt[3] == (0, 1)
        assert latt[5] == (2, 1)

    @pytest.mark.parametrize("lx,ly,expected_sites", [
        (2, 2, 4),
        (3, 2, 6),
        (4, 3, 12),
        (1, 5, 5),
    ])
    def test_total_sites(self, lx, ly, expected_sites):
        """Total number of sites equals lx * ly; all indices are unique."""
        ord_map, latt = generate_zigzag_order(lx, ly)
        assert len(latt) == expected_sites
        flat = [s for row in ord_map for s in row]
        assert len(set(flat)) == expected_sites

    def test_1d_chain(self):
        """1D chain (ly=1) produces sequential zigzag order, same as snake."""
        ord_map, latt = generate_zigzag_order(lx=5, ly=1)
        assert ord_map == [[0, 1, 2, 3, 4]]
        assert latt == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

    def test_differs_from_snake_for_2d(self):
        """Zigzag and snake produce different ord_maps for any 2D lattice."""
        snake_map, _ = generate_snake_order(lx=4, ly=4)
        zigzag_map, _ = generate_zigzag_order(lx=4, ly=4)
        assert snake_map != zigzag_map

    def test_all_columns_top_to_bottom(self):
        """Every column in zigzag order has strictly increasing site indices top→bottom."""
        ord_map, _ = generate_zigzag_order(lx=5, ly=4)
        for col in range(5):
            col_indices = [ord_map[row][col] for row in range(4)]
            assert col_indices == sorted(col_indices)


# ---------------------------------------------------------------------------
# build_geometry — Geometry struct
# ---------------------------------------------------------------------------

class TestBuildGeometry:
    """Tests for `build_geometry`, which returns a `Geometry` dataclass."""

    def test_returns_geometry_instance(self):
        """build_geometry returns a Geometry instance."""
        geo_cfg = {'lx': 4, 'ly': 3, 'lattice': 'square', 'traverse': 'snake'}
        geo = build_geometry(geo_cfg)
        assert isinstance(geo, Geometry)

    def test_square_attrs(self):
        """Geometry has correct lx, ly, L, lattice, traverse for a square lattice."""
        geo_cfg = {'lx': 4, 'ly': 3, 'lattice': 'square', 'traverse': 'snake'}
        geo = build_geometry(geo_cfg)
        assert geo.lx      == 4
        assert geo.ly      == 3
        assert geo.L       == 12
        assert geo.lattice  == 'square'
        assert geo.traverse == 'snake'

    def test_chain_attrs(self):
        """Geometry for a chain has ly=1, L=lx."""
        geo_cfg = {'lx': 8, 'lattice': 'chain'}
        geo = build_geometry(geo_cfg)
        assert geo.lx == 8
        assert geo.ly == 1
        assert geo.L  == 8

    def test_ord_map_shape(self):
        """ord_map has shape ly × lx."""
        geo = build_geometry({'lx': 4, 'ly': 3, 'lattice': 'square'})
        assert len(geo.ord_map) == 3
        assert all(len(row) == 4 for row in geo.ord_map)

    def test_latt_length(self):
        """latt has length L."""
        geo = build_geometry({'lx': 4, 'ly': 3, 'lattice': 'square'})
        assert len(geo.latt) == 12

    def test_to_1d_to_2d_roundtrip(self):
        """to_2d(to_1d(row, col)) == (row, col) for all sites."""
        geo = build_geometry({'lx': 4, 'ly': 3, 'lattice': 'square'})
        for row in range(geo.ly):
            for col in range(geo.lx):
                site = geo.to_1d(row, col)
                assert geo.to_2d(site) == (row, col)

    def test_unknown_lattice_raises(self):
        """Unknown lattice name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown lattice"):
            build_geometry({'lx': 2, 'ly': 2, 'lattice': 'honeycomb'})

    def test_unknown_traverse_raises(self):
        """Unknown traverse name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown traversal"):
            build_geometry({'lx': 2, 'ly': 2, 'lattice': 'square', 'traverse': 'hilbert'})


# ---------------------------------------------------------------------------
# build_intrcmap dispatcher
# ---------------------------------------------------------------------------

class TestBuildIntrcmap:
    """Tests for the `build_intrcmap` public dispatcher."""

    def test_square_snake_dispatches(self):
        """build_intrcmap with lattice='square', traverse='snake' works."""
        geo_cfg = {
            'lx': 4, 'ly': 3,
            'bcx': 'OBC', 'bcy': 'OBC',
            'n2x': True, 'n2y': True, 'n3d': False, 'n3o': False,
            'lattice': 'square',
            'traverse': 'snake',
        }
        interactions = build_intrcmap(build_geometry(geo_cfg))
        assert len(interactions) > 0
        assert all(intr.cpl == 0.0 for intr in interactions)

    def test_chain_dispatches(self):
        """build_intrcmap with lattice='chain' produces lx-1 OBC bonds with cpl==0.0."""
        geo_cfg = {'lx': 8, 'bcx': 'OBC', 'n2x': True, 'lattice': 'chain'}
        interactions = build_intrcmap(build_geometry(geo_cfg))
        assert len(interactions) == 7
        assert all(intr.cpl == 0.0 for intr in interactions)

    def test_square_zigzag_dispatches(self):
        """build_intrcmap with lattice='square', traverse='zigzag' works."""
        geo_cfg = {
            'lx': 4, 'ly': 3,
            'bcx': 'OBC', 'bcy': 'OBC',
            'n2x': True, 'n2y': True, 'n3d': False, 'n3o': False,
            'lattice': 'square',
            'traverse': 'zigzag',
        }
        interactions = build_intrcmap(build_geometry(geo_cfg))
        assert len(interactions) > 0
        assert all(intr.cpl == 0.0 for intr in interactions)

    def test_default_traverse(self):
        """Missing traverse key defaults to 'snake'."""
        geo_cfg = {
            'lx': 3, 'ly': 2,
            'bcx': 'OBC', 'bcy': 'OBC',
            'n2x': True, 'n2y': True, 'n3d': False, 'n3o': False,
            'lattice': 'square',
        }
        interactions = build_intrcmap(build_geometry(geo_cfg))
        assert len(interactions) == 7  # 3×2 OBC: 4 x-bonds + 3 y-bonds
