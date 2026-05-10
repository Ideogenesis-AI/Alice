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
from alice.physics.square import build_traversal_sequential, build_traversal_serpentine


# Suppress INFO-level geometry logs during tests.
@pytest.fixture(autouse=True)
def _quiet_geometry(caplog):
    with caplog.at_level(logging.WARNING, logger='alice.physics'):
        yield


# ---------------------------------------------------------------------------
# Sequential order
# ---------------------------------------------------------------------------

class TestSequentialOrder:
    """Tests for `build_traversal_sequential`."""

    def test_2x3_lattice(self):
        """Sequential order for a 2×3 lattice."""
        ord_map, latt = build_traversal_sequential(lx=2, ly=3)

        # Expected (0-based): all columns top→bottom, no reversals.
        # 00. . .03
        # |      |
        # 01. . .04
        # |      |
        # 02. . .05
        assert ord_map[(0, 0)] == 0
        assert ord_map[(1, 0)] == 1
        assert ord_map[(2, 0)] == 2
        assert ord_map[(0, 1)] == 3
        assert ord_map[(1, 1)] == 4
        assert ord_map[(2, 1)] == 5
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
        ord_map, latt = build_traversal_sequential(lx, ly)
        assert len(latt) == expected_sites
        assert len(ord_map) == expected_sites
        assert len(set(ord_map.values())) == expected_sites

    def test_1d_chain(self):
        """1D chain (ly=1) produces sequential order, same as serpentine."""
        ord_map, latt = build_traversal_sequential(lx=5, ly=1)
        assert ord_map == {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4}
        assert latt == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

    def test_differs_from_serpentine_for_2d(self):
        """Sequential and serpentine produce different ord_maps for any 2D lattice."""
        serpentine_map, _ = build_traversal_serpentine(lx=4, ly=4)
        sequential_map, _ = build_traversal_sequential(lx=4, ly=4)
        assert serpentine_map != sequential_map

    def test_all_columns_top_to_bottom(self):
        """Every column in sequential order has strictly increasing site indices top→bottom."""
        ord_map, _ = build_traversal_sequential(lx=5, ly=4)
        for col in range(5):
            col_indices = [ord_map[(row, col)] for row in range(4)]
            assert col_indices == sorted(col_indices)


# ---------------------------------------------------------------------------
# Serpentine order
# ---------------------------------------------------------------------------

class TestSerpentineOrder:
    """Tests for `build_traversal_serpentine`."""

    def test_2x3_lattice(self):
        """Serpentine order for a 2×3 lattice."""
        ord_map, latt = build_traversal_serpentine(lx=2, ly=3)

        # Expected (0-based):
        # 00. . .05
        # |      |
        # 01. . .04
        # |      |
        # 02-----03
        assert ord_map[(0, 0)] == 0
        assert ord_map[(1, 0)] == 1
        assert ord_map[(2, 0)] == 2
        assert ord_map[(2, 1)] == 3
        assert ord_map[(1, 1)] == 4
        assert ord_map[(0, 1)] == 5
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
        ord_map, latt = build_traversal_serpentine(lx, ly)
        assert len(latt) == expected_sites
        assert len(ord_map) == expected_sites
        assert len(set(ord_map.values())) == expected_sites

    def test_1d_chain(self):
        """For ly=1, serpentine order is trivially sequential."""
        ord_map, latt = build_traversal_serpentine(lx=5, ly=1)
        assert ord_map == {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4}
        assert latt == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


# ---------------------------------------------------------------------------
# build_geometry — Geometry struct
# ---------------------------------------------------------------------------

class TestBuildGeometry:
    """Tests for `build_geometry`, which returns a `Geometry` dataclass."""

    def test_returns_geometry_instance(self):
        """build_geometry returns a Geometry instance."""
        geo_cfg = {'lx': 4, 'ly': 3, 'lattice': 'square', 'traverse': 'sequential'}
        geo = build_geometry(geo_cfg)
        assert isinstance(geo, Geometry)

    def test_square_attrs(self):
        """Geometry has correct lx, ly, L, lattice, traverse for a square lattice."""
        geo_cfg = {'lx': 4, 'ly': 3, 'lattice': 'square', 'traverse': 'sequential'}
        geo = build_geometry(geo_cfg)
        assert geo.lx      == 4
        assert geo.ly      == 3
        assert geo.L       == 12
        assert geo.lattice  == 'square'
        assert geo.traverse == 'sequential'

    def test_chain_attrs(self):
        """Geometry for a chain has ly=1, L=lx."""
        geo_cfg = {'lx': 8, 'lattice': 'chain'}
        geo = build_geometry(geo_cfg)
        assert geo.lx == 8
        assert geo.ly == 1
        assert geo.L  == 8

    def test_ord_map_shape(self):
        """ord_map has lx * ly entries with 2-tuple keys."""
        geo = build_geometry({'lx': 4, 'ly': 3, 'lattice': 'square'})
        assert len(geo.ord_map) == 12
        assert all(isinstance(k, tuple) and len(k) == 2 for k in geo.ord_map)

    def test_latt_length(self):
        """latt has length L."""
        geo = build_geometry({'lx': 4, 'ly': 3, 'lattice': 'square'})
        assert len(geo.latt) == 12

    def test_to_1d_to_2d_roundtrip(self):
        """to_2d(to_1d((row, col))) == (row, col) for all sites."""
        geo = build_geometry({'lx': 4, 'ly': 3, 'lattice': 'square'})
        for row in range(geo.ly):
            for col in range(geo.lx):
                site = geo.to_1d((row, col))
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

    def test_square_sequential_dispatches(self):
        """build_intrcmap with lattice='square', traverse='sequential' works."""
        geo_cfg = {
            'lx': 4, 'ly': 3,
            'bcx': 'OBC', 'bcy': 'OBC',
            'n2x': True, 'n2y': True, 'n3d': False, 'n3o': False,
            'lattice': 'square',
            'traverse': 'sequential',
        }
        interactions = build_intrcmap(build_geometry(geo_cfg))
        assert len(interactions) > 0
        assert all(intr.cpl == 0.0 for intr in interactions)

    def test_square_serpentine_dispatches(self):
        """build_intrcmap with lattice='square', traverse='serpentine' works."""
        geo_cfg = {
            'lx': 4, 'ly': 3,
            'bcx': 'OBC', 'bcy': 'OBC',
            'n2x': True, 'n2y': True, 'n3d': False, 'n3o': False,
            'lattice': 'square',
            'traverse': 'serpentine',
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

    def test_default_traverse(self):
        """Missing traverse key defaults to 'sequential'."""
        geo_cfg = {
            'lx': 3, 'ly': 2,
            'bcx': 'OBC', 'bcy': 'OBC',
            'n2x': True, 'n2y': True, 'n3d': False, 'n3o': False,
            'lattice': 'square',
        }
        interactions = build_intrcmap(build_geometry(geo_cfg))
        assert len(interactions) == 7  # 3×2 OBC: 4 x-bonds + 3 y-bonds
