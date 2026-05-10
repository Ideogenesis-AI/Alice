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


"""Tests for `kagome.py`: traversal builder and `intrcmap_kagome`."""

from __future__ import annotations

import logging

import pytest

from alice.physics.geometry import build_geometry, build_intrcmap
from alice.physics.kagome import (
    build_traversal_serpentine,
    build_traversal_sequential,
    intrcmap_kagome,
)


# Suppress INFO-level geometry logs during tests.
@pytest.fixture(autouse=True)
def _quiet_geometry(caplog):
    with caplog.at_level(logging.WARNING, logger='alice.physics'):
        yield


# ---------------------------------------------------------------------------
# Kagome sequential traversal
# ---------------------------------------------------------------------------

class TestKagomeSequentialTraversal:
    """Tests for `build_traversal_sequential` for the Kagome lattice."""

    def test_3x3_sequential(self):
        """Sequential ord_map matches the reference for lx=3, ly=3."""
        ord_map, latt = build_traversal_sequential(lx=3, ly=3)

        # Col 0: rows 0→1→2, sites 0–8
        assert ord_map[(0, 0, 0)] == 0
        assert ord_map[(0, 0, 1)] == 1
        assert ord_map[(0, 0, 2)] == 2
        assert ord_map[(2, 0, 0)] == 6
        assert ord_map[(2, 0, 2)] == 8

        # Col 1: rows 0→1→2, sites 9–17
        assert ord_map[(0, 1, 0)] == 9
        assert ord_map[(0, 1, 1)] == 10
        assert ord_map[(0, 1, 2)] == 11
        assert ord_map[(2, 1, 0)] == 15
        assert ord_map[(2, 1, 2)] == 17

        # Col 2: rows 0→1→2, sites 18–26
        assert ord_map[(0, 2, 0)] == 18
        assert ord_map[(2, 2, 2)] == 26

        assert latt[0]  == (0, 0, 0)
        assert latt[9]  == (0, 1, 0)
        assert latt[18] == (0, 2, 0)

    def test_latt_inverses(self):
        """latt[site] is the inverse of ord_map for lx=3, ly=3."""
        ord_map, latt = build_traversal_sequential(lx=3, ly=3)
        for coord, site in ord_map.items():
            assert latt[site] == coord

    @pytest.mark.parametrize("lx,ly", [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 2),
        (2, 4),
    ])
    def test_total_sites(self, lx, ly):
        """ord_map has lx*ly*3 entries, all values unique; latt has the same length."""
        expected = lx * ly * 3
        ord_map, latt = build_traversal_sequential(lx=lx, ly=ly)
        assert len(ord_map) == expected
        assert len(set(ord_map.values())) == expected
        assert len(latt) == expected

    def test_keys_are_3_tuples(self):
        """All ord_map keys are 3-tuples (row, col, u) with u in {0, 1, 2}."""
        ord_map, _ = build_traversal_sequential(lx=3, ly=2)
        for key in ord_map:
            assert isinstance(key, tuple) and len(key) == 3
            row, col, u = key
            assert 0 <= row < 2
            assert 0 <= col < 3
            assert u in (0, 1, 2)

    def test_values_cover_0_to_L_minus_1(self):
        """ord_map values form a complete permutation of 0 .. L-1."""
        lx, ly = 3, 2
        L = lx * ly * 3
        ord_map, _ = build_traversal_sequential(lx=lx, ly=ly)
        assert set(ord_map.values()) == set(range(L))

    def test_all_cols_top_to_bottom(self):
        """Every column fills rows 0→ly-1 with consecutive A,B,C triples."""
        lx, ly = 4, 3
        ord_map, _ = build_traversal_sequential(lx=lx, ly=ly)
        for col in range(lx):
            base = col * ly * 3
            for row in range(ly):
                assert ord_map[(row, col, 0)] == base + row * 3
                assert ord_map[(row, col, 1)] == base + row * 3 + 1
                assert ord_map[(row, col, 2)] == base + row * 3 + 2

    def test_differs_from_serpentine_for_2d(self):
        """Sequential and serpentine produce different ord_maps for a 2D lattice."""
        serpentine_map, _ = build_traversal_serpentine(lx=3, ly=3)
        sequential_map, _ = build_traversal_sequential(lx=3, ly=3)
        assert serpentine_map != sequential_map

    def test_same_as_serpentine_for_ly_1(self):
        """For ly=1 sequential and serpentine are identical."""
        serpentine_map, _ = build_traversal_serpentine(lx=4, ly=1)
        sequential_map, _ = build_traversal_sequential(lx=4, ly=1)
        assert serpentine_map == sequential_map


# ---------------------------------------------------------------------------
# Kagome serpentine traversal
# ---------------------------------------------------------------------------

class TestKagomeSerpentineTraversal:
    """Tests for `build_traversal_serpentine` for the Kagome lattice."""

    def test_3x3_serpentine(self):
        """Serpentine ord_map matches the plan reference for lx=3, ly=3."""
        ord_map, latt = build_traversal_serpentine(lx=3, ly=3)

        # Col 0 (even): rows 0→1→2, sites 0–8
        assert ord_map[(0, 0, 0)] == 0
        assert ord_map[(0, 0, 1)] == 1
        assert ord_map[(0, 0, 2)] == 2
        assert ord_map[(2, 0, 0)] == 6
        assert ord_map[(2, 0, 2)] == 8

        # Col 1 (odd): rows 2→1→0, sites 9–17
        assert ord_map[(2, 1, 0)] == 9
        assert ord_map[(2, 1, 1)] == 10
        assert ord_map[(2, 1, 2)] == 11
        assert ord_map[(0, 1, 0)] == 15
        assert ord_map[(0, 1, 2)] == 17

        # Col 2 (even): rows 0→1→2, sites 18–26
        assert ord_map[(0, 2, 0)] == 18
        assert ord_map[(2, 2, 2)] == 26

        assert latt[0]  == (0, 0, 0)
        assert latt[9]  == (2, 1, 0)
        assert latt[18] == (0, 2, 0)

    def test_latt_inverses(self):
        """latt[site] is the inverse of ord_map for lx=3, ly=3."""
        ord_map, latt = build_traversal_serpentine(lx=3, ly=3)
        for coord, site in ord_map.items():
            assert latt[site] == coord

    @pytest.mark.parametrize("lx,ly", [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 2),
        (2, 4),
    ])
    def test_total_sites(self, lx, ly):
        """ord_map has lx*ly*3 entries, all values unique; latt has the same length."""
        expected = lx * ly * 3
        ord_map, latt = build_traversal_serpentine(lx=lx, ly=ly)
        assert len(ord_map) == expected
        assert len(set(ord_map.values())) == expected
        assert len(latt) == expected

    def test_keys_are_3_tuples(self):
        """All ord_map keys are 3-tuples (row, col, u) with u in {0, 1, 2}."""
        ord_map, _ = build_traversal_serpentine(lx=3, ly=2)
        for key in ord_map:
            assert isinstance(key, tuple) and len(key) == 3
            row, col, u = key
            assert 0 <= row < 2
            assert 0 <= col < 3
            assert u in (0, 1, 2)

    def test_values_cover_0_to_L_minus_1(self):
        """ord_map values form a complete permutation of 0 .. L-1."""
        lx, ly = 3, 2
        L = lx * ly * 3
        ord_map, _ = build_traversal_serpentine(lx=lx, ly=ly)
        assert set(ord_map.values()) == set(range(L))

    def test_even_col_top_to_bottom(self):
        """Even columns fill rows 0→ly-1 with consecutive A,B,C triples."""
        lx, ly = 4, 3
        ord_map, _ = build_traversal_serpentine(lx=lx, ly=ly)
        for col in range(0, lx, 2):
            base = col * ly * 3
            for row in range(ly):
                assert ord_map[(row, col, 0)] == base + row * 3
                assert ord_map[(row, col, 1)] == base + row * 3 + 1
                assert ord_map[(row, col, 2)] == base + row * 3 + 2

    def test_odd_col_bottom_to_top(self):
        """Odd columns fill rows ly-1→0 with consecutive A,B,C triples."""
        lx, ly = 4, 3
        ord_map, _ = build_traversal_serpentine(lx=lx, ly=ly)
        for col in range(1, lx, 2):
            base = col * ly * 3
            for row in range(ly):
                assert ord_map[(row, col, 0)] == base + (ly - 1 - row) * 3
                assert ord_map[(row, col, 1)] == base + (ly - 1 - row) * 3 + 1
                assert ord_map[(row, col, 2)] == base + (ly - 1 - row) * 3 + 2


# ---------------------------------------------------------------------------
# Geometry round-trip via build_geometry
# ---------------------------------------------------------------------------

class TestKagomeGeometry:
    """Tests for `build_geometry` and `to_1d`/`to_2d` with Kagome."""

    def test_returns_correct_latt_length(self):
        """len(geo.latt) == lx * ly * 3 (distinct from geo.L == lx * ly)."""
        geo = build_geometry({'lx': 3, 'ly': 3, 'lattice': 'kagome'})
        assert len(geo.latt) == 3 * 3 * 3
        assert len(geo.ord_map) == 3 * 3 * 3

    def test_to_1d_to_2d_roundtrip(self):
        """geo.to_2d(geo.to_1d((row, col, u))) == (row, col, u) for all sites."""
        geo = build_geometry({'lx': 3, 'ly': 3, 'lattice': 'kagome'})
        for row in range(geo.ly):
            for col in range(geo.lx):
                for u in range(3):
                    site = geo.to_1d((row, col, u))
                    assert geo.to_2d(site) == (row, col, u)

    def test_ord_map_keys_are_3_tuples(self):
        """geo.ord_map keys are all 3-tuples."""
        geo = build_geometry({'lx': 2, 'ly': 2, 'lattice': 'kagome'})
        assert all(isinstance(k, tuple) and len(k) == 3 for k in geo.ord_map)

    def test_unknown_traverse_raises(self):
        """build_geometry raises ValueError for an unrecognised traverse key."""
        with pytest.raises(ValueError, match="Unknown traversal"):
            build_geometry({'lx': 2, 'ly': 2, 'lattice': 'kagome', 'traverse': 'hilbert'})


# ---------------------------------------------------------------------------
# intrcmap_kagome — bond counts and properties
# ---------------------------------------------------------------------------

class TestIntrcmapKagome:
    """Tests for `intrcmap_kagome` via `build_intrcmap`."""

    def _geo(self, **kwargs):
        defaults = {'lx': 3, 'ly': 3, 'lattice': 'kagome',
                    'bcx': 'OBC', 'bcy': 'OBC', 'n2u': True, 'n2d': True}
        defaults.update(kwargs)
        return build_geometry(defaults)

    # --- N2U bonds -----------------------------------------------------------

    @pytest.mark.parametrize("lx,ly", [(2, 2), (3, 3), (4, 2)])
    def test_n2u_count_obc(self, lx, ly):
        """N2U bonds == 3 * lx * ly for OBC (one upward triangle per unit cell)."""
        geo = self._geo(lx=lx, ly=ly, n2u=True, n2d=False)
        ints = build_intrcmap(geo)
        assert len(ints) == 3 * lx * ly

    def test_n2u_labels(self):
        """All N2U interactions carry labels ['NN', 'N2U']."""
        geo = self._geo(n2u=True, n2d=False)
        ints = build_intrcmap(geo)
        for intr in ints:
            assert intr.label == ['NN', 'N2U']

    def test_n2u_off(self):
        """Setting n2u=False excludes all upward-triangle bonds."""
        geo = self._geo(n2u=False, n2d=False)
        assert len(build_intrcmap(geo)) == 0

    # --- N2D bonds (OBC) -----------------------------------------------------

    @pytest.mark.parametrize("lx,ly", [(2, 2), (3, 3), (4, 2), (2, 4)])
    def test_n2d_count_obc(self, lx, ly):
        """N2D bonds (OBC) == (lx-1)*ly + lx*(ly-1) + (lx-1)*(ly-1)."""
        expected = (lx - 1) * ly + lx * (ly - 1) + (lx - 1) * (ly - 1)
        geo = self._geo(lx=lx, ly=ly, n2u=False, n2d=True)
        assert len(build_intrcmap(geo)) == expected

    def test_n2d_labels_obc(self):
        """All OBC N2D interactions carry labels ['NN', 'N2D']."""
        geo = self._geo(n2u=False, n2d=True)
        ints = build_intrcmap(geo)
        for intr in ints:
            assert intr.label == ['NN', 'N2D']

    def test_n2d_off(self):
        """Setting n2d=False excludes all downward-triangle bonds."""
        geo = self._geo(n2u=False, n2d=False)
        assert len(build_intrcmap(geo)) == 0

    # --- PBC bonds -----------------------------------------------------------

    def test_bcx_pbc_bond4_count(self):
        """bcx=PBC adds ly extra bond-4 wraps."""
        lx, ly = 3, 3
        geo_obc = self._geo(lx=lx, ly=ly, n2u=False, n2d=True, bcx='OBC', bcy='OBC')
        geo_pbc = self._geo(lx=lx, ly=ly, n2u=False, n2d=True, bcx='PBC', bcy='OBC')
        n_obc = len(build_intrcmap(geo_obc))
        n_pbc = len(build_intrcmap(geo_pbc))
        # bcx PBC adds: ly bond-4 wraps + (ly-1) bond-6 wraps
        assert n_pbc - n_obc == ly + (ly - 1)

    def test_bcy_pbc_bond5_count(self):
        """bcy=PBC adds lx extra bond-5 wraps and lx-1 extra bond-6 wraps."""
        lx, ly = 3, 3
        geo_obc = self._geo(lx=lx, ly=ly, n2u=False, n2d=True, bcx='OBC', bcy='OBC')
        geo_pbc = self._geo(lx=lx, ly=ly, n2u=False, n2d=True, bcx='OBC', bcy='PBC')
        n_obc = len(build_intrcmap(geo_obc))
        n_pbc = len(build_intrcmap(geo_pbc))
        # bcy PBC adds: lx bond-5 wraps + (lx-1) bond-6 wraps
        assert n_pbc - n_obc == lx + (lx - 1)

    def test_bcx_bcy_pbc_corner_bond(self):
        """bcx+bcy PBC adds one extra corner bond on top of each individual PBC."""
        lx, ly = 3, 3
        geo_bcx = self._geo(lx=lx, ly=ly, n2u=False, n2d=True, bcx='PBC', bcy='OBC')
        geo_bcy = self._geo(lx=lx, ly=ly, n2u=False, n2d=True, bcx='OBC', bcy='PBC')
        geo_both = self._geo(lx=lx, ly=ly, n2u=False, n2d=True, bcx='PBC', bcy='PBC')
        geo_obc  = self._geo(lx=lx, ly=ly, n2u=False, n2d=True, bcx='OBC', bcy='OBC')
        n_bcx  = len(build_intrcmap(geo_bcx))
        n_bcy  = len(build_intrcmap(geo_bcy))
        n_both = len(build_intrcmap(geo_both))
        n_obc  = len(build_intrcmap(geo_obc))
        # Inclusion-exclusion: both = bcx + bcy - obc + 1 (corner bond)
        assert n_both == n_bcx + n_bcy - n_obc + 1

    @pytest.mark.parametrize("lx,ly", [(2, 2), (3, 3), (3, 2)])
    def test_full_pbc_total_bonds(self, lx, ly):
        """Full PBC (n2u+n2d) gives 6 * lx * ly bonds (6 NN bonds per unit cell)."""
        geo = self._geo(lx=lx, ly=ly, n2u=True, n2d=True, bcx='PBC', bcy='PBC')
        assert len(build_intrcmap(geo)) == 6 * lx * ly

    def test_pbc_labels_contain_pbc(self):
        """PBC N2D bonds all carry 'PBC' in their label."""
        geo = self._geo(lx=3, ly=3, n2u=False, n2d=True, bcx='PBC', bcy='PBC')
        ints = build_intrcmap(geo)
        pbc_bonds = [i for i in ints if 'PBC' in i.label]
        assert len(pbc_bonds) > 0
        for intr in pbc_bonds:
            assert 'NN' in intr.label
            assert 'N2D' in intr.label

    # --- General properties --------------------------------------------------

    def test_cpl_zero(self):
        """All interactions have cpl == 0.0 (set by model builder, not here)."""
        geo = self._geo()
        assert all(intr.cpl == 0.0 for intr in build_intrcmap(geo))

    def test_sorted_by_leading_site(self):
        """Interactions are sorted in ascending order of leading_site."""
        geo = self._geo()
        ints = build_intrcmap(geo)
        leading = [i.leading_site for i in ints]
        assert leading == sorted(leading)

    def test_leading_lt_terminal(self):
        """leading_site < terminal_site for all interactions."""
        geo = self._geo()
        for intr in build_intrcmap(geo):
            assert intr.leading_site < intr.terminal_site

    def test_all_labels_contain_nn(self):
        """Every interaction has 'NN' as the first label entry."""
        geo = self._geo()
        for intr in build_intrcmap(geo):
            assert intr.label[0] == 'NN'

    def test_no_duplicate_bonds(self):
        """No (leading_site, terminal_site) pair appears twice."""
        geo = self._geo()
        pairs = [(i.leading_site, i.terminal_site) for i in build_intrcmap(geo)]
        assert len(pairs) == len(set(pairs))

    def test_minimal_1x1(self):
        """1×1 Kagome unit cell (OBC) has exactly 3 N2U bonds and 0 N2D bonds."""
        geo = build_geometry({'lx': 1, 'ly': 1, 'lattice': 'kagome',
                              'bcx': 'OBC', 'bcy': 'OBC', 'n2u': True, 'n2d': True})
        ints = build_intrcmap(geo)
        assert len(ints) == 3   # only A–B, A–C, B–C; no inter-cell bonds
        assert all(intr.label == ['NN', 'N2U'] for intr in ints)

    def test_dispatch_via_build_geometry(self):
        """intrcmap_kagome is reachable via the public build_intrcmap dispatcher."""
        geo = build_geometry({
            'lx': 3, 'ly': 2,
            'lattice': 'kagome',
            'bcx': 'OBC', 'bcy': 'OBC',
        })
        ints = build_intrcmap(geo)
        assert len(ints) > 0
        assert all(intr.cpl == 0.0 for intr in ints)
