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


"""Tests for `intrcmap_square` — the snake-order square-lattice geometry builder."""

from __future__ import annotations

import logging

import pytest

from alice.physics.geometry import intrcmap_square


# Suppress INFO-level geometry logs during tests.
@pytest.fixture(autouse=True)
def _quiet_geometry(caplog):
    with caplog.at_level(logging.WARNING, logger='alice.physics.geometry'):
        yield


def _geo(lx, ly, *, bcx='OBC', bcy='OBC', n2x=True, n2y=True, n3d=False, n3o=False):
    """Convenience factory for a geometry sub-dict."""
    return {
        'lx': lx, 'ly': ly,
        'bcx': bcx, 'bcy': bcy,
        'n2x': n2x, 'n2y': n2y,
        'n3d': n3d, 'n3o': n3o,
    }


# ---------------------------------------------------------------------------
# Nearest-neighbor bonds
# ---------------------------------------------------------------------------

class TestNearestNeighbor:
    """Tests for NN bond generation."""

    def test_1d_chain_n2x_label(self):
        """1D chain (ly=1) bonds must carry the N2X label, not N2Y."""
        geo = _geo(lx=10, ly=1)
        interactions = intrcmap_square(geo)

        assert len(interactions) == 9
        for intr in interactions:
            assert 'NN'  in intr.label
            assert 'N2X' in intr.label, (
                f"Expected N2X but got {intr.label} for site "
                f"{intr.leading_site}→{intr.terminal_site}"
            )
            assert 'N2Y' not in intr.label

    def test_2d_square_obc(self):
        """2D square lattice with OBC has the right NN count per direction."""
        lx, ly = 6, 4
        geo = _geo(lx=lx, ly=ly)
        interactions = intrcmap_square(geo)

        x_intrs = [i for i in interactions if 'N2X' in i.label]
        y_intrs = [i for i in interactions if 'N2Y' in i.label]

        assert len(x_intrs) == ly * (lx - 1)
        assert len(y_intrs) == lx * (ly - 1)
        assert len(interactions) == len(x_intrs) + len(y_intrs)

    @pytest.mark.parametrize("lx,ly,expected_nn", [
        (2, 2, 4),
        (3, 2, 7),
        (4, 3, 17),
    ])
    def test_nn_count_obc(self, lx, ly, expected_nn):
        """NN count matches analytic formula for OBC square lattice."""
        interactions = intrcmap_square(_geo(lx=lx, ly=ly))
        assert len(interactions) == expected_nn

    def test_n2x_flag_off(self):
        """Setting n2x=False suppresses all x-direction NN bonds."""
        geo = _geo(lx=4, ly=3, n2x=False)
        interactions = intrcmap_square(geo)
        assert all('N2X' not in i.label for i in interactions)
        assert any('N2Y' in i.label for i in interactions)

    def test_n2y_flag_off(self):
        """Setting n2y=False suppresses all y-direction NN bonds."""
        geo = _geo(lx=4, ly=3, n2y=False)
        interactions = intrcmap_square(geo)
        assert all('N2Y' not in i.label for i in interactions)
        assert any('N2X' in i.label for i in interactions)

    def test_both_nn_flags_off_2d(self):
        """With n2x=n2y=False on a 2D lattice, no NN bonds are generated."""
        geo = _geo(lx=3, ly=3, n2x=False, n2y=False)
        interactions = intrcmap_square(geo)
        nn = [i for i in interactions if 'NN' in i.label]
        assert len(nn) == 0


# ---------------------------------------------------------------------------
# Periodic boundary conditions
# ---------------------------------------------------------------------------

class TestPeriodicBoundary:
    """Tests for PBC bond generation."""

    def test_pbc_y_cylinder(self):
        """PBC along Y (cylinder) adds lx PBC bonds, all tagged N2Y."""
        lx = 8
        geo = _geo(lx=lx, ly=4, bcy='PBC')
        interactions = intrcmap_square(geo)

        pbc = [i for i in interactions if 'PBC' in i.label]
        assert len(pbc) == lx
        for intr in pbc:
            assert 'N2Y' in intr.label

    def test_pbc_x(self):
        """PBC along X adds ly PBC bonds, all tagged N2X."""
        ly = 2
        geo = _geo(lx=3, ly=ly, bcx='PBC')
        interactions = intrcmap_square(geo)

        pbc = [i for i in interactions if 'PBC' in i.label]
        assert len(pbc) == ly
        for intr in pbc:
            assert 'N2X' in intr.label

    def test_pbc_torus(self):
        """Torus has PBC bonds in both x and y directions."""
        geo = _geo(lx=8, ly=6, bcx='PBC', bcy='PBC')
        interactions = intrcmap_square(geo)

        pbc = [i for i in interactions if 'PBC' in i.label]
        pbc_x = [i for i in pbc if 'N2X' in i.label]
        pbc_y = [i for i in pbc if 'N2Y' in i.label]

        assert len(pbc_x) > 0
        assert len(pbc_y) > 0


# ---------------------------------------------------------------------------
# Next-nearest-neighbor bonds
# ---------------------------------------------------------------------------

class TestNextNearestNeighbor:
    """Tests for NNN bond generation via n3d / n3o flags."""

    def test_n3d_and_n3o(self):
        """Both NNN types enabled: (lx-1)*(ly-1) bonds each."""
        lx, ly = 6, 6
        geo = _geo(lx=lx, ly=ly, n3d=True, n3o=True)
        interactions = intrcmap_square(geo)

        nnn_d = [i for i in interactions if 'N3D' in i.label]
        nnn_o = [i for i in interactions if 'N3O' in i.label]

        expected = (lx - 1) * (ly - 1)
        assert len(nnn_d) == expected
        assert len(nnn_o) == expected

    def test_n3d_only(self):
        """Only diagonal NNN bonds when n3d=True, n3o=False."""
        geo = _geo(lx=3, ly=3, n3d=True, n3o=False)
        interactions = intrcmap_square(geo)

        assert len([i for i in interactions if 'N3D' in i.label]) == 4
        assert len([i for i in interactions if 'N3O' in i.label]) == 0

    def test_n3o_only(self):
        """Only off-diagonal NNN bonds when n3o=True, n3d=False."""
        geo = _geo(lx=3, ly=3, n3d=False, n3o=True)
        interactions = intrcmap_square(geo)

        assert len([i for i in interactions if 'N3D' in i.label]) == 0
        assert len([i for i in interactions if 'N3O' in i.label]) == 4

    def test_nnn_with_pbc(self):
        """NNN bonds with PBC produce additional PBC-tagged NNN interactions."""
        geo = _geo(lx=3, ly=3, bcx='PBC', bcy='PBC', n3d=True, n3o=True)
        interactions = intrcmap_square(geo)

        nnn_pbc = [i for i in interactions if 'NNN' in i.label and 'PBC' in i.label]
        assert len(nnn_pbc) > 0

    def test_no_nnn_by_default(self):
        """Default flags produce no NNN interactions."""
        geo = _geo(lx=4, ly=4)
        interactions = intrcmap_square(geo)
        assert all('NNN' not in i.label for i in interactions)


# ---------------------------------------------------------------------------
# Coupling and structural invariants
# ---------------------------------------------------------------------------

class TestCouplingAndStructure:
    """Verify cpl==0.0 and structural properties of returned interactions."""

    def test_cpl_always_zero(self):
        """Geometry stage must never set a non-zero coupling."""
        geo = _geo(lx=6, ly=4, bcx='PBC', bcy='PBC', n3d=True, n3o=True)
        interactions = intrcmap_square(geo)
        for intr in interactions:
            assert intr.cpl == 0.0, (
                f"Expected cpl=0.0 but got {intr.cpl} for bond "
                f"{intr.leading_site}→{intr.terminal_site} (labels={intr.label})"
            )

    def test_tensors_are_none(self):
        """Geometry stage must not populate any tensor fields."""
        geo = _geo(lx=4, ly=3)
        interactions = intrcmap_square(geo)
        for intr in interactions:
            assert intr.leading_tnsr  is None
            assert intr.terminal_tnsr is None
            assert intr.intermid_tnsr is None

    def test_sorted_by_leading_site(self):
        """Interactions are sorted by leading_site in non-decreasing order."""
        geo = _geo(lx=8, ly=6, bcx='PBC', bcy='PBC', n3d=True, n3o=True)
        interactions = intrcmap_square(geo)
        for i in range(len(interactions) - 1):
            assert interactions[i].leading_site <= interactions[i + 1].leading_site

    def test_leading_less_than_terminal(self):
        """Every bond satisfies leading_site < terminal_site."""
        geo = _geo(lx=6, ly=4, n3d=True, n3o=True)
        interactions = intrcmap_square(geo)
        for intr in interactions:
            assert intr.leading_site < intr.terminal_site, (
                f"Reversed bond: {intr.leading_site}→{intr.terminal_site}"
            )

    def test_label_is_nonempty_list(self):
        """Every interaction carries at least one string label."""
        geo = _geo(lx=4, ly=3, n3d=True, n3o=True)
        interactions = intrcmap_square(geo)
        for intr in interactions:
            assert isinstance(intr.label, list)
            assert len(intr.label) > 0
