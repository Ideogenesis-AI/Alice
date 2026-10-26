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


"""Tests for the infinite unit-cell transfer operators."""

from __future__ import annotations

import pytest
from nicole import identity

from alice.algorithm.idmrg import transfer_left, transfer_right

from .conftest import random_imps

def _delta(idx, itag):
    """Build an identity on a bond index, tagged for both ends of a cell."""
    delta = identity(idx)
    delta.retag([0, 1], [itag, itag])
    return delta


def _deviation(a, b):
    """Return the norm of the difference between two tensors."""
    return float((a - b).norm())


# ---------------------------------------------------------------------------
# MPS transfer operators
# ---------------------------------------------------------------------------

class TestMPSTransfer:
    """Orthonormality seen from the transfer operator's point of view."""

    @pytest.mark.parametrize('N', [2, 4])
    def test_identity_is_fixed_under_left_isometries(self, spin_half, N):
        imps = random_imps(spin_half, N)
        isometries = imps.left_isometries()
        delta = _delta(isometries[0].indices[0], isometries[0].itags[0])
        assert _deviation(transfer_left(delta, isometries), delta) < 1e-12

    @pytest.mark.parametrize('N', [2, 4])
    def test_identity_is_fixed_under_right_isometries(self, spin_half, N):
        imps = random_imps(spin_half, N)
        isometries = imps.right_isometries()
        last = isometries[N - 1]
        delta = _delta(last.indices[1], last.itags[1])
        assert _deviation(transfer_right(delta, isometries), delta) < 1e-12

    def test_transfer_can_be_iterated_around_the_ring(self, spin_half):
        # The premise of the whole module is that the wrap-around itags let
        # a cell be absorbed over and over without retagging in between.
        # Five cells of an isometric state must still leave the identity.
        imps = random_imps(spin_half, 2)
        isometries = imps.left_isometries()
        delta = _delta(isometries[0].indices[0], isometries[0].itags[0])
        carried = delta
        for _ in range(5):
            carried = transfer_left(carried, isometries)
        assert _deviation(carried, delta) < 1e-11


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------
