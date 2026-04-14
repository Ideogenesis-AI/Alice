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


"""Unit tests for alice.network.observe."""

from __future__ import annotations

import math

import pytest

from alice.network import observe


class TestObserve:
    """Unit tests for the `observe` function."""

    def test_returns_float(self, mps_tensors, mpo_tensors):
        """observe returns a Python float."""
        val = observe(mps_tensors, mpo_tensors)
        assert isinstance(val, float)

    def test_raises_on_length_mismatch(self, mps_tensors, mpo_tensors):
        """observe raises ValueError when mps and mpo have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            observe(mps_tensors[:3], mpo_tensors)

    def test_linear_in_mpo(self, mps_tensors, mpo_tensors):
        """observe scales linearly with the MPO: observe(mps, c·H) = c·observe(mps, H).

        Scaling H → c·H is achieved by scaling a single MPO tensor (e.g. the
        first one) by c, since the MPO is a linear function of each local tensor.
        Scaling all L tensors would multiply the result by c^L, not c.
        """
        c = 3.7
        scaled = [mpo_tensors[0] * c] + mpo_tensors[1:]
        val        = observe(mps_tensors, mpo_tensors)
        val_scaled = observe(mps_tensors, scaled)
        assert math.isclose(val_scaled, c * val, rel_tol=1e-12)

    def test_zero_mpo_gives_zero(self, mps_tensors, mpo_tensors):
        """observe returns 0 for a zero MPO."""
        zero_mpo = [W * 0.0 for W in mpo_tensors]
        assert math.isclose(observe(mps_tensors, zero_mpo), 0.0, abs_tol=1e-14)
