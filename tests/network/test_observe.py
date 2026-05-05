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


"""Unit tests for alice.network.observe."""

from __future__ import annotations

import math

import pytest

from alice.network import MPS, MPO, observe

_L = 10   # must match the fixture chain length in conftest.py


class TestObserveDispatch:
    """Tests for the observe dispatcher logic."""

    def test_raises_for_mpo_state(self, mpo_tensors):
        """observe raises NotImplementedError when state is an MPO (thermal state)."""
        rho = MPO(mpo_tensors)
        with pytest.raises(NotImplementedError):
            observe(rho, mpo_tensors)

    def test_raises_for_invalid_state_type(self, mpo_tensors):
        """observe raises TypeError for an unrecognised state type."""
        with pytest.raises(TypeError, match="MPS or a sequence"):
            observe(42, mpo_tensors)

    def test_accepts_mps_object(self, mps_tensors, mpo_tensors):
        """observe accepts an MPS object as state and returns the same value as a list."""
        mps = MPS(mps_tensors, center=_L - 1)
        val_obj  = observe(mps,          mpo_tensors)
        val_list = observe(mps_tensors,  mpo_tensors)
        assert math.isclose(val_obj, val_list, rel_tol=1e-14)

    def test_accepts_mpo_object_as_observable(self, mps_tensors, mpo_tensors):
        """observe accepts an MPO object as the observable."""
        mpo = MPO(mpo_tensors)
        val_obj  = observe(mps_tensors, mpo)
        val_list = observe(mps_tensors, mpo_tensors)
        assert math.isclose(val_obj, val_list, rel_tol=1e-14)


class TestObserveMPS:
    """Tests for the MPS-MPO contraction path."""

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

    def test_additive_in_mpo(self, mps_tensors, mpo_tensors):
        """observe is additive: observe(mps, H1 + H2) = observe(mps, H1) + observe(mps, H2).

        Additivity is tested by splitting the MPO into c·H and (1-c)·H via the
        first tensor, so both partial MPOs share the same bond structure.
        """
        c = 0.4
        mpo_a = [mpo_tensors[0] * c]       + mpo_tensors[1:]
        mpo_b = [mpo_tensors[0] * (1 - c)] + mpo_tensors[1:]
        val   = observe(mps_tensors, mpo_tensors)
        val_a = observe(mps_tensors, mpo_a)
        val_b = observe(mps_tensors, mpo_b)
        assert math.isclose(val_a + val_b, val, rel_tol=1e-12)

    # ------------------------------------------------------------------
    # SU(2) symmetry — exercises the Bridge weight path at the boundary
    # ------------------------------------------------------------------

    def test_returns_float_su2(self, mps_tensors_su2, mpo_tensors_su2):
        """observe returns a Python float for SU(2) tensors."""
        val = observe(mps_tensors_su2, mpo_tensors_su2)
        assert isinstance(val, float)

    def test_linear_in_mpo_su2(self, mps_tensors_su2, mpo_tensors_su2):
        """observe scales linearly with the SU(2) MPO: observe(mps, c·H) = c·observe(mps, H)."""
        c = 3.7
        scaled = [mpo_tensors_su2[0] * c] + mpo_tensors_su2[1:]
        val        = observe(mps_tensors_su2, mpo_tensors_su2)
        val_scaled = observe(mps_tensors_su2, scaled)
        assert math.isclose(val_scaled, c * val, rel_tol=1e-12)

    def test_zero_mpo_gives_zero_su2(self, mps_tensors_su2, mpo_tensors_su2):
        """observe returns 0 for a zero SU(2) MPO."""
        zero_mpo = [W * 0.0 for W in mpo_tensors_su2]
        assert math.isclose(observe(mps_tensors_su2, zero_mpo), 0.0, abs_tol=1e-14)
