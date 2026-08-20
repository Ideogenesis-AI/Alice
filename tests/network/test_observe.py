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

from alice.network import MPS, MPO, observe, thermal_mpo

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
        with pytest.raises(TypeError, match="MPS"):
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


# ---------------------------------------------------------------------------
# Thermal (NormalMPO) observation path — Abelian and non-Abelian
# ---------------------------------------------------------------------------

class _ObserveThermalTests:
    """Shared test logic for `observe()` on a `NormalMPO` thermal state.

    Subclasses supply a `hamiltonian` fixture yielding `(H_mpo, Spc)` for a
    specific symmetry group (see `heisenberg_mpo_u1`/`heisenberg_mpo_su2`
    in `conftest.py`); every test method below is shared verbatim between
    the Abelian and non-Abelian cases, only the underlying symmetry group
    differs.
    """

    _BETA = 0.5
    _ORDER = 20   # Taylor order for thermal_mpo; ample for this beta/L=4 chain.

    def test_returns_finite_float(self, hamiltonian):
        """observe(rho, H) returns a finite Python float."""
        H_mpo, Spc = hamiltonian
        rho = thermal_mpo(H_mpo, self._BETA, self._ORDER, Spc)
        E = observe(rho, H_mpo)
        assert isinstance(E, float)
        assert math.isfinite(E)

    def test_raises_on_length_mismatch(self, hamiltonian):
        """observe raises ValueError when rho and the observable have different lengths."""
        H_mpo, Spc = hamiltonian
        rho = thermal_mpo(H_mpo, self._BETA, self._ORDER, Spc)
        truncated = [H_mpo[i] for i in range(H_mpo.L - 1)]
        with pytest.raises(ValueError, match="same length"):
            observe(rho, truncated)

    def test_linear_in_observable(self, hamiltonian):
        """observe(rho, c·H) = c·observe(rho, H).

        Scaling H → c·H is achieved by scaling a single MPO tensor (e.g.
        the first one) by c, since the MPO is a linear function of each
        local tensor. Scaling all L tensors would multiply the result by
        c^L, not c.
        """
        H_mpo, Spc = hamiltonian
        rho = thermal_mpo(H_mpo, self._BETA, self._ORDER, Spc)
        c = 2.5
        scaled = [H_mpo[0] * c] + [H_mpo[i] for i in range(1, H_mpo.L)]
        E = observe(rho, H_mpo)
        E_scaled = observe(rho, scaled)
        assert math.isclose(E_scaled, c * E, rel_tol=1e-9)

    def test_raises_for_zero_observable(self, hamiltonian):
        """observe raises ValueError for an identically-zero observable.

        Unlike the MPS path, the thermal path normalizes the observable
        via `NormalMPO.from_mpo()` before sweeping, which requires a
        nonzero Frobenius norm — an all-zero observable has zero norm by
        construction, so this raises rather than silently returning 0.
        """
        H_mpo, Spc = hamiltonian
        rho = thermal_mpo(H_mpo, self._BETA, self._ORDER, Spc)
        zero_mpo = [H_mpo[i] * 0.0 for i in range(H_mpo.L)]
        with pytest.raises(ValueError, match="zero-norm"):
            observe(rho, zero_mpo)


class TestObserveThermalU1(_ObserveThermalTests):
    """Thermal observe() tests for Abelian U(1) symmetry."""

    @pytest.fixture(scope='class')
    def hamiltonian(self, heisenberg_mpo_u1):
        return heisenberg_mpo_u1


class TestObserveThermalSU2(_ObserveThermalTests):
    """Thermal observe() tests for non-Abelian SU(2) symmetry.

    Also exercises the Bridge (`intw`) weight-extraction branch at the
    environment-sweep boundary in `_observe_thermal`, which only ever
    activates for a generic (non-Abelian) symmetry group.
    """

    @pytest.fixture(scope='class')
    def hamiltonian(self, heisenberg_mpo_su2):
        return heisenberg_mpo_su2

    def test_matches_u1_restriction(self, heisenberg_mpo_su2, heisenberg_mpo_u1):
        """SU(2) and U(1) realizations of the same Heisenberg chain agree.

        `H = J Σ S_i · S_{i+1}` is the same physical operator whether built
        with full SU(2) spin-rotation symmetry or only its U(1) (Sz-
        conserving) subgroup, so the thermal energy `⟨H⟩_β` must agree
        between the two — the strongest correctness signal available here,
        since there is no simple closed-form reference for a generic
        Heisenberg chain.
        """
        H_su2, Spc_su2 = heisenberg_mpo_su2
        H_u1, Spc_u1 = heisenberg_mpo_u1

        rho_su2 = thermal_mpo(H_su2, self._BETA, self._ORDER, Spc_su2)
        rho_u1 = thermal_mpo(H_u1, self._BETA, self._ORDER, Spc_u1)

        E_su2 = observe(rho_su2, H_su2)
        E_u1 = observe(rho_u1, H_u1)
        assert math.isclose(E_su2, E_u1, rel_tol=1e-6)
