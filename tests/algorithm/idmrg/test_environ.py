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


"""Tests for the infinite unit-cell transfer operators and their fixed points."""

from __future__ import annotations

import pytest
import torch
from nicole import identity

from alice.algorithm.idmrg import (
    cell_left_env,
    cell_right_env,
    energy_density,
    left_environment,
    right_environment,
    transfer_left,
    transfer_right,
)

from .conftest import (
    HEISENBERG,
    HEISENBERG_SU2,
    MAJUMDAR_GHOSH,
    MAJUMDAR_GHOSH_SU2,
    dimer_imps,
    dimer_imps_su2,
    ihamiltonian,
    neel_imps,
    random_imps,
)

# The Majumdar-Ghosh dimer state is the exact ground state at J2 = J1 / 2:
# half the nearest-neighbor bonds sit inside a singlet and contribute -3/4,
# the rest contribute nothing, and every next-nearest-neighbor bond vanishes.
DIMER_ENERGY = -0.375

# A Neel state has no transverse correlations, so only the Ising part of the
# Heisenberg coupling survives: -1/4 on every bond.
NEEL_ENERGY = -0.25

# Alice's spin operators reach Nicole as single-precision table entries
# promoted to double, which caps any U(1) spin energy at roughly eight
# digits. The SU(2) tensors avoid those tables, which is why the same
# physical check is tightened by five orders of magnitude below.
TABLE_TOL = 1e-7


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
# Fixed points on states with a known energy
# ---------------------------------------------------------------------------

class TestKnownEnergies:
    """States whose energy density can be written down without computing it."""

    @pytest.mark.parametrize('N', [2, 4])
    def test_neel_state_against_heisenberg(self, spin_half, N):
        imps = neel_imps(spin_half, N)
        energy = energy_density(imps, ihamiltonian(HEISENBERG, N))
        assert energy == pytest.approx(NEEL_ENERGY, abs=TABLE_TOL)

    def test_dimer_state_against_majumdar_ghosh(self, spin_half):
        imps = dimer_imps(spin_half)
        impo = ihamiltonian(MAJUMDAR_GHOSH, 2, n3x=True)
        assert energy_density(imps, impo) == pytest.approx(
            DIMER_ENERGY, abs=TABLE_TOL
        )

    def test_dimer_state_against_majumdar_ghosh_su2(self, spin_half_su2):
        # Under SU(2) the same physics is exact to machine precision,
        # because the dimer tensors are pure Clebsch-Gordan coefficients.
        imps = dimer_imps_su2(spin_half_su2)
        impo = ihamiltonian(MAJUMDAR_GHOSH_SU2, 2, n3x=True)
        assert energy_density(imps, impo) == pytest.approx(
            DIMER_ENERGY, abs=1e-12
        )

    def test_dimer_state_against_plain_heisenberg_su2(self, spin_half_su2):
        # Dropping the next-nearest-neighbor term changes nothing, since the
        # dimer state has no correlation across two sites at all.
        imps = dimer_imps_su2(spin_half_su2)
        impo = ihamiltonian(HEISENBERG_SU2, 2)
        assert energy_density(imps, impo) == pytest.approx(
            DIMER_ENERGY, abs=1e-12
        )


# ---------------------------------------------------------------------------
# Consistency of the two sides
# ---------------------------------------------------------------------------

class TestEnvironmentConsistency:
    """Checks that hold for any state, exactly solvable or not."""

    @pytest.mark.parametrize('N', [2, 4])
    def test_both_sides_report_the_same_energy(self, spin_half, N):
        # The two environments describe disjoint halves of one chain, so
        # agreement is a genuine cross-check rather than a tautology: the
        # left side sweeps AL forwards and the right side sweeps AR back.
        imps = random_imps(spin_half, N)
        impo = ihamiltonian(HEISENBERG, N)
        _, left = left_environment(imps, impo)
        _, right = right_environment(imps, impo)
        assert left == pytest.approx(right, rel=1e-10)

    @pytest.mark.parametrize(
        'solver, absorb, basis',
        [
            (left_environment, cell_left_env, 'left_isometries'),
            (right_environment, cell_right_env, 'right_isometries'),
        ],
    )
    @pytest.mark.parametrize('N', [2, 4])
    def test_the_environment_grows_by_a_constant(
        self, spin_half, N, solver, absorb, basis,
    ):
        # The solved environment is a fixed point of everything except the
        # finish channel, which gains one cell of energy per cell absorbed.
        # So the increment must be the same every time: if any other channel
        # were still moving, the second increment would differ from the
        # first. This is the defining equation, checked without reaching
        # into the solver.
        imps = random_imps(spin_half, N)
        impo = ihamiltonian(HEISENBERG, N)
        E, _ = solver(imps, impo)
        isometries = getattr(imps, basis)()
        once = absorb(E, isometries, impo)
        twice = absorb(once, isometries, impo)
        assert _deviation(twice - once, once - E) < 1e-9
        assert _deviation(once, E) > 1e-3

    @pytest.mark.parametrize('reps', [2, 3])
    def test_energy_density_ignores_the_unit_cell_size(self, spin_half, reps):
        # Describing one state with a cell twice or three times as long must
        # not move the energy per site. This catches an energy that is
        # accumulated per cell but divided per site incorrectly, and any
        # dependence on where the environment happens to be anchored.
        base = random_imps(spin_half, 2)
        stretched = random_imps(spin_half, 2 * reps, period=2)
        assert energy_density(stretched, ihamiltonian(HEISENBERG, 2 * reps)) \
            == pytest.approx(
                energy_density(base, ihamiltonian(HEISENBERG, 2)), rel=1e-10
            )

    def test_complex_state_gives_a_real_energy(self, spin_half):
        imps = random_imps(spin_half, 2)
        imps.canonical()
        imps.to(dtype=torch.complex128)
        energy = energy_density(imps, ihamiltonian(HEISENBERG, 2))
        assert isinstance(energy, float)
        assert energy == pytest.approx(
            energy_density(random_imps(spin_half, 2), ihamiltonian(HEISENBERG, 2)),
            rel=1e-10,
        )


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------
