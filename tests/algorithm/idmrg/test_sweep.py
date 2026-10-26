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


"""Tests for the unit-cell environments and the VUMPS iteration."""

from __future__ import annotations

import pytest
from nicole import identity

from alice.algorithm.idmrg import (
    Options,
    cell_environments,
    cell_update,
    left_environment,
    right_environment,
    transfer_left,
    transfer_right,
)

from .conftest import (
    HEISENBERG,
    MAJUMDAR_GHOSH,
    dimer_imps,
    ihamiltonian,
    random_imps,
)

DIMER_ENERGY = -0.375

# Alice's spin operators reach Nicole as single-precision table entries
# promoted to double, which caps any U(1) spin energy at roughly eight
# digits.
TABLE_TOL = 1e-7


def _delta(idx, itag):
    """Build an identity on a bond index, tagged for both ends of a cell."""
    delta = identity(idx)
    delta.retag([0, 1], [itag, itag])
    return delta


# ---------------------------------------------------------------------------
# Environments over the cell
# ---------------------------------------------------------------------------

class TestCellEnvironments:
    """Blocks must cover every bond and agree with the boundary solvers."""

    @pytest.mark.parametrize('N', [2, 4])
    def test_one_block_per_site_on_each_side(self, spin_half, N):
        imps = random_imps(spin_half, N)
        env_left, env_right, _ = cell_environments(imps, ihamiltonian(HEISENBERG, N))
        assert len(env_left) == N
        assert len(env_right) == N
        assert all(block is not None for block in env_left + env_right)

    @pytest.mark.parametrize('N', [2, 4])
    def test_boundary_blocks_match_the_fixed_point_solvers(self, spin_half, N):
        # Block 0 on the left and block N-1 on the right are the two fixed
        # points themselves; the rest are reached by absorbing sites.
        imps = random_imps(spin_half, N)
        impo = ihamiltonian(HEISENBERG, N)
        env_left, env_right, energy = cell_environments(imps, impo)

        boundary_left, cell_energy = left_environment(imps, impo)
        boundary_right, _ = right_environment(imps, impo)
        assert float((env_left[0] - boundary_left).norm()) < 1e-12
        assert float((env_right[N - 1] - boundary_right).norm()) < 1e-12
        assert energy == pytest.approx(cell_energy)

    def test_energy_is_per_unit_cell(self, spin_half):
        # The same physical state described by cells of two sizes must give
        # energies in the ratio of the cell lengths.
        two = random_imps(spin_half, 2, period=2)
        four = random_imps(spin_half, 4, period=2)
        _, _, e_two = cell_environments(two, ihamiltonian(HEISENBERG, 2))
        _, _, e_four = cell_environments(four, ihamiltonian(HEISENBERG, 4))
        assert e_four / 4 == pytest.approx(e_two / 2, abs=TABLE_TOL)


# ---------------------------------------------------------------------------
# One VUMPS iteration
# ---------------------------------------------------------------------------

class TestCellUpdate:
    """An iteration must leave a usable state and move downhill."""

    @pytest.mark.parametrize('N', [2, 4])
    def test_state_stays_canonical_and_normalized(self, spin_half, N):
        imps = random_imps(spin_half, N)
        impo = ihamiltonian(HEISENBERG, N)
        env_left, env_right, _ = cell_environments(imps, impo)
        cell_update(imps, impo, env_left, env_right, Options())

        # set_canonical installs the gauge directly, so these accessors must
        # work without a further canonical() call.
        right_isos = imps.right_isometries()
        last = right_isos[N - 1]
        delta = _delta(last.indices[1], last.itags[1])
        assert float((transfer_right(delta, right_isos) - delta).norm()) < 1e-10

        left_isos = imps.left_isometries()
        delta = _delta(left_isos[0].indices[0], left_isos[0].itags[0])
        assert float((transfer_left(delta, left_isos) - delta).norm()) < 1e-10

        assert abs(float(imps.bond_matrix(0).norm()) - 1.0) < 1e-10

    @pytest.mark.parametrize('N', [2, 4])
    def test_bond_dimensions_are_preserved(self, spin_half, N):
        # VUMPS neither splits nor truncates, so the cell keeps its width.
        imps = random_imps(spin_half, N)
        impo = ihamiltonian(HEISENBERG, N)
        before = list(imps.bond_dims)
        env_left, env_right, _ = cell_environments(imps, impo)
        cell_update(imps, impo, env_left, env_right, Options())
        assert imps.bond_dims == before

    def test_energy_decreases(self, spin_half):
        imps = random_imps(spin_half, 2)
        impo = ihamiltonian(HEISENBERG, 2)
        env_left, env_right, before = cell_environments(imps, impo)
        cell_update(imps, impo, env_left, env_right, Options())
        _, _, after = cell_environments(imps, impo)
        assert after < before

    def test_exact_ground_state_is_stationary(self, spin_half):
        # The dimer state solves the Majumdar-Ghosh point exactly, so an
        # iteration must report no gradient and leave the energy alone.
        imps = dimer_imps(spin_half)
        impo = ihamiltonian(MAJUMDAR_GHOSH, 2, n3x=True)
        env_left, env_right, before = cell_environments(imps, impo)
        gradient = cell_update(imps, impo, env_left, env_right, Options())
        _, _, after = cell_environments(imps, impo)
        assert gradient < 1e-7
        assert before / 2 == pytest.approx(DIMER_ENERGY, abs=TABLE_TOL)
        assert after / 2 == pytest.approx(DIMER_ENERGY, abs=TABLE_TOL)

    def test_unimplemented_scheme_is_rejected(self, spin_half):
        imps = random_imps(spin_half, 2)
        impo = ihamiltonian(HEISENBERG, 2)
        env_left, env_right, _ = cell_environments(imps, impo)
        opts = Options()
        opts.scheme = '2s'
        with pytest.raises(NotImplementedError, match='unknown scheme'):
            cell_update(imps, impo, env_left, env_right, opts)
