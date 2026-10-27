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
    Environment,
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
    neel_imps,
    random_imps,
    single_site_imps,
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
    """The right side must cover every bond and agree with the solvers."""

    @pytest.mark.parametrize('N', [2, 4])
    def test_the_right_side_covers_every_bond(self, spin_half, N):
        # The left side comes back as the one block on bond 0, since a
        # pass builds the rest of it as it walks.
        imps = random_imps(spin_half, N)
        _, env_right, _ = cell_environments(imps, ihamiltonian(HEISENBERG, N))
        assert len(env_right) == N
        assert all(env_right[k] is not None for k in range(N))

    @pytest.mark.parametrize('N', [2, 4])
    def test_boundary_blocks_match_the_fixed_point_solvers(self, spin_half, N):
        # Both sides are the fixed points on bond 0 themselves; the other
        # right blocks are reached from there by absorbing sites.
        imps = random_imps(spin_half, N)
        impo = ihamiltonian(HEISENBERG, N)
        env_left, env_right, energy = cell_environments(imps, impo)

        boundary_left, cell_energy = left_environment(imps, impo)
        boundary_right, _ = right_environment(imps, impo)
        assert float((env_left - boundary_left).norm()) < 1e-12
        assert float((env_right[0] - boundary_right).norm()) < 1e-12
        assert energy == pytest.approx(cell_energy)

    def test_energy_is_per_unit_cell(self, spin_half):
        # The same physical state described by cells of two sizes must give
        # energies in the ratio of the cell lengths.
        two = random_imps(spin_half, 2, period=2)
        four = random_imps(spin_half, 4, period=2)
        _, _, e_two = cell_environments(two, ihamiltonian(HEISENBERG, 2))
        _, _, e_four = cell_environments(four, ihamiltonian(HEISENBERG, 4))
        assert e_four / 4 == pytest.approx(e_two / 2, abs=TABLE_TOL)

    def test_storage_for_the_left_side_names_the_missing_direction(
        self, spin_half,
    ):
        # Only the stored side needs somewhere to live, and going forward
        # that is the right one.
        imps = random_imps(spin_half, 2)
        impo = ihamiltonian(HEISENBERG, 2)
        with pytest.raises(NotImplementedError, match='right-to-left'):
            cell_environments(imps, impo, env_left=Environment(2))


# ---------------------------------------------------------------------------
# One VUMPS iteration
# ---------------------------------------------------------------------------

class TestCellUpdate:
    """An iteration must leave a usable state and move downhill."""

    @pytest.mark.parametrize('scheme', ['1s', '2s'])
    @pytest.mark.parametrize('N', [2, 4])
    def test_state_stays_canonical_and_normalized(self, spin_half, N, scheme):
        imps = random_imps(spin_half, N)
        impo = ihamiltonian(HEISENBERG, N)
        env_left, env_right, _ = cell_environments(imps, impo)
        cell_update(imps, impo, env_left, env_right, Options(scheme=scheme))

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
    def test_one_site_preserves_the_bond_dimensions(self, spin_half, N):
        # The one-site scheme neither splits nor truncates, so the cell
        # keeps exactly the width it came in with.
        imps = random_imps(spin_half, N)
        impo = ihamiltonian(HEISENBERG, N)
        before = list(imps.bond_dims)
        env_left, env_right, _ = cell_environments(imps, impo)
        cell_update(imps, impo, env_left, env_right, Options(scheme='1s'))
        assert imps.bond_dims == before

    @pytest.mark.parametrize('scheme', ['1s', '2s'])
    def test_energy_decreases(self, spin_half, scheme):
        imps = random_imps(spin_half, 2)
        impo = ihamiltonian(HEISENBERG, 2)
        env_left, env_right, before = cell_environments(imps, impo)
        cell_update(imps, impo, env_left, env_right, Options(scheme=scheme))
        _, _, after = cell_environments(imps, impo)
        assert after < before

    @pytest.mark.parametrize('scheme', ['1s', '2s'])
    def test_exact_ground_state_is_stationary(self, spin_half, scheme):
        # The dimer state solves the Majumdar-Ghosh point exactly, so an
        # iteration must report no gradient and leave the energy alone.
        imps = dimer_imps(spin_half)
        impo = ihamiltonian(MAJUMDAR_GHOSH, 2, n3x=True)
        env_left, env_right, before = cell_environments(imps, impo)
        gradient, _ = cell_update(
            imps, impo, env_left, env_right, Options(scheme=scheme),
        )
        _, _, after = cell_environments(imps, impo)
        assert gradient < 1e-7
        assert before / 2 == pytest.approx(DIMER_ENERGY, abs=TABLE_TOL)
        assert after / 2 == pytest.approx(DIMER_ENERGY, abs=TABLE_TOL)

    def test_only_the_two_site_scheme_reports_a_discarded_weight(self, spin_half):
        imps = random_imps(spin_half, 2)
        impo = ihamiltonian(HEISENBERG, 2)
        env_left, env_right, _ = cell_environments(imps, impo)
        _, discarded = cell_update(
            imps, impo, env_left, env_right, Options(scheme='1s'),
        )
        assert discarded == 0.0

    def test_unimplemented_scheme_is_rejected(self, spin_half):
        imps = random_imps(spin_half, 2)
        impo = ihamiltonian(HEISENBERG, 2)
        env_left, env_right, _ = cell_environments(imps, impo)
        opts = Options()
        opts.scheme = '1sp'
        with pytest.raises(NotImplementedError, match='unknown scheme'):
            cell_update(imps, impo, env_left, env_right, opts)

    def test_mirrored_environments_name_the_missing_direction(self, spin_half):
        # Streaming the right side and storing the left describes a
        # backward pass, which the signature admits and nothing runs yet.
        imps = random_imps(spin_half, 2)
        impo = ihamiltonian(HEISENBERG, 2)
        env_left, env_right, _ = cell_environments(imps, impo)
        with pytest.raises(NotImplementedError, match='right-to-left'):
            cell_update(imps, impo, env_right, env_left, Options())

    @pytest.mark.parametrize('scheme', ['1s', '2s'])
    def test_two_sides_in_one_shape_are_rejected(self, spin_half, scheme):
        imps = random_imps(spin_half, 2)
        impo = ihamiltonian(HEISENBERG, 2)
        env_left, env_right, _ = cell_environments(imps, impo)
        with pytest.raises(TypeError, match='same shape'):
            cell_update(
                imps, impo, env_left, env_right[1], Options(scheme=scheme),
            )


class TestTwoSiteCellUpdate:
    """What only the two-site scheme can do, and what it refuses."""

    def test_bond_dimension_grows_from_a_product_state(self, spin_half):
        # Every bond of the Neel state is one-dimensional, which is the
        # case the one-site scheme can never leave.
        imps = neel_imps(spin_half, 2)
        impo = ihamiltonian(HEISENBERG, 2)
        assert imps.bond_dims == [1, 1]
        env_left, env_right, _ = cell_environments(imps, impo)
        cell_update(imps, impo, env_left, env_right, Options(scheme='2s'))
        assert max(imps.bond_dims) > 1

    def test_growth_respects_the_bond_cap(self, spin_half):
        imps = random_imps(spin_half, 2, width=1, dim=1)
        impo = ihamiltonian(HEISENBERG, 2)
        opts = Options(scheme='2s', max_bond=3)
        for _ in range(3):
            env_left, env_right, _ = cell_environments(imps, impo)
            cell_update(imps, impo, env_left, env_right, opts)
            assert max(imps.bond_dims) <= 3

    def test_truncation_is_reported(self, spin_half):
        # A cap tight enough to bite must show up as a discarded weight.
        imps = random_imps(spin_half, 2)
        impo = ihamiltonian(HEISENBERG, 2)
        env_left, env_right, _ = cell_environments(imps, impo)
        _, discarded = cell_update(
            imps, impo, env_left, env_right, Options(scheme='2s', max_bond=2),
        )
        assert discarded > 0.0

    def test_single_site_cell_is_rejected(self, spin_half):
        # Refused before anything is read, so the environments only have to
        # be of the right shape; any tensor stands in for the left block.
        imps = single_site_imps(spin_half)
        impo = ihamiltonian(HEISENBERG, 1)
        with pytest.raises(ValueError, match='at least two sites'):
            cell_update(
                imps, impo, impo[0], Environment(1), Options(scheme='2s'),
            )


class _Watched(Environment):
    """An `Environment` that remembers how full it ever got."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak = 0

    def __setitem__(self, k, E):
        super().__setitem__(k, E)
        self.peak = max(self.peak, self.n_resident)

    def fetch(self, k):
        block = super().fetch(k)
        self.peak = max(self.peak, self.n_resident)
        return block


class TestCachedCellUpdate:
    """A pass must read the right side off disk without noticing."""

    @pytest.mark.parametrize('scheme', ['1s', '2s'])
    def test_a_cached_pass_lands_where_an_uncached_one_does(
        self, spin_half, tmp_path, scheme,
    ):
        impo = ihamiltonian(HEISENBERG, 4)
        opts = Options(scheme=scheme)

        plain_imps = random_imps(spin_half, 4)
        plain = cell_update(
            plain_imps, impo, *cell_environments(plain_imps, impo)[:2], opts,
        )

        cached_imps = random_imps(spin_half, 4)
        env_left, env_right, _ = cell_environments(
            cached_imps, impo, env_right=Environment(4, tmp_path, window=2),
        )
        cached = cell_update(cached_imps, impo, env_left, env_right, opts)

        assert cached == pytest.approx(plain, abs=1e-12)
        _, _, plain_energy = cell_environments(plain_imps, impo)
        _, _, cached_energy = cell_environments(cached_imps, impo)
        assert cached_energy == pytest.approx(plain_energy, abs=1e-12)

    @pytest.mark.parametrize('scheme', ['1s', '2s'])
