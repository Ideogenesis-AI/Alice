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


"""Tests for alice.algorithm.dmrg.sweep (right_sweep and left_sweep)."""

from __future__ import annotations

from alice.algorithm.dmrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
)
from alice.algorithm.dmrg.scheme_1s import optimize_site
from alice.algorithm.dmrg.sweep import left_sweep, right_sweep


_DAVIDSON_OPTS = {'max_iter': 100, 'tol': 1e-10, 'max_subspace': 20}
_TRUNC = None  # no truncation for small test chains


class TestRightSweep:
    """Tests for right_sweep."""

    def test_right_sweep_moves_center_to_last_site(self, heisenberg_L2):
        """After right_sweep, mps.center == L-1."""
        mps, mpo = heisenberg_L2
        L = mps.L

        env_left = Environment(L)
        env_right = Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)

        right_sweep(mps, mpo, env_left, env_right, _TRUNC, _DAVIDSON_OPTS)
        assert mps.center == L - 1

    def test_right_sweep_energy_is_finite(self, heisenberg_L2):
        mps, mpo = heisenberg_L2
        L = mps.L
        env_left = Environment(L)
        env_right = Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)

        energy = right_sweep(mps, mpo, env_left, env_right, _TRUNC, _DAVIDSON_OPTS)
        assert energy == energy  # not NaN


class TestLeftSweep:
    """Tests for left_sweep."""

    def test_left_sweep_moves_center_to_site_zero(self, heisenberg_L2):
        """After left_sweep, mps.center == 0."""
        mps, mpo = heisenberg_L2
        L = mps.L

        env_left = Environment(L)
        env_right = Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)

        # First do a right_sweep so mps.center is at L-1.
        right_sweep(mps, mpo, env_left, env_right, _TRUNC, _DAVIDSON_OPTS)
        left_sweep(mps, mpo, env_left, env_right, _TRUNC, _DAVIDSON_OPTS)

        assert mps.center == 0

    def test_left_sweep_energy_is_finite(self, heisenberg_L2):
        mps, mpo = heisenberg_L2
        L = mps.L
        env_left = Environment(L)
        env_right = Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)

        right_sweep(mps, mpo, env_left, env_right, _TRUNC, _DAVIDSON_OPTS)
        energy = left_sweep(mps, mpo, env_left, env_right, _TRUNC, _DAVIDSON_OPTS)
        assert energy == energy  # not NaN


class TestFullSweep:
    """Tests for a complete right + left half-sweep pair."""

    def test_energy_does_not_increase_after_full_sweep(self, heisenberg_L2):
        """Energy after a full sweep is ≤ energy before (variational principle).

        ``observe()`` computes ⟨ψ|H|ψ⟩ without normalizing, so for a
        non-normalized initial MPS it does not yield the Rayleigh quotient.
        We therefore use the Davidson eigenvalue at site 0 as the ``before``
        reference, which is the normalized variational energy and comparable
        to the per-site Davidson eigenvalues returned by the sweep.
        """
        mps, mpo = heisenberg_L2
        L = mps.L

        env_left = Environment(L)
        env_right = Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)

        # Pre-sweep: normalized Rayleigh quotient at site 0 (Davidson eigenvalue).
        energy_before, _ = optimize_site(
            mps[0], env_left[0], mpo[0], env_right[0], _DAVIDSON_OPTS
        )

        right_sweep(mps, mpo, env_left, env_right, _TRUNC, _DAVIDSON_OPTS)
        energy_after = left_sweep(mps, mpo, env_left, env_right, _TRUNC, _DAVIDSON_OPTS)

        assert energy_after <= energy_before + 1e-9, (
            f"energy increased after sweep: {energy_before} -> {energy_after}"
        )

    def test_environments_stay_consistent_after_sweep(self, heisenberg_L2):
        """After a full sweep, env_left[0] and env_right[L-1] are still trivial (dim-1)."""
        mps, mpo = heisenberg_L2
        L = mps.L

        env_left = Environment(L)
        env_right = Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)

        right_sweep(mps, mpo, env_left, env_right, _TRUNC, _DAVIDSON_OPTS)
        left_sweep(mps, mpo, env_left, env_right, _TRUNC, _DAVIDSON_OPTS)

        # Left boundary is still trivial.
        for idx in env_left[0].indices:
            assert idx.dim == 1
        # Right boundary is still trivial.
        for idx in env_right[L - 1].indices:
            assert idx.dim == 1
