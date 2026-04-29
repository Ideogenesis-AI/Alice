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


"""Tests for alice.algorithm.dmrg.sweep (forward_sweep and backward_sweep)."""

from __future__ import annotations

from alice.algorithm.dmrg import Options
from alice.algorithm.dmrg.davidson import _inner_product
from alice.algorithm.dmrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
)
from alice.algorithm.dmrg.scheme_1s import optimize_1site
from alice.algorithm.dmrg.scheme_2s import build_bulk, matvec_2s
from alice.algorithm.dmrg.sweep import backward_sweep, forward_sweep


_OPTS_1S = Options(scheme='1s', trunc_thresh=0.0, davidson_tol=1e-10)
_OPTS_2S = Options(scheme='2s', trunc_thresh=0.0, davidson_tol=1e-10)


def _setup_envs(mps, mpo):
    """Build a fresh pair of left/right environments for the given MPS/MPO."""
    L = mps.L
    env_left = Environment(L)
    env_right = Environment(L)
    env_left[0] = left_env_boundary(mps, mpo)
    build_right_envs(mps, mpo, env_right)
    return env_left, env_right


class TestForwardSweep:
    """Tests for forward_sweep (both 1-site and 2-site schemes)."""

    def test_forward_sweep_1s_moves_center_to_last_site(self, heisenberg_L2):
        """After 1-site forward_sweep, mps.center == L-1."""
        mps, mpo = heisenberg_L2
        env_left, env_right = _setup_envs(mps, mpo)
        forward_sweep(mps, mpo, env_left, env_right, _OPTS_1S)
        assert mps.center == mps.L - 1

    def test_forward_sweep_1s_energy_is_finite(self, heisenberg_L2):
        mps, mpo = heisenberg_L2
        env_left, env_right = _setup_envs(mps, mpo)
        energy = forward_sweep(mps, mpo, env_left, env_right, _OPTS_1S)
        assert energy == energy  # not NaN

    def test_forward_sweep_2s_moves_center_to_last_site(self, heisenberg_L2):
        """After 2-site forward_sweep, mps.center == L-1."""
        mps, mpo = heisenberg_L2
        env_left, env_right = _setup_envs(mps, mpo)
        forward_sweep(mps, mpo, env_left, env_right, _OPTS_2S)
        assert mps.center == mps.L - 1

    def test_forward_sweep_2s_energy_is_finite(self, heisenberg_L2):
        mps, mpo = heisenberg_L2
        env_left, env_right = _setup_envs(mps, mpo)
        energy = forward_sweep(mps, mpo, env_left, env_right, _OPTS_2S)
        assert energy == energy  # not NaN


class TestBackwardSweep:
    """Tests for backward_sweep (both 1-site and 2-site schemes)."""

    def test_backward_sweep_1s_moves_center_to_site_zero(self, heisenberg_L2):
        """After 1-site backward_sweep, mps.center == 0."""
        mps, mpo = heisenberg_L2
        env_left, env_right = _setup_envs(mps, mpo)
        forward_sweep(mps, mpo, env_left, env_right, _OPTS_1S)
        backward_sweep(mps, mpo, env_left, env_right, _OPTS_1S)
        assert mps.center == 0

    def test_backward_sweep_1s_energy_is_finite(self, heisenberg_L2):
        mps, mpo = heisenberg_L2
        env_left, env_right = _setup_envs(mps, mpo)
        forward_sweep(mps, mpo, env_left, env_right, _OPTS_1S)
        energy = backward_sweep(mps, mpo, env_left, env_right, _OPTS_1S)
        assert energy == energy  # not NaN

    def test_backward_sweep_2s_moves_center_to_site_zero(self, heisenberg_L2):
        """After 2-site backward_sweep, mps.center == 0."""
        mps, mpo = heisenberg_L2
        env_left, env_right = _setup_envs(mps, mpo)
        forward_sweep(mps, mpo, env_left, env_right, _OPTS_2S)
        backward_sweep(mps, mpo, env_left, env_right, _OPTS_2S)
        assert mps.center == 0

    def test_backward_sweep_2s_energy_is_finite(self, heisenberg_L2):
        mps, mpo = heisenberg_L2
        env_left, env_right = _setup_envs(mps, mpo)
        forward_sweep(mps, mpo, env_left, env_right, _OPTS_2S)
        energy = backward_sweep(mps, mpo, env_left, env_right, _OPTS_2S)
        assert energy == energy  # not NaN


class TestFullSweep:
    """Tests for a complete right + left half-sweep pair (both schemes)."""

    def test_1s_energy_does_not_increase_after_full_sweep(self, heisenberg_L2):
        """Energy after a full 1-site sweep is ≤ energy before (variational principle).

        We use the Davidson eigenvalue at site 0 as the pre-sweep reference
        (normalised Rayleigh quotient), comparable to what the sweeps return.
        """
        mps, mpo = heisenberg_L2
        env_left, env_right = _setup_envs(mps, mpo)

        davidson_opts = {'max_iter': 100, 'tol': 1e-10, 'max_subspace': 20}
        energy_before, _, _ = optimize_1site(
            mps[0], env_left[0], mpo[0], env_right[0], davidson_opts
        )

        forward_sweep(mps, mpo, env_left, env_right, _OPTS_1S)
        energy_after = backward_sweep(mps, mpo, env_left, env_right, _OPTS_1S)

        assert energy_after <= energy_before + 1e-9, (
            f"1-site energy increased after sweep: {energy_before} -> {energy_after}"
        )

    def test_1s_environments_stay_consistent_after_sweep(self, heisenberg_L2):
        """After a full 1-site sweep, the boundary environment blocks are still trivial."""
        mps, mpo = heisenberg_L2
        env_left, env_right = _setup_envs(mps, mpo)

        forward_sweep(mps, mpo, env_left, env_right, _OPTS_1S)
        backward_sweep(mps, mpo, env_left, env_right, _OPTS_1S)

        for idx in env_left[0].indices:
            assert idx.dim == 1
        for idx in env_right[mps.L - 1].indices:
            assert idx.dim == 1

    def test_2s_energy_does_not_increase_after_full_sweep(self, heisenberg_L2):
        """Energy after a full 2-site sweep is ≤ initial Rayleigh quotient."""
        mps, mpo = heisenberg_L2
        env_left, env_right = _setup_envs(mps, mpo)

        # Pre-sweep: 2-site Rayleigh quotient at bond (0,1) as reference.
        Theta = build_bulk(mps[0], mps[1])
        H_Theta = matvec_2s(Theta, env_left[0], mpo[0], mpo[1], env_right[1])
        energy_before = (
            _inner_product(Theta, H_Theta).real / _inner_product(Theta, Theta).real
        )

        forward_sweep(mps, mpo, env_left, env_right, _OPTS_2S)
        energy_after = backward_sweep(mps, mpo, env_left, env_right, _OPTS_2S)

        assert energy_after <= energy_before + 1e-9, (
            f"2-site energy increased after sweep: {energy_before} -> {energy_after}"
        )

    def test_2s_environments_stay_consistent_after_sweep(self, heisenberg_L2):
        """After a full 2-site sweep, the boundary environment blocks are still trivial."""
        mps, mpo = heisenberg_L2
        env_left, env_right = _setup_envs(mps, mpo)

        forward_sweep(mps, mpo, env_left, env_right, _OPTS_2S)
        backward_sweep(mps, mpo, env_left, env_right, _OPTS_2S)

        for idx in env_left[0].indices:
            assert idx.dim == 1
        for idx in env_right[mps.L - 1].indices:
            assert idx.dim == 1
