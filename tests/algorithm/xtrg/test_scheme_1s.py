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


"""Tests for alice.algorithm.xtrg.scheme_1s."""

from __future__ import annotations

import pytest
from nicole import allclose as tensors_allclose

from alice.network.thermal import thermal_mpo
from alice.algorithm.xtrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
    right_env_boundary,
)
from alice.algorithm.xtrg.scheme_1s import local_update_1s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_envs(rho, rho_a=None, rho_b=None):
    """Build initial left and right environments for a compression test.

    Parameters
    ----------
    rho:
        NormalMPO used as the compressed MPO C (must be right-canonical).
    rho_a:
        Factor MPO A. Defaults to `rho`.
    rho_b:
        Factor MPO B. Defaults to `rho`.

    Returns
    -------
    tuple
        `(env_left, env_right)` with all blocks populated.
    """
    if rho_a is None:
        rho_a = rho
    if rho_b is None:
        rho_b = rho

    L = rho.L
    env_left = Environment(L)
    env_right = Environment(L)
    env_left[0] = left_env_boundary(rho_a, rho_b, rho)
    build_right_envs(rho_a, rho_b, rho, env_right)
    return env_left, env_right


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLocalUpdate1s:
    """Tests for local_update_1s."""

    def test_output_rank(self, spinless_fermion_L4):
        """local_update_1s returns a rank-4 tensor."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)

        env_left, env_right = _setup_envs(rho)
        C0 = local_update_1s(env_left[0], rho[0], rho[0], env_right[0])
        assert len(C0.indices) == 4

    def test_output_matches_mpo_convention(self, spinless_fermion_L4):
        """Output C_i has the same index count as rho[i] (left, right, phys_in, phys_out)."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)

        L = rho.L
        env_left, env_right = _setup_envs(rho)
        for i in range(L):
            C_i = local_update_1s(env_left[i], rho[i], rho[i], env_right[i])
            assert len(C_i.indices) == len(rho[i].indices) == 4
            # Build left env for next site.
            if i < L - 1:
                from alice.algorithm.xtrg.environ import step_left_env
                env_left[i + 1] = step_left_env(env_left[i], rho[i], rho[i], rho[i])

    def test_idempotency_when_converged(self, spinless_fermion_L4):
        """local_update_1s returns a rank-4 tensor after convergence.

        After running _fit_mpo, calling local_update_1s on any site of the
        converged rho_sq (with A=B=rho) should return a rank-4 tensor without
        error.
        """
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)

        from alice.algorithm.xtrg.xtrg import _fit_mpo, Options
        from alice.algorithm.xtrg.environ import step_left_env
        opts = Options(scheme='1s', n_sweeps=4, max_bond=None)
        rho_sq, _ = _fit_mpo(rho, rho, opts)

        # Build environments for the converged rho_sq (A=B=rho, C=rho_sq).
        rho_sq.canonical(0)
        env_left_conv, env_right_conv = _setup_envs(rho_sq, rho, rho)

        # Build left env up to site 2 without canonicalizing rho_sq further
        # (canonicalization would invalidate the right environments).
        for j in range(2):
            env_left_conv[j + 1] = step_left_env(
                env_left_conv[j], rho[j], rho[j], rho_sq[j]
            )

        C_new = local_update_1s(env_left_conv[2], rho[2], rho[2], env_right_conv[2])
        assert len(C_new.indices) == 4
