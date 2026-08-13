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


"""Tests for alice.algorithm.xtrg.scheme_2s."""

from __future__ import annotations

import math

from alice.network.thermal import thermal_mpo
from alice.algorithm.xtrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
)
from alice.algorithm.xtrg.scheme_2s import (
    discarded_weight,
    local_update_2s,
    split_backward,
    split_forward,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_envs(rho):
    """Build env_left (boundary only) and full env_right for a 2s test."""
    L = rho.L
    env_left = Environment(L)
    env_right = Environment(L)
    env_left[0] = left_env_boundary(rho, rho, rho)
    build_right_envs(rho, rho, rho, env_right)
    return env_left, env_right


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLocalUpdate2s:
    """Tests for local_update_2s."""

    def test_output_rank(self, spinless_fermion_L4):
        """local_update_2s returns a rank-6 tensor Θ."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)

        env_left, env_right = _setup_envs(rho)
        theta = local_update_2s(
            env_left[0], rho[0], rho[0], rho[1], rho[1], env_right[1]
        )
        assert len(theta.indices) == 6

    def test_output_for_all_bonds(self, spinless_fermion_L4):
        """local_update_2s produces rank-6 tensors for every bond."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)

        L = rho.L
        env_left, env_right = _setup_envs(rho)
        from alice.algorithm.xtrg.environ import step_left_env

        for i in range(L - 1):
            theta = local_update_2s(
                env_left[i], rho[i], rho[i], rho[i + 1], rho[i + 1], env_right[i + 1]
            )
            assert len(theta.indices) == 6
            if i < L - 2:
                itag = rho._bond_itag(i + 1)
                C_i, C_i1 = split_forward(theta, itag, None)
                env_left[i + 1] = step_left_env(env_left[i], rho[i], rho[i], C_i)


class TestSplitForward:
    """Tests for split_forward."""

    def test_output_ranks(self, spinless_fermion_L4):
        """split_forward returns two rank-4 tensors (MPO convention)."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)

        env_left, env_right = _setup_envs(rho)
        theta = local_update_2s(
            env_left[0], rho[0], rho[0], rho[1], rho[1], env_right[1]
        )
        itag = rho._bond_itag(1)
        C_i, C_i1 = split_forward(theta, itag, None)
        assert len(C_i.indices) == 4
        assert len(C_i1.indices) == 4

    def test_bond_limit_respected(self, spinless_fermion_L4):
        """split_forward with nkeep=1 produces bond dim ≤ 1 at the cut."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)

        env_left, env_right = _setup_envs(rho)
        theta = local_update_2s(
            env_left[0], rho[0], rho[0], rho[1], rho[1], env_right[1]
        )
        itag = rho._bond_itag(1)
        trunc = {'nkeep': 1}
        C_i, C_i1 = split_forward(theta, itag, trunc)
        # Bond dim at the cut (axis 1 of C_i, axis 0 of C_i1) should be ≤ 1.
        assert C_i.indices[1].dim <= 1 or C_i1.indices[0].dim <= 1


class TestSplitBackward:
    """Tests for split_backward."""

    def test_output_ranks(self, spinless_fermion_L4):
        """split_backward returns two rank-4 tensors."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)

        env_left, env_right = _setup_envs(rho)
        theta = local_update_2s(
            env_left[0], rho[0], rho[0], rho[1], rho[1], env_right[1]
        )
        itag = rho._bond_itag(1)
        C_i, C_i1 = split_backward(theta, itag, None)
        assert len(C_i.indices) == 4
        assert len(C_i1.indices) == 4


class TestDiscardedWeight:
    """Tests for discarded_weight."""

    def test_zero_when_trunc_is_none(self, spinless_fermion_L4):
        """discarded_weight returns 0.0 when trunc is None."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)

        env_left, env_right = _setup_envs(rho)
        theta = local_update_2s(
            env_left[0], rho[0], rho[0], rho[1], rho[1], env_right[1]
        )
        dw = discarded_weight(theta, None)
        assert dw == 0.0

    def test_nonnegative(self, spinless_fermion_L4):
        """discarded_weight is always non-negative."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)

        env_left, env_right = _setup_envs(rho)
        theta = local_update_2s(
            env_left[0], rho[0], rho[0], rho[1], rho[1], env_right[1]
        )
        dw = discarded_weight(theta, {'nkeep': 1})
        assert dw >= 0.0

    def test_zero_for_large_nkeep(self, spinless_fermion_L4):
        """discarded_weight is 0.0 when nkeep is larger than the bond dim."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)

        env_left, env_right = _setup_envs(rho)
        theta = local_update_2s(
            env_left[0], rho[0], rho[0], rho[1], rho[1], env_right[1]
        )
        # With nkeep=1000 nothing should be discarded.
        dw = discarded_weight(theta, {'nkeep': 1000, 'thresh': 0.0})
        assert math.isclose(dw, 0.0, abs_tol=1e-12)
