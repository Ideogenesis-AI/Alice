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


"""Tests for alice.algorithm.xtrg.environ."""

from __future__ import annotations

import pytest
from nicole import allclose as tensors_allclose

from alice.network.thermal import NormalMPO, thermal_mpo
from alice.algorithm.xtrg.environ import (
    Environment,
    build_left_envs,
    build_right_envs,
    left_env_boundary,
    right_env_boundary,
    step_left_env,
    step_right_env,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rho(spinless_fermion_L4):
    """Build a simple NormalMPO from the spinless fermion model."""
    mpo, spc, _ = spinless_fermion_L4
    # Use tau_0=2^-12, order=4 for a simple (low-bond-dim) thermal MPO.
    return thermal_mpo(mpo, 2 ** -12, 4, spc)


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class TestEnvironment:
    """Tests for the Environment container class."""

    def test_setitem_getitem_roundtrip(self, spinless_fermion_L4):
        """Stored tensor is returned unchanged by __getitem__."""
        rho = _make_rho(spinless_fermion_L4)
        env = Environment(rho.L)
        E = left_env_boundary(rho, rho, rho)
        env[0] = E
        assert env[0] is E

    def test_out_of_range_getitem_raises(self, spinless_fermion_L4):
        rho = _make_rho(spinless_fermion_L4)
        env = Environment(rho.L)
        with pytest.raises(IndexError):
            _ = env[rho.L]

    def test_out_of_range_setitem_raises(self, spinless_fermion_L4):
        rho = _make_rho(spinless_fermion_L4)
        env = Environment(rho.L)
        E = left_env_boundary(rho, rho, rho)
        with pytest.raises(IndexError):
            env[-1] = E

    def test_uninitialized_getitem_raises(self, spinless_fermion_L4):
        rho = _make_rho(spinless_fermion_L4)
        env = Environment(rho.L)
        with pytest.raises(RuntimeError):
            _ = env[0]

    def test_fetch_delegates_to_getitem(self, spinless_fermion_L4):
        """fetch(i) returns the same object as __getitem__(i)."""
        rho = _make_rho(spinless_fermion_L4)
        env = Environment(rho.L)
        E = left_env_boundary(rho, rho, rho)
        env[0] = E
        assert env.fetch(0) is E

    def test_len(self, spinless_fermion_L4):
        rho = _make_rho(spinless_fermion_L4)
        env = Environment(rho.L)
        assert len(env) == rho.L

    def test_repr_shows_initialized_count(self, spinless_fermion_L4):
        rho = _make_rho(spinless_fermion_L4)
        env = Environment(rho.L)
        E = left_env_boundary(rho, rho, rho)
        env[0] = E
        r = repr(env)
        assert 'initialized=1' in r


# ---------------------------------------------------------------------------
# Boundary constructors
# ---------------------------------------------------------------------------

class TestFitBoundaries:
    """Tests for left_env_boundary and right_env_boundary."""

    def test_left_boundary_rank(self, spinless_fermion_L4):
        """Left boundary is a rank-3 tensor."""
        rho = _make_rho(spinless_fermion_L4)
        E = left_env_boundary(rho, rho, rho)
        assert len(E.indices) == 3

    def test_right_boundary_rank(self, spinless_fermion_L4):
        """Right boundary is a rank-3 tensor."""
        rho = _make_rho(spinless_fermion_L4)
        E = right_env_boundary(rho, rho, rho)
        assert len(E.indices) == 3

    def test_left_boundary_all_dim1(self, spinless_fermion_L4):
        """Left boundary has all-dim-1 indices (OBC trivial left bond)."""
        rho = _make_rho(spinless_fermion_L4)
        E = left_env_boundary(rho, rho, rho)
        assert all(idx.dim == 1 for idx in E.indices)

    def test_right_boundary_all_dim1(self, spinless_fermion_L4):
        """Right boundary has all-dim-1 indices (OBC trivial right bond)."""
        rho = _make_rho(spinless_fermion_L4)
        E = right_env_boundary(rho, rho, rho)
        assert all(idx.dim == 1 for idx in E.indices)


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

class TestStepFitLeftEnv:
    """Tests for step_left_env."""

    def test_output_rank(self, spinless_fermion_L4):
        """step_left_env produces a rank-3 tensor."""
        rho = _make_rho(spinless_fermion_L4)
        rho.canonical(0)  # right-canonical

        env_left = Environment(rho.L)
        build_right_envs(rho, rho, rho, env_right := Environment(rho.L))
        env_left[0] = left_env_boundary(rho, rho, rho)
        # Absorb site 0 into the left environment.
        E1 = step_left_env(env_left[0], rho[0], rho[0], rho[0])
        assert len(E1.indices) == 3

    def test_step_builds_consistent_chain(self, spinless_fermion_L4):
        """Sequentially stepping left → right reproduces build_left_envs."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(rho.L - 1)  # left-canonical

        # Build via bulk function.
        env_left_bulk = Environment(rho.L)
        build_left_envs(rho, rho, rho, env_left_bulk)

        # Build manually step by step.
        env_left_manual = Environment(rho.L)
        env_left_manual[0] = left_env_boundary(rho, rho, rho)
        for i in range(rho.L - 1):
            env_left_manual[i + 1] = step_left_env(
                env_left_manual[i], rho[i], rho[i], rho[i]
            )

        # Compare all populated blocks.
        for i in range(1, rho.L):
            assert tensors_allclose(env_left_bulk[i], env_left_manual[i], atol=1e-12), \
                f"env_left mismatch at site {i}"


class TestStepFitRightEnv:
    """Tests for step_right_env."""

    def test_output_rank(self, spinless_fermion_L4):
        """step_right_env produces a rank-3 tensor."""
        rho = _make_rho(spinless_fermion_L4)
        rho.canonical(0)

        L = rho.L
        env_right = Environment(L)
        env_right[L - 1] = right_env_boundary(rho, rho, rho)
        E = step_right_env(env_right[L - 1], rho[L - 1], rho[L - 1], rho[L - 1])
        assert len(E.indices) == 3

    def test_step_builds_consistent_chain(self, spinless_fermion_L4):
        """Sequentially stepping right → left reproduces build_right_envs."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)  # right-canonical

        L = rho.L

        # Build via bulk function.
        env_right_bulk = Environment(L)
        build_right_envs(rho, rho, rho, env_right_bulk)

        # Build manually step by step.
        env_right_manual = Environment(L)
        env_right_manual[L - 1] = right_env_boundary(rho, rho, rho)
        for i in range(L - 2, -1, -1):
            env_right_manual[i] = step_right_env(
                env_right_manual[i + 1], rho[i + 1], rho[i + 1], rho[i + 1]
            )

        # Compare all populated blocks.
        for i in range(L - 1):
            assert tensors_allclose(env_right_bulk[i], env_right_manual[i], atol=1e-12), \
                f"env_right mismatch at site {i}"


# ---------------------------------------------------------------------------
# Bulk build
# ---------------------------------------------------------------------------

class TestBuildFitEnvs:
    """Tests for build_left_envs and build_right_envs."""

    def test_build_left_requires_left_canonical(self, spinless_fermion_L4):
        """build_left_envs raises if mpo_c is not left-canonical."""
        rho = _make_rho(spinless_fermion_L4)
        rho.canonical(0)  # right-canonical, not left
        env_left = Environment(rho.L)
        with pytest.raises(ValueError, match='center'):
            build_left_envs(rho, rho, rho, env_left)

    def test_build_right_requires_right_canonical(self, spinless_fermion_L4):
        """build_right_envs raises if mpo_c is not right-canonical."""
        rho = _make_rho(spinless_fermion_L4)
        rho.canonical(rho.L - 1)  # left-canonical, not right
        env_right = Environment(rho.L)
        with pytest.raises(ValueError, match='center'):
            build_right_envs(rho, rho, rho, env_right)

    def test_build_left_populates_all_slots(self, spinless_fermion_L4):
        """build_left_envs fills env_left[0] through env_left[fetch_hi]."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(rho.L - 1)

        env_left = Environment(rho.L)
        build_left_envs(rho, rho, rho, env_left)
        for i in range(rho.L):
            _ = env_left[i]  # must not raise

    def test_build_right_populates_all_slots(self, spinless_fermion_L4):
        """build_right_envs fills env_right[fetch_lo] through env_right[L-1]."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)

        env_right = Environment(rho.L)
        build_right_envs(rho, rho, rho, env_right)
        for i in range(rho.L):
            _ = env_right[i]  # must not raise

    def test_left_and_right_env_consistent(self, spinless_fermion_L4):
        """E_left[L-1] and E_right[0] are built from the same tensors and agree."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        L = rho.L

        # Build left envs from left-canonical C.
        rho_l = NormalMPO([t.clone() for t in rho], scale=rho._scale, bc=rho.bc)
        rho_l.canonical(L - 1)
        env_left = Environment(L)
        build_left_envs(rho_l, rho_l, rho_l, env_left)

        # Build right envs from right-canonical C.
        rho_r = NormalMPO([t.clone() for t in rho], scale=rho._scale, bc=rho.bc)
        rho_r.canonical(0)
        env_right = Environment(L)
        build_right_envs(rho_r, rho_r, rho_r, env_right)

        # env_left[L-1] should be a dim-1 tensor; it accumulated all sites.
        E_l = env_left[L - 1]
        assert len(E_l.indices) == 3

        # env_right[0] should also be a dim-1 tensor.
        E_r = env_right[0]
        assert len(E_r.indices) == 3
