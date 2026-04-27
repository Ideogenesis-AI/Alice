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


"""Tests for alice.algorithm.dmrg.environ."""

from __future__ import annotations

import pytest

from alice.network import observe
from alice.algorithm.dmrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
    right_env_boundary,
    step_left_env,
    step_right_env,
)


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class TestEnvironment:
    """Tests for the Environment container class."""

    def test_setitem_getitem_roundtrip(self, heisenberg_L2):
        """Stored tensor is returned unchanged by __getitem__."""
        mps, mpo = heisenberg_L2
        env = Environment(mps.L)
        E = left_env_boundary(mps, mpo)
        env[0] = E
        assert env[0] is E

    def test_out_of_range_getitem_raises(self, heisenberg_L2):
        mps, _ = heisenberg_L2
        env = Environment(mps.L)
        with pytest.raises(IndexError):
            _ = env[mps.L]

    def test_out_of_range_setitem_raises(self, heisenberg_L2):
        mps, mpo = heisenberg_L2
        env = Environment(mps.L)
        E = left_env_boundary(mps, mpo)
        with pytest.raises(IndexError):
            env[-1] = E

    def test_uninitialised_getitem_raises(self, heisenberg_L2):
        mps, _ = heisenberg_L2
        env = Environment(mps.L)
        with pytest.raises(RuntimeError):
            _ = env[0]

    def test_fetch_delegates_to_getitem(self, heisenberg_L2):
        """fetch(i) returns the same object as __getitem__(i)."""
        mps, mpo = heisenberg_L2
        env = Environment(mps.L)
        E = left_env_boundary(mps, mpo)
        env[0] = E
        assert env.fetch(0) is E

    def test_cache_raises_not_implemented(self, heisenberg_L2):
        mps, _ = heisenberg_L2
        env = Environment(mps.L)
        with pytest.raises(NotImplementedError):
            env.cache(0)

    def test_len(self, heisenberg_L2):
        mps, _ = heisenberg_L2
        env = Environment(mps.L)
        assert len(env) == mps.L

    def test_repr(self, heisenberg_L2):
        mps, mpo = heisenberg_L2
        env = Environment(mps.L)
        env[0] = left_env_boundary(mps, mpo)
        r = repr(env)
        assert 'Environment' in r
        assert 'initialised=1' in r


# ---------------------------------------------------------------------------
# Boundary constructors
# ---------------------------------------------------------------------------

class TestBoundaries:
    """Tests for left_env_boundary and right_env_boundary."""

    def test_left_boundary_rank(self, heisenberg_L2):
        """Left boundary is a rank-3 tensor."""
        mps, mpo = heisenberg_L2
        E = left_env_boundary(mps, mpo)
        assert len(E.indices) == 3

    def test_right_boundary_rank(self, heisenberg_L2):
        """Right boundary is a rank-3 tensor."""
        mps, mpo = heisenberg_L2
        E = right_env_boundary(mps, mpo)
        assert len(E.indices) == 3

    def test_left_boundary_dim1(self, heisenberg_L2):
        """Left boundary has total dimension 1 on all indices (trivial OBC boundary)."""
        mps, mpo = heisenberg_L2
        E = left_env_boundary(mps, mpo)
        for idx in E.indices:
            assert idx.dim == 1

    def test_right_boundary_dim1(self, heisenberg_L2):
        """Right boundary has total dimension 1 on all indices."""
        mps, mpo = heisenberg_L2
        E = right_env_boundary(mps, mpo)
        for idx in E.indices:
            assert idx.dim == 1


# ---------------------------------------------------------------------------
# step_left_env: consistency with observe()
# ---------------------------------------------------------------------------

class TestStepLeftEnv:
    """Tests for step_left_env."""

    def test_composed_equals_observe(self, heisenberg_L2):
        """Composing step_left_env L times reproduces the scalar from observe()."""
        mps, mpo = heisenberg_L2
        L = mps.L

        # Bring to a canonical form so that observe() gives a meaningful result.
        # (mps is already in right-canonical form from the fixture.)
        obs_val = observe(mps, mpo)

        # Manually compose the left environment from site 0 to L.
        E = left_env_boundary(mps, mpo)
        for i in range(L):
            E = step_left_env(E, mps[i], mpo[i])

        # After sweeping all L sites, E is a 1×1×1 tensor. Extract the scalar.
        k, v = next(iter(E.data.items()))
        weight = 1.0 if E.intw is None else float(E.intw[k].weights[0, 0])
        env_val = float(v.item()) * weight

        assert abs(env_val - obs_val) < 1e-10, (
            f"step_left_env composed value {env_val} != observe() {obs_val}"
        )


# ---------------------------------------------------------------------------
# step_right_env: consistency with build_right_envs
# ---------------------------------------------------------------------------

class TestStepRightEnv:
    """Tests for step_right_env and build_right_envs."""

    def test_build_right_envs_populates_all_slots(self, heisenberg_L4):
        """After build_right_envs every slot env_right[i] is set."""
        mps, mpo = heisenberg_L4
        env_right = Environment(mps.L)
        build_right_envs(mps, mpo, env_right)
        for i in range(mps.L):
            _ = env_right[i]  # should not raise

    def test_build_right_envs_requires_center_zero(self, heisenberg_L4):
        mps, mpo = heisenberg_L4
        # Move center away from 0.
        mps.canonical(1, trunc=None)
        env_right = Environment(mps.L)
        with pytest.raises(ValueError, match="center"):
            build_right_envs(mps, mpo, env_right)
        # Restore for subsequent tests.
        mps.canonical(0, trunc=None)

    def test_left_right_env_consistency(self, heisenberg_L2):
        """Left env composed from the left + right env at the same site equals observe()."""
        mps, mpo = heisenberg_L2
        obs_val = observe(mps, mpo)

        env_right = Environment(mps.L)
        build_right_envs(mps, mpo, env_right)

        # Left env at site 0 is just the boundary.
        E_left = left_env_boundary(mps, mpo)

        # The full expectation value equals the contraction of E_left, M[0],
        # W[0], and E_right[1]. Here we verify just the two-site chain directly
        # by checking that the environments contract to the same scalar.
        from alice.algorithm.dmrg.scheme_1s import matvec
        from alice.algorithm.dmrg.davidson import _inner_product

        Mv = matvec(mps[0], E_left, mpo[0], env_right[0])
        energy_site0 = _inner_product(mps[0], Mv).real
        # For the full 1-site chain the energy from E_left+site0+E_right must equal observe().
        assert abs(energy_site0 - obs_val) < 1e-9, (
            f"energy from matvec {energy_site0} != observe() {obs_val}"
        )
