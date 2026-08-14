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


"""Tests for alice.algorithm.dmrg.environ."""

from __future__ import annotations

from concurrent.futures import Future

import pytest
from nicole import allclose as tensors_allclose

from alice.network import observe
from alice.algorithm.dmrg.environ import (
    Environment,
    build_left_envs,
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

    def test_uninitialized_getitem_raises(self, heisenberg_L2):
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
        assert 'initialized=1' in r


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
# step_right_env: consistency with observe()
# ---------------------------------------------------------------------------

class TestStepRightEnv:
    """Tests for step_right_env."""

    def test_composed_equals_observe(self, heisenberg_L2):
        """Composing step_right_env L times reproduces the scalar from observe()."""
        mps, mpo = heisenberg_L2
        L = mps.L

        # mps is already in right-canonical form from the fixture.
        obs_val = observe(mps, mpo)

        # Manually compose the right environment from site L-1 down to 0.
        E = right_env_boundary(mps, mpo)
        for i in range(L - 1, -1, -1):
            E = step_right_env(E, mps[i], mpo[i])

        # After sweeping all L sites, E is a 1×1×1 tensor. Extract the scalar.
        k, v = next(iter(E.data.items()))
        weight = 1.0 if E.intw is None else float(E.intw[k].weights[0, 0])
        env_val = float(v.item()) * weight

        assert abs(env_val - obs_val) < 1e-10, (
            f"step_right_env composed value {env_val} != observe() {obs_val}"
        )


# ---------------------------------------------------------------------------
# build_left_envs
# ---------------------------------------------------------------------------

class TestBuildLeftEnvs:
    """Tests for build_left_envs."""

    def test_build_left_envs_populates_all_slots(self, heisenberg_L4):
        """After build_left_envs every slot env_left[i] is set."""
        mps, mpo = heisenberg_L4
        L = mps.L
        mps.canonical(L - 1, trunc=None)
        env_left = Environment(L)
        build_left_envs(mps, mpo, env_left)
        for i in range(L):
            _ = env_left[i]  # should not raise

    def test_build_left_envs_requires_center_L_minus_1(self, heisenberg_L4):
        """build_left_envs raises ValueError when mps.center != L-1."""
        mps, mpo = heisenberg_L4
        # center is 0 from the fixture — wrong for build_left_envs.
        env_left = Environment(mps.L)
        with pytest.raises(ValueError, match="center"):
            build_left_envs(mps, mpo, env_left)

    def test_build_left_envs_skips_block_above_fetch_hi(self, heisenberg_L4):
        """With fetch_hi=L-2 (2-site mode), env_left[L-1] is never computed."""
        mps, mpo = heisenberg_L4
        L = mps.L
        mps.canonical(L - 1, trunc=None)
        env_left = Environment(L, fetch_hi=L - 2)
        build_left_envs(mps, mpo, env_left)
        for i in range(L - 1):
            _ = env_left[i]  # slots 0 … L-2 must be filled
        with pytest.raises(RuntimeError):
            _ = env_left[L - 1]  # slot L-1 was never computed


# ---------------------------------------------------------------------------
# build_right_envs
# ---------------------------------------------------------------------------

class TestBuildRightEnvs:
    """Tests for build_right_envs."""

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

        Mv = matvec(mps[0], mpo[0], E_left, env_right[0])
        energy_site0 = _inner_product(mps[0], Mv).real
        # For the full 1-site chain the energy from E_left+site0+E_right must equal observe().
        assert abs(energy_site0 - obs_val) < 1e-9, (
            f"energy from matvec {energy_site0} != observe() {obs_val}"
        )


# ---------------------------------------------------------------------------
# Disk caching — basic I/O
# ---------------------------------------------------------------------------

class TestEnvironmentDiskCache:
    """Tests for the disk-backed caching behavior of Environment.

    All tests use `async_io=False` so every write completes synchronously
    before the assertion — no timing complexity.
    """

    def test_setitem_writes_file(self, heisenberg_L2, tmp_path):
        """After env[0] = E the cache file 00000.pt must exist on disk."""
        mps, mpo = heisenberg_L2
        env = Environment(mps.L, tmp_path, async_io=False)
        E = left_env_boundary(mps, mpo)
        env[0] = E
        assert (tmp_path / "00000.pt").exists()

    def test_fetch_roundtrip_after_memory_clear(self, heisenberg_L2, tmp_path):
        """Tensor loaded from disk after memory eviction equals the original."""
        mps, mpo = heisenberg_L2
        env = Environment(mps.L, tmp_path, async_io=False)
        E = left_env_boundary(mps, mpo)
        env[0] = E
        # Bypass normal eviction — just wipe the in-memory slot.
        env._blocks[0] = None
        loaded = env.fetch(0)
        assert tensors_allclose(E, loaded)

    def test_fetch_in_memory_returns_same_object(self, heisenberg_L2, tmp_path):
        """fetch(i) returns the identical object when the block is in memory."""
        mps, mpo = heisenberg_L2
        env = Environment(mps.L, tmp_path, async_io=False)
        E = left_env_boundary(mps, mpo)
        env[0] = E
        assert env.fetch(0) is E

    def test_fetch_raises_without_path_and_no_memory(self, heisenberg_L2):
        """fetch(i) on an uninitialized slot raises RuntimeError when path=None."""
        mps, _ = heisenberg_L2
        env = Environment(mps.L)
        with pytest.raises(RuntimeError):
            env.fetch(0)

    def test_eviction_clears_memory_slot(self, heisenberg_L2, tmp_path):
        """Sequential fetch(0)/fetch(1) with window=1 evicts block 0 from memory."""
        mps, mpo = heisenberg_L2
        L = mps.L
        # window=1: after fetch(1), block at index 1-1*1=0 is evicted.
        env = Environment(L, tmp_path, async_io=False, window=1,
                          fetch_lo=0, fetch_hi=L - 1)
        E0 = left_env_boundary(mps, mpo)
        E1 = right_env_boundary(mps, mpo)
        env[0] = E0
        env[1] = E1
        env.fetch(0)
        env.fetch(1)
        assert env._blocks[0] is None

    def test_shutdown_completes_pending_writes(self, heisenberg_L2, tmp_path):
        """All cache files exist on disk after shutdown() even with async_io=True."""
        mps, mpo = heisenberg_L2
        env = Environment(mps.L, tmp_path, async_io=True)
        env[0] = left_env_boundary(mps, mpo)
        env[1] = right_env_boundary(mps, mpo)
        env.shutdown()
        assert (tmp_path / "00000.pt").exists()
        assert (tmp_path / "00001.pt").exists()

    def test_no_file_written_without_path(self, heisenberg_L2, tmp_path):
        """path=None: __setitem__ must not create any .pt files."""
        mps, mpo = heisenberg_L2
        env = Environment(mps.L)
        env[0] = left_env_boundary(mps, mpo)
        pt_files = list(tmp_path.rglob("*.pt"))
        assert pt_files == []

    def test_setitem_discards_stale_prefetch(self, heisenberg_L2, tmp_path):
        """Writing to slot i discards any pending prefetch future for that slot."""
        mps, mpo = heisenberg_L2
        env = Environment(mps.L, tmp_path, async_io=False)
        E = left_env_boundary(mps, mpo)
        # Inject a dummy completed future into _read_futures[0].
        dummy: Future = Future()
        dummy.set_result(E)
        env._read_futures[0] = dummy
        # Writing to slot 0 must clear the stale prefetch.
        env[0] = E
        assert 0 not in env._read_futures
        assert env._blocks[0] is E


# ---------------------------------------------------------------------------
# Sliding-window mechanics
# ---------------------------------------------------------------------------

class TestEnvironmentSlidingWindow:
    """Tests for the direction-tracking and window eviction logic.

    All tests use a disk path so that the window machinery is active.
    `async_io=False` keeps writes synchronous.
    """

    def _populate(self, env, mps, mpo):
        """Write boundary blocks for all L sites so fetch can load them."""
        E_left = left_env_boundary(mps, mpo)
        E_right = right_env_boundary(mps, mpo)
        for i in range(env._fetch_lo, env._fetch_hi + 1):
            # Alternate left/right boundaries — the exact tensor is not
            # important for these structural tests.
            env[i] = E_left if i % 2 == 0 else E_right

    def test_direction_flips_at_fetch_hi(self, heisenberg_L4, tmp_path):
        """_direction flips to -1 after fetching the highest valid index."""
        mps, mpo = heisenberg_L4
        L = mps.L
        env = Environment(L, tmp_path, async_io=False, window=2,
                          fetch_lo=0, fetch_hi=L - 1)
        self._populate(env, mps, mpo)
        # Fetch sequentially up to the boundary.
        for i in range(L):
            env.fetch(i)
        assert env._direction == -1

    def test_direction_flips_at_fetch_lo(self, heisenberg_L4, tmp_path):
        """_direction flips back to +1 after reaching the lowest valid index."""
        mps, mpo = heisenberg_L4
        L = mps.L
        env = Environment(L, tmp_path, async_io=False, window=2,
                          fetch_lo=1, fetch_hi=L - 1)
        self._populate(env, mps, mpo)
        # Forward pass to trigger the first flip at fetch_hi.
        for i in range(1, L):
            env.fetch(i)
        # direction is now -1; backward pass to trigger flip at fetch_lo.
        for i in range(L - 1, 0, -1):
            env.fetch(i)
        assert env._direction == 1

    def test_window_2_only_two_blocks_in_memory(self, heisenberg_L4, tmp_path):
        """After fetch(0), fetch(1), fetch(2) with window=2, block 0 is evicted."""
        mps, mpo = heisenberg_L4
        L = mps.L
        env = Environment(L, tmp_path, async_io=False, window=2,
                          fetch_lo=0, fetch_hi=L - 1)
        self._populate(env, mps, mpo)
        env.fetch(0)
        env.fetch(1)
        env.fetch(2)
        assert env._blocks[0] is None

    def test_window_1_no_prefetch(self, heisenberg_L4, tmp_path):
        """window=1 never queues any read futures."""
        mps, mpo = heisenberg_L4
        L = mps.L
        env = Environment(L, tmp_path, async_io=False, window=1,
                          fetch_lo=0, fetch_hi=L - 1)
        self._populate(env, mps, mpo)
        env.fetch(0)
        env.fetch(1)
        assert env._read_futures == {}

    def test_2site_fetch_lo_boundary(self, heisenberg_L4, tmp_path):
        """Block 0 is never loaded by fetch when fetch_lo=1 (2-site env_right)."""
        mps, mpo = heisenberg_L4
        L = mps.L
        env = Environment(L, tmp_path, async_io=False, window=2,
                          fetch_lo=1, fetch_hi=L - 1)
        self._populate(env, mps, mpo)
        # Forward pass over the valid range.
        for i in range(1, L):
            env.fetch(i)
        # Block 0 must never have been loaded.
        assert env._blocks[0] is None

    def test_left_edge_evicts_block_ahead_not_behind(self, heisenberg_L4, tmp_path):
        """At the 2-site left boundary, fetch(1) evicts block 3, not block 2.

        With fetch_lo=1, window=2, backward pass ending at fetch(1):
          evict(1 - (-1)*2 = 3)  →  block 3 cleared
          flip to +1
          prefetch(2)  →  block 2 already in memory, no-op

        Block 2 must still be in memory after fetch(1).
        """
        mps, mpo = heisenberg_L4
        L = mps.L  # 4
        env = Environment(L, tmp_path, async_io=False, window=2,
                          fetch_lo=1, fetch_hi=L - 1)
        self._populate(env, mps, mpo)
        # Forward pass to establish direction=-1 at the right boundary.
        for i in range(1, L):
            env.fetch(i)
        # direction is now -1. Backward pass.
        for i in range(L - 1, 0, -1):
            env.fetch(i)
        # After fetch(1): block 3 evicted, block 2 retained.
        assert env._blocks[3] is None
        assert env._blocks[2] is not None


# ---------------------------------------------------------------------------
# Disk caching with build_left_envs
# ---------------------------------------------------------------------------

class TestBuildLeftEnvsWithCache:
    """Tests that build_left_envs auto-caches blocks and respects the window."""

    def test_creates_all_cache_files(self, heisenberg_L4, tmp_path):
        """All L cache files exist after build_left_envs with a disk path."""
        mps, mpo = heisenberg_L4
        L = mps.L
        mps.canonical(L - 1, trunc=None)
        env_left = Environment(L, tmp_path, async_io=False)
        build_left_envs(mps, mpo, env_left)
        env_left.shutdown()
        for i in range(L):
            assert (tmp_path / f"{i:05d}.pt").exists(), (
                f"cache file for block {i} not found"
            )

    def test_fetch_after_memory_clear(self, heisenberg_L4, tmp_path):
        """fetch(i) reloads a block from disk after its memory slot is cleared."""
        mps, mpo = heisenberg_L4
        L = mps.L
        mps.canonical(L - 1, trunc=None)
        env_left = Environment(L, tmp_path, async_io=False)
        build_left_envs(mps, mpo, env_left)
        env_left._blocks[2] = None
        reloaded = env_left.fetch(2)
        assert len(reloaded.indices) == 3
        env_left.shutdown()

    def test_only_window_retained_in_memory(self, heisenberg_L4, tmp_path):
        """After build_left_envs only the initial window blocks stay in memory.

        With L=4, fetch_hi=2 (2-site mode), window=2 the keep range is [1, 2].
        Block 0 (left boundary seed) is evicted once block 1 is computed;
        block 3 is never computed because it lies above fetch_hi.
        """
        mps, mpo = heisenberg_L4
        L = mps.L  # 4
        mps.canonical(L - 1, trunc=None)
        env_left = Environment(L, tmp_path, async_io=False, window=2,
                               fetch_hi=L - 2)
        build_left_envs(mps, mpo, env_left)
        # Window [1, 2] must be in memory.
        assert env_left._blocks[1] is not None
        assert env_left._blocks[2] is not None
        # Block 0 (left boundary seed) was evicted after block 1 was built.
        assert env_left._blocks[0] is None
        # Block 3 was never computed (above fetch_hi).
        assert env_left._blocks[3] is None
        env_left.shutdown()


# ---------------------------------------------------------------------------
# Disk caching with build_right_envs
# ---------------------------------------------------------------------------

class TestBuildRightEnvsWithCache:
    """Tests that build_right_envs auto-caches blocks via the upgraded __setitem__."""

    def test_creates_all_cache_files(self, heisenberg_L4, tmp_path):
        """All L cache files exist after build_right_envs with a disk path."""
        mps, mpo = heisenberg_L4
        L = mps.L
        env_right = Environment(L, tmp_path, async_io=False)
        build_right_envs(mps, mpo, env_right)
        env_right.shutdown()
        for i in range(L):
            assert (tmp_path / f"{i:05d}.pt").exists(), (
                f"cache file for block {i} not found"
            )

    def test_fetch_after_memory_clear(self, heisenberg_L4, tmp_path):
        """fetch(i) reloads a block from disk after its memory slot is cleared."""
        mps, mpo = heisenberg_L4
        L = mps.L
        env_right = Environment(L, tmp_path, async_io=False)
        build_right_envs(mps, mpo, env_right)
        # Clear one block from memory without going through eviction.
        env_right._blocks[1] = None
        reloaded = env_right.fetch(1)
        # The reloaded block must be rank-3.
        assert len(reloaded.indices) == 3
        env_right.shutdown()

    def test_only_window_retained_in_memory(self, heisenberg_L4, tmp_path):
        """After build_right_envs only the initial window blocks stay in memory.

        With L=4, fetch_lo=1, window=2 the keep range is [1, 2].
        Block 3 (right boundary) is evicted once block 2 is computed;
        block 0 is never computed because it lies below fetch_lo.
        """
        mps, mpo = heisenberg_L4
        L = mps.L  # 4
        env_right = Environment(L, tmp_path, async_io=False, window=2,
                                fetch_lo=1)
        build_right_envs(mps, mpo, env_right)
        # Window [1, 2] must be in memory.
        assert env_right._blocks[1] is not None
        assert env_right._blocks[2] is not None
        # Block 0 was never computed (below fetch_lo).
        assert env_right._blocks[0] is None
        # Block 3 (right boundary seed) was evicted after block 2 was built.
        assert env_right._blocks[3] is None
        env_right.shutdown()
