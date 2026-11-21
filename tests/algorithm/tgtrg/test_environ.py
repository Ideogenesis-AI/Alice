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


"""Tests for alice.algorithm.tgtrg.environ.

The physical check is a closed contraction: with `ρ` canonical at the site
being contracted, the environments plus the local tensor reproduce
`Tr[ρ̂ H ρ̂]`, computed independently through `NormalMPO` products and
`log_trace`. Both sweep directions are checked, as is their agreement, since
a density matrix normalized into `log_scale` represents the same operator
whichever end it is canonical at.

The remaining tests cover the storage class: the sliding window, the disk
cache, and the direction tracking that decides what to prefetch.
"""

from __future__ import annotations

import pytest
from nicole import Direction

from alice.algorithm.tgtrg.environ import (
    Environment,
    build_left_envs,
    build_right_envs,
    left_env_boundary,
    right_env_boundary,
    step_left_env,
    step_right_env,
)
from alice.algorithm.tgtrg.krylov import inner_product
from alice.algorithm.tgtrg.scheme_1s import matvec_1s

from .conftest import energy_reference, thermal_state


@pytest.fixture
def rho_H(spinless_fermion_L4):
    """A right-canonical `ρ(β=0.1)` on the L=4 chain, with its Hamiltonian."""
    mpo, spc, _ = spinless_fermion_L4
    return thermal_state(mpo, spc, 0.1), mpo


# ---------------------------------------------------------------------------
# Boundary tensors
# ---------------------------------------------------------------------------

class TestBoundaries:
    """Tests for the trivial left and right boundary blocks."""

    def test_left_boundary_structure(self, rho_H):
        """The left boundary is a 3rd-order tensor of dimension 1 everywhere."""
        rho, mpo = rho_H
        E = left_env_boundary(rho, mpo)
        assert len(E.indices) == 3
        assert [i.dim for i in E.indices] == [1, 1, 1]
        assert E.itags == (rho[0].itags[0], mpo[0].itags[0], rho[0].itags[0])

    def test_right_boundary_structure(self, rho_H):
        """The right boundary mirrors the left one at the far end of the chain."""
        rho, mpo = rho_H
        last = rho.L - 1
        E = right_env_boundary(rho, mpo)
        assert len(E.indices) == 3
        assert [i.dim for i in E.indices] == [1, 1, 1]
        assert E.itags == (rho[last].itags[1], mpo[last].itags[1], rho[last].itags[1])

    def test_mpo_boundary_directions_are_opposite(self, rho_H):
        """The MPO axis points out of the left block and into the right one."""
        rho, mpo = rho_H
        assert left_env_boundary(rho, mpo).indices[1].direction is Direction.OUT
        assert right_env_boundary(rho, mpo).indices[1].direction is Direction.IN


# ---------------------------------------------------------------------------
# The closed contraction
# ---------------------------------------------------------------------------

class TestExpectationValue:
    """Tests that the environments contract to the right number."""

    def test_right_sweep_reproduces_the_reference(self, rho_H):
        """With ρ canonical at site 0, the right environments close on Tr[ρ̂Hρ̂]."""
        rho, mpo = rho_H
        env_right = Environment(rho.L)
        build_right_envs(rho, mpo, env_right)

        value = inner_product(
            rho[0], matvec_1s(rho[0], mpo[0], left_env_boundary(rho, mpo),
                              env_right[0]))
        assert abs(value.imag) < 1e-12
        assert value.real == pytest.approx(energy_reference(rho, mpo), rel=1e-10)

    def test_left_sweep_reproduces_the_reference(self, rho_H):
        """With ρ canonical at site L-1, the left environments do the same."""
        rho, mpo = rho_H
        rho.canonical(rho.L - 1)
        env_left = Environment(rho.L)
        build_left_envs(rho, mpo, env_left)

        last = rho.L - 1
        value = inner_product(
            rho[last], matvec_1s(rho[last], mpo[last], env_left[last],
                                 right_env_boundary(rho, mpo)))
        assert value.real == pytest.approx(energy_reference(rho, mpo), rel=1e-10)

    def test_both_directions_agree(self, rho_H):
        """The two canonical forms give the same expectation value.

        Canonicalization moves norm into `log_scale`, so the internal tensor
        train represents the same normalized operator at either end, and the
        two sweeps must land on the same number.
        """
        rho, mpo = rho_H
        env_right = Environment(rho.L)
        build_right_envs(rho, mpo, env_right)
        from_right = inner_product(
            rho[0], matvec_1s(rho[0], mpo[0], left_env_boundary(rho, mpo),
                              env_right[0])).real

        rho.canonical(rho.L - 1)
        env_left = Environment(rho.L)
        build_left_envs(rho, mpo, env_left)
        last = rho.L - 1
        from_left = inner_product(
            rho[last], matvec_1s(rho[last], mpo[last], env_left[last],
                                 right_env_boundary(rho, mpo))).real

        assert from_left == pytest.approx(from_right, rel=1e-10)

    def test_holds_for_the_spinful_chain(self, spinful_fermion_L4):
        """A product of two symmetry groups changes nothing structurally."""
        mpo, spc, _ = spinful_fermion_L4
        # A low Taylor order keeps the doubled local space cheap; the series
        # need not be accurate here, only a legitimate density matrix.
        rho = thermal_state(mpo, spc, 0.1, order=3)
        env_right = Environment(rho.L)
        build_right_envs(rho, mpo, env_right)

        value = inner_product(
            rho[0], matvec_1s(rho[0], mpo[0], left_env_boundary(rho, mpo),
                              env_right[0]))
        assert value.real == pytest.approx(energy_reference(rho, mpo), rel=1e-10)

    def test_holds_at_a_colder_temperature(self, spinless_fermion_L4):
        """A larger β grows the bond dimension but not the index bookkeeping."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_state(mpo, spc, 1.0)
        env_right = Environment(rho.L)
        build_right_envs(rho, mpo, env_right)

        value = inner_product(
            rho[0], matvec_1s(rho[0], mpo[0], left_env_boundary(rho, mpo),
                              env_right[0]))
        assert value.real == pytest.approx(energy_reference(rho, mpo), rel=1e-10)


# ---------------------------------------------------------------------------
# Transfer steps
# ---------------------------------------------------------------------------

class TestTransferSteps:
    """Tests for the single-site transfer-matrix updates."""

    def test_right_step_matches_the_bulk_builder(self, rho_H):
        """One manual step reproduces what `build_right_envs` stores."""
        rho, mpo = rho_H
        env_right = Environment(rho.L)
        build_right_envs(rho, mpo, env_right)

        last = rho.L - 1
        manual = step_right_env(right_env_boundary(rho, mpo), rho[last], mpo[last])
        stored = env_right[last - 1]
        assert manual.itags == stored.itags
        assert (manual - stored).norm() < 1e-12

    def test_left_step_matches_the_bulk_builder(self, rho_H):
        """The same for the left environments."""
        rho, mpo = rho_H
        rho.canonical(rho.L - 1)
        env_left = Environment(rho.L)
        build_left_envs(rho, mpo, env_left)

        manual = step_left_env(left_env_boundary(rho, mpo), rho[0], mpo[0])
        stored = env_left[1]
        assert manual.itags == stored.itags
        assert (manual - stored).norm() < 1e-12

    def test_blocks_carry_the_bond_itags(self, rho_H):
        """Block `i` is tagged by the bond it sits on, on all three axes."""
        rho, mpo = rho_H
        env_right = Environment(rho.L)
        build_right_envs(rho, mpo, env_right)

        for i in range(rho.L - 1):
            E = env_right[i]
            bond = rho[i].itags[1]
            assert E.itags == (bond, mpo[i].itags[1], bond)

    def test_block_dimensions_follow_the_bonds(self, rho_H):
        """Block `i` has the ρ bond dimension on its outer axes."""
        rho, mpo = rho_H
        env_right = Environment(rho.L)
        build_right_envs(rho, mpo, env_right)

        for i in range(rho.L - 1):
            E = env_right[i]
            assert E.indices[0].dim == rho[i].indices[1].dim
            assert E.indices[2].dim == rho[i].indices[1].dim
            assert E.indices[1].dim == mpo[i].indices[1].dim


# ---------------------------------------------------------------------------
# Canonical-form requirements
# ---------------------------------------------------------------------------

class TestCanonicalRequirements:
    """The builders refuse a canonical form that would make the blocks wrong."""

    def test_right_builder_requires_center_zero(self, rho_H):
        """Right environments presume right-isometric tensors."""
        rho, mpo = rho_H
        rho.canonical(rho.L - 1)
        with pytest.raises(ValueError, match='center == 0'):
            build_right_envs(rho, mpo, Environment(rho.L))

    def test_left_builder_requires_center_at_the_end(self, rho_H):
        """Left environments presume left-isometric tensors."""
        rho, mpo = rho_H
        with pytest.raises(ValueError, match='center == L-1'):
            build_left_envs(rho, mpo, Environment(rho.L))


# ---------------------------------------------------------------------------
# Storage behavior
# ---------------------------------------------------------------------------

class TestStorage:
    """Tests for the in-memory interface of `Environment`."""

    def test_length_is_the_chain_length(self):
        """`len` reports the number of slots, not the number filled."""
        assert len(Environment(7)) == 7

    def test_uninitialized_block_raises(self):
        """Reading a slot that was never written is a programming error."""
        with pytest.raises(RuntimeError, match='not been initialized'):
            Environment(4)[2]

    def test_out_of_range_read_raises(self):
        """Indices outside the chain raise `IndexError`."""
        env = Environment(4)
        with pytest.raises(IndexError):
            env[4]
        with pytest.raises(IndexError):
            env[-1]

    def test_out_of_range_write_raises(self, rho_H):
        """Writing outside the chain raises too."""
        rho, mpo = rho_H
        env = Environment(4)
        with pytest.raises(IndexError):
            env[9] = left_env_boundary(rho, mpo)

    def test_round_trips_a_block(self, rho_H):
        """A stored block comes back unchanged."""
        rho, mpo = rho_H
        env = Environment(rho.L)
        E = left_env_boundary(rho, mpo)
        env[0] = E
        assert env[0] is E

    def test_repr_counts_initialized_blocks(self, rho_H):
        """`repr` is informative enough to debug a half-built sweep."""
        rho, mpo = rho_H
        env = Environment(rho.L)
        env[0] = left_env_boundary(rho, mpo)
        assert repr(env) == f"Environment(L={rho.L}, initialized=1)"

    def test_fetch_returns_the_stored_block(self, rho_H):
        """Without a cache path, `fetch` is a plain read."""
        rho, mpo = rho_H
        env = Environment(rho.L)
        env[0] = left_env_boundary(rho, mpo)
        assert env.fetch(0) is env[0]

    def test_fetch_of_a_missing_block_raises_without_a_path(self):
        """With nowhere to load from, a missing block is an error."""
        with pytest.raises(RuntimeError, match='not been initialized'):
            Environment(4).fetch(1)

    def test_shutdown_without_a_path_is_a_noop(self):
        """Shutting down a memory-only instance is safe and repeatable."""
        env = Environment(4)
        env.shutdown()
        env.shutdown()


class TestDiskCache:
    """Tests for the sliding window and its backing files."""

    def test_writes_a_file_per_block(self, tmp_path, rho_H):
        """Each stored block lands in its own cache file."""
        rho, mpo = rho_H
        env = Environment(rho.L, path=tmp_path)
        build_right_envs(rho, mpo, env)
        env.shutdown()
        assert sorted(p.name for p in tmp_path.glob('*.pt')) == [
            f"{i:05d}.pt" for i in range(rho.L)]

    def test_reloads_an_evicted_block(self, tmp_path, rho_H):
        """A block dropped from memory is restored bit-for-bit from disk."""
        rho, mpo = rho_H
        env = Environment(rho.L, path=tmp_path, async_io=False)
        build_right_envs(rho, mpo, env)
        expected = env.fetch(2)

        env._evict(2)
        assert env._blocks[2] is None

        reloaded = env.fetch(2)
        assert reloaded.itags == expected.itags
        assert (reloaded - expected).norm() < 1e-14

    def test_eviction_keeps_the_window_resident(self, tmp_path, rho_H):
        """The builder leaves resident exactly the blocks the next sweep needs."""
        rho, mpo = rho_H
        env = Environment(rho.L, path=tmp_path, async_io=False, window=2)
        build_right_envs(rho, mpo, env)
        resident = [i for i, b in enumerate(env._blocks) if b is not None]
        assert resident == [0, 1]

    def test_window_is_clamped_to_the_fetch_range(self, tmp_path):
        """A window wider than the chain is narrowed instead of overrunning it."""
        env = Environment(3, path=tmp_path, window=10, fetch_lo=1, fetch_hi=2)
        assert env._window == 2

    def test_direction_flips_at_the_far_end(self, tmp_path, rho_H):
        """Reaching the top of the fetch range turns the sweep around."""
        rho, mpo = rho_H
        env = Environment(rho.L, path=tmp_path, async_io=False, window=2)
        build_right_envs(rho, mpo, env)

        assert env._direction == 1
        for i in range(rho.L):
            env.fetch(i)
        assert env._direction == -1

    def test_direction_flips_back_at_the_near_end(self, tmp_path, rho_H):
        """And again at the bottom, as one cooling step flips twice."""
        rho, mpo = rho_H
        env = Environment(rho.L, path=tmp_path, async_io=False, window=2)
        build_right_envs(rho, mpo, env)
        for i in list(range(rho.L)) + list(range(rho.L - 2, -1, -1)):
            env.fetch(i)
        assert env._direction == 1

    def test_a_full_sweep_returns_every_block(self, tmp_path, rho_H):
        """Sweeping back and forth through the cache never loses a block."""
        rho, mpo = rho_H
        env = Environment(rho.L, path=tmp_path, async_io=False, window=2)
        build_right_envs(rho, mpo, env)
        expected = [env.fetch(i) for i in range(rho.L)]

        for i in range(rho.L - 2, -1, -1):
            env.fetch(i)
        for i in range(rho.L):
            assert (env.fetch(i) - expected[i]).norm() < 1e-14
        env.shutdown()

    def test_overwriting_supersedes_the_cached_copy(self, tmp_path, rho_H):
        """A later fetch sees the rewritten block, not the stale file."""
        rho, mpo = rho_H
        env = Environment(rho.L, path=tmp_path, async_io=False)
        build_right_envs(rho, mpo, env)

        replacement = env[1] * 2.0
        env[1] = replacement
        env._evict(1)
        assert (env.fetch(1) - replacement).norm() < 1e-14

    def test_async_writes_complete_on_shutdown(self, tmp_path, rho_H):
        """Background writes are flushed before the executor goes away."""
        rho, mpo = rho_H
        env = Environment(rho.L, path=tmp_path, async_io=True)
        build_right_envs(rho, mpo, env)
        env.shutdown()
        assert env._executor is None
        assert len(list(tmp_path.glob('*.pt'))) == rho.L

    def test_out_of_range_eviction_is_ignored(self, tmp_path):
        """Window arithmetic runs off the ends routinely; that must be harmless."""
        env = Environment(3, path=tmp_path)
        env._evict(-5)
        env._evict(99)
