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


"""DMRG environment block storage and transfer-matrix update functions.

An environment block `E` is a rank-3 tensor with axes
`(bra_bond, mpo_bond, ket_bond)`. Two `Environment` instances are used per
DMRG run — one for the left environments and one for the right environments.

Index convention used throughout this module:

- `a, b, c, d` — virtual bond indices of the MPS (bra or ket side)
- `o, p, q`    — virtual bond indices of the MPO
- `r, s, t`    — physical indices

Left environment at site `i` (`env_left[i]`) has axes
`(a=bra_left, o=mpo_left, b=ket_left)` and accumulates sites `0…i-1`.
`env_left[0]` is the trivial left boundary.

Right environment at site `i` (`env_right[i]`) has axes
`(c=bra_right, p=mpo_right, d=ket_right)` and accumulates sites `i+1…L-1`.
`env_right[L-1]` is the trivial right boundary.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import torch
from nicole import Direction, Tensor, einsum, identity
from nicole import deserialize as _deserialize_tensor
from nicole import serialize as _serialize_tensor

from alice.network import MPS, MPO

logger = logging.getLogger(__name__)


class Environment:
    """Environment block storage for one sweep direction with optional disk caching.

    A single class instantiated twice per DMRG run — once as `env_left` and
    once as `env_right`. The `__getitem__` / `__setitem__` interface mirrors
    `MPS` and `MPO` for consistency.

    When `path` is provided, the class maintains a sliding window of
    `window` blocks in memory and caches the rest to disk. Disk writes are
    submitted asynchronously via a `ThreadPoolExecutor` so they overlap with
    the Davidson optimization step. Prefetch reads are also submitted
    asynchronously; `fetch` waits on them if needed.

    The sweep direction is tracked as an explicit `_direction` state variable
    (`+1` forward, `-1` backward) and is flipped automatically when the next
    predicted index exits `[fetch_lo, fetch_hi]`.

    Parameters
    ----------
    L:
        Number of sites in the chain.
    path:
        Directory for cache files. Each block is saved as `{i:05d}.pt`.
        If `None`, all blocks are kept in memory (no disk I/O).
    async_io:
        If `True` (default), disk writes are submitted to a background
        thread so they overlap with computation. Reads (prefetch) are also
        asynchronous. Has no effect when `path` is `None`.
    window:
        Number of blocks to keep in memory at once (current block plus
        up to `window-1` prefetched ahead). Clamped to the fetch range
        width. Has no effect when `path` is `None`.
    fetch_lo:
        Lowest site index that will ever be passed to `fetch`. Used to
        detect sweep-direction reversals at the left boundary. Defaults
        to `0`.
    fetch_hi:
        Highest site index that will ever be passed to `fetch`. Defaults
        to `L-1`. For 2-site DMRG, `env_left` should pass `L-2` and
        `env_right` should pass `L-1` (but `fetch_lo=1` for `env_right`).
    """

    def __init__(
        self,
        L: int,
        path: Optional[Path] = None,
        *,
        async_io: bool = True,
        window: int = 2,
        fetch_lo: int = 0,
        fetch_hi: Optional[int] = None,
    ) -> None:
        self._blocks: List[Optional[Tensor]] = [None] * L
        self._path: Optional[Path] = path
        self._async_io: bool = async_io and path is not None

        self._fetch_lo: int = fetch_lo
        self._fetch_hi: int = fetch_hi if fetch_hi is not None else L - 1
        # Clamp window so prefetch never systematically targets OOB indices
        # on short chains where the fetch range is narrower than `window`.
        fetch_range = max(1, self._fetch_hi - self._fetch_lo + 1)
        self._window: int = min(window, fetch_range)

        # Sweep direction: +1 = forward (increasing index), -1 = backward.
        # Initialized to +1 because every DMRG run begins with a forward sweep.
        self._direction: int = 1

        # Lazy-created executor shared for both reads and writes.
        self._executor: Optional[ThreadPoolExecutor] = None
        self._write_futures: Dict[int, Future] = {}
        self._read_futures: Dict[int, Future] = {}

    # ------------------------------------------------------------------
    # Index-style access (mirrors MPS / MPO convention)
    # ------------------------------------------------------------------

    def __getitem__(self, i: int) -> Tensor:
        """Return the environment block at site `i`.

        Raises
        ------
        IndexError
            If `i` is out of range.
        RuntimeError
            If the block has not been initialized yet.
        """
        if not (0 <= i < len(self._blocks)):
            raise IndexError(f"site index {i} out of range for L={len(self._blocks)}")
        block = self._blocks[i]
        if block is None:
            raise RuntimeError(f"environment block at site {i} has not been initialized")
        return block

    def __setitem__(self, i: int, E: Tensor) -> None:
        """Store the environment block at site `i` and schedule a disk write.

        If `path` was provided at construction, the block is asynchronously
        written to `{path}/{i:05d}.pt`. Any stale prefetch future for this
        index is discarded (the freshly written block supersedes it). If a
        previous write for the same index is still in flight, this call waits
        for it to complete before submitting the new write, preventing
        concurrent access to the same file.

        Raises
        ------
        IndexError
            If `i` is out of range.
        """
        if not (0 <= i < len(self._blocks)):
            raise IndexError(f"site index {i} out of range for L={len(self._blocks)}")
        self._blocks[i] = E

        # Discard any stale prefetch — the block is now overwritten in memory.
        self._read_futures.pop(i, None)

        if self._path is not None:
            # Wait for any previous write to the same slot to avoid a race.
            prev = self._write_futures.pop(i, None)
            if prev is not None:
                prev.result()
            self._write_futures[i] = self._submit(_save, self._path / f"{i:05d}.pt", E)

    # ------------------------------------------------------------------
    # Fetch — sliding-window read with direction tracking
    # ------------------------------------------------------------------

    def fetch(self, i: int) -> Tensor:
        """Return block `i`, loading from disk if evicted, then update the window.

        If the block is in memory, it is returned immediately. Otherwise the
        method waits on a pending prefetch future or falls back to a
        synchronous disk load.

        After resolving the block the method (in this exact order):

        1. Evicts the block that has fallen off the back of the window
           (`i - _direction * window`), using the direction *before* any flip.
        2. Flips `_direction` if the next index (`i + _direction`) exits
           `[fetch_lo, fetch_hi]`.
        3. Schedules prefetch of the next `window-1` blocks in the (possibly
           new) direction.

        The ordering of steps 1 and 2 ensures that the evicted block and the
        prefetched blocks are always disjoint — a block is never evicted and
        immediately re-prefetched in the same call.

        Parameters
        ----------
        i:
            Site index.

        Returns
        -------
        Tensor
            The environment block at site `i`.

        Raises
        ------
        IndexError
            If `i` is out of range.
        RuntimeError
            If the block is not in memory and no `path` was provided.
        """
        if not (0 <= i < len(self._blocks)):
            raise IndexError(f"site index {i} out of range for L={len(self._blocks)}")

        # --- Resolve block from memory, prefetch future, or disk --------
        if self._blocks[i] is not None:
            block = self._blocks[i]
        elif i in self._read_futures:
            block = self._read_futures.pop(i).result()
            self._blocks[i] = block
        elif self._path is not None:
            fp = self._path / f"{i:05d}.pt"
            logger.debug("environ: synchronous load of block %d (prefetch missed)", i)
            block = _load(fp)
            self._blocks[i] = block
        else:
            raise RuntimeError(
                f"environment block at site {i} has not been initialized"
            )

        # --- Sliding-window maintenance ---------------------------------
        if self._path is not None:
            # Step 1: evict the block that dropped off the back of the window,
            # using the current direction *before* any possible flip.
            self._evict(i - self._direction * self._window)

            # Step 2: flip direction if the next step would exit the fetch range.
            next_i = i + self._direction
            if not (self._fetch_lo <= next_i <= self._fetch_hi):
                self._direction = -self._direction

            # Step 3: prefetch blocks ahead in the (possibly new) direction.
            for k in range(1, self._window):
                self._prefetch(i + self._direction * k)

        return block

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Flush all pending I/O and release the background thread.

        Waits for every outstanding write future to complete, discards
        pending prefetch futures, and shuts down the executor. Safe to call
        even when `path` is `None` (becomes a no-op).
        """
        for fut in self._write_futures.values():
            fut.result()
        self._write_futures.clear()
        # Cancel (drain) pending prefetch futures without waiting — the
        # loaded blocks would not be used after shutdown anyway.
        for fut in self._read_futures.values():
            try:
                fut.result()
            except Exception:  # noqa: BLE001
                pass
        self._read_futures.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_executor(self) -> ThreadPoolExecutor:
        """Return the shared executor, creating it lazily on first use."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        return self._executor

    def _submit(self, fn, *args):
        """Submit a callable to the executor (sync or async depending on config)."""
        if self._async_io:
            return self._get_executor().submit(fn, *args)
        # Synchronous fallback: run now and hand back an already-resolved
        # Future so callers can treat both paths identically.
        fut: Future = Future()
        fut.set_result(fn(*args))
        return fut

    def _evict(self, i: int) -> None:
        """Free the in-memory block at site `i`, waiting for pending writes first.

        Out-of-range indices are silently ignored.
        """
        if not (0 <= i < len(self._blocks)):
            return
        # Wait for any in-flight write so the block is safely on disk first.
        fut = self._write_futures.pop(i, None)
        if fut is not None:
            fut.result()
        self._blocks[i] = None

    def _prefetch(self, i: int) -> None:
        """Schedule an asynchronous load of block `i` into memory.

        Skipped if: `i` is out of the fetch range, `_path` is `None`, the
        cache file does not yet exist, the block is already in memory, or a
        read is already in flight.
        """
        if not (self._fetch_lo <= i <= self._fetch_hi):
            return
        if self._path is None:
            return
        if self._blocks[i] is not None:
            return
        if i in self._read_futures:
            return
        fp = self._path / f"{i:05d}.pt"
        if not fp.exists():
            return
        self._read_futures[i] = self._submit(_load, fp)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        n_init = sum(1 for b in self._blocks if b is not None)
        return f"Environment(L={len(self._blocks)}, initialized={n_init})"


# ---------------------------------------------------------------------------
# Module-level I/O helpers (plain functions for clean executor submission)
# ---------------------------------------------------------------------------

def _save(fp: Path, block: Tensor) -> None:
    """Serialize and write a single environment block to `fp`."""
    torch.save(_serialize_tensor(block), fp)


def _load(fp: Path) -> Tensor:
    """Load and deserialize a single environment block from `fp`."""
    return _deserialize_tensor(torch.load(fp, weights_only=True))


# ---------------------------------------------------------------------------
# Boundary constructors
# ---------------------------------------------------------------------------

def left_env_boundary(mps: MPS, mpo: MPO) -> Tensor:
    """Build the trivial left boundary environment tensor.

    Constructs a rank-3 tensor with all indices of dimension 1 representing
    the left boundary. The axes follow the convention
    `(a=bra_left, o=mpo_left, b=ket_left)`.

    The construction mirrors the boundary initialization in
    `alice.network.observe._observe_mps`.

    Parameters
    ----------
    mps:
        The MPS whose left boundary index is used.
    mpo:
        The MPO whose left boundary index is used.

    Returns
    -------
    Tensor
        Rank-3 boundary tensor with axes `(bra_left, mpo_left, ket_left)`.
    """
    # Build a 2-index identity on the left (dim-1) bond of mps[0].
    # identity() returns a tensor with axes (IN, OUT) both carrying the
    # same index; retagging both axes to the MPS left-bond itag makes the
    # bra and ket indices share that tag so later einsum steps contract them.
    E = identity(mps[0].indices[0])
    E.retag([0, 1], [mps[0].itags[0], mps[0].itags[0]])
    # Insert the MPO left-bond index at axis 1 (between the bra and ket indices).
    E.insert_index(1, direction=Direction.OUT, itag=mpo[0].itags[0])
    # E axes: (bra_left=a, mpo_left=o, ket_left=b)
    return E


def right_env_boundary(mps: MPS, mpo: MPO) -> Tensor:
    """Build the trivial right boundary environment tensor.

    Constructs a rank-3 tensor with all indices of dimension 1 representing
    the right boundary. The axes follow the convention
    `(c=bra_right, p=mpo_right, d=ket_right)`.

    Parameters
    ----------
    mps:
        The MPS whose right boundary index is used.
    mpo:
        The MPO whose right boundary index is used.

    Returns
    -------
    Tensor
        Rank-3 boundary tensor with axes `(bra_right, mpo_right, ket_right)`.
    """
    L = mps.L
    # Right bonds of the last site (dim-1 for OBC).
    E = identity(mps[L - 1].indices[1])
    E.retag([0, 1], [mps[L - 1].itags[1], mps[L - 1].itags[1]])
    # The MPO right boundary has direction IN (opposite to the left boundary OUT).
    E.insert_index(1, direction=Direction.IN, itag=mpo[L - 1].itags[1])
    # E axes: (bra_right=c, mpo_right=p, ket_right=d)
    return E


# ---------------------------------------------------------------------------
# Transfer-matrix update steps (pure functions)
# ---------------------------------------------------------------------------

def step_left_env(E: Tensor, M: Tensor, W: Tensor) -> Tensor:
    """Absorb one site rightward into the left environment.

    Contracts the current left environment `E`, the (conjugate) bra MPS
    tensor, the MPO tensor, and the ket MPS tensor to produce the updated
    left environment one site to the right.

    Index convention:

    - `E(a, o, b)` — `(bra_left, mpo_left, ket_left)`
    - `M.conj(a, c, r)` — `(bra_left, bra_right, phys_bra)`
    - `W(o, p, r, s)` — `(mpo_left, mpo_right, phys_bra, phys_ket)`
    - `M(b, d, s)` — `(ket_left, ket_right, phys_ket)`
    - output `(c, p, d)` — `(bra_right, mpo_right, ket_right)`

    Parameters
    ----------
    E:
        Current left environment with axes `(bra_left, mpo_left, ket_left)`.
    M:
        MPS site tensor with axes `(left_bond, right_bond, physical)`.
    W:
        MPO site tensor with axes `(left_bond, right_bond, phys_bra, phys_ket)`.

    Returns
    -------
    Tensor
        Updated left environment with axes `(bra_right, mpo_right, ket_right)`.
    """
    return einsum('aob,acr,oprs,bds->cpd', E, M.conj(), W, M)


def step_right_env(E: Tensor, M: Tensor, W: Tensor) -> Tensor:
    """Absorb one site leftward into the right environment.

    Contracts the current right environment `E`, the (conjugate) bra MPS
    tensor, the MPO tensor, and the ket MPS tensor to produce the updated
    right environment one site to the left.

    Index convention:

    - `E(c, p, d)` — `(bra_right, mpo_right, ket_right)`
    - `M.conj(a, c, r)` — `(bra_left, bra_right, phys_bra)`
    - `W(o, p, r, s)` — `(mpo_left, mpo_right, phys_bra, phys_ket)`
    - `M(b, d, s)` — `(ket_left, ket_right, phys_ket)`
    - output `(a, o, b)` — `(bra_left, mpo_left, ket_left)`

    Parameters
    ----------
    E:
        Current right environment with axes `(bra_right, mpo_right, ket_right)`.
    M:
        MPS site tensor with axes `(left_bond, right_bond, physical)`.
    W:
        MPO site tensor with axes `(left_bond, right_bond, phys_bra, phys_ket)`.

    Returns
    -------
    Tensor
        Updated right environment with axes `(bra_left, mpo_left, ket_left)`.
    """
    return einsum('cpd,acr,oprs,bds->aob', E, M.conj(), W, M)


# ---------------------------------------------------------------------------
# Bulk initialization
# ---------------------------------------------------------------------------

def build_left_envs(mps: MPS, mpo: MPO, env_left: Environment) -> None:
    """Populate all left environment blocks by sweeping left-to-right.

    Starting from the trivial left boundary at site `0`, absorbs each site in
    turn (going rightward) and stores the value in `env_left`. After this call
    every slot `env_left[0]` … `env_left[fetch_hi]` is filled, where
    `fetch_hi` is the highest index `env_left` will ever be asked for. Blocks
    above `fetch_hi` are never computed (e.g. `env_left[L-1]` is skipped
    entirely in 2-site mode where `fetch_hi == L-2`).

    This function requires `mps` to be in left-canonical form with
    `center == L-1`, so that each site tensor is already left-isometric.

    Parameters
    ----------
    mps:
        Left-canonical MPS (`center == L-1`).
    mpo:
        Hamiltonian MPO of the same length.
    env_left:
        `Environment` instance to populate in-place.

    Raises
    ------
    ValueError
        If `mps.center != L-1`.
    """
    L = mps.L
    if mps.center != L - 1:
        raise ValueError(
            f"build_left_envs requires mps.center == L-1 (= {L - 1}), "
            f"got center={mps.center}"
        )
    # Initialize the left boundary (site 0 has a trivial left bond for OBC).
    env_left[0] = left_env_boundary(mps, mpo)
    # Sweep left-to-right: env_left[i+1] accumulates sites 0 … i.
    # When disk caching is active, evict block i immediately after it has
    # been consumed to compute block i+1 — its on-disk copy is already present
    # from the __setitem__ write, so keeping it in memory serves no purpose.
    # The window [fetch_hi - window + 1, fetch_hi] is preserved because those
    # are the blocks the first backward sweep will need immediately.
    _cache = env_left._path is not None
    if _cache:
        _keep_hi = env_left._fetch_hi
        _keep_lo = _keep_hi - env_left._window + 1
    # Stop at fetch_hi: blocks above it are never fetched (e.g. env_left[L-1]
    # in 2-site mode where fetch_hi == L-2), so there is no reason to compute them.
    for i in range(env_left._fetch_hi):
        env_left[i + 1] = step_left_env(env_left[i], mps[i], mpo[i])
        if _cache and not (_keep_lo <= i <= _keep_hi):
            env_left._evict(i)


def build_right_envs(mps: MPS, mpo: MPO, env_right: Environment) -> None:
    """Populate all right environment blocks by sweeping right-to-left.

    Starting from the trivial right boundary at site `L-1`, absorbs each
    site in turn (going leftward) and stores the value in `env_right`.
    After this call every slot `env_right[fetch_lo]` … `env_right[L-1]` is
    filled, where `fetch_lo` is the lowest index `env_right` will ever be
    asked for. Blocks below `fetch_lo` are never computed (e.g. `env_right[0]`
    is skipped entirely in 2-site mode where `fetch_lo == 1`).

    This function requires `mps` to be in right-canonical form with
    `center == 0`, so that each site tensor is already right-isometric.

    Parameters
    ----------
    mps:
        Right-canonical MPS (`center == 0`).
    mpo:
        Hamiltonian MPO of the same length.
    env_right:
        `Environment` instance to populate in-place.

    Raises
    ------
    ValueError
        If `mps.center != 0`.
    """
    if mps.center != 0:
        raise ValueError(
            f"build_right_envs requires mps.center == 0, got center={mps.center}"
        )
    L = mps.L
    # Initialize the right boundary (site L-1 has a trivial right bond for OBC).
    env_right[L - 1] = right_env_boundary(mps, mpo)
    # Sweep right-to-left: env_right[i] accumulates sites i+1 … L-1.
    # When disk caching is active, evict block i+1 immediately after it has
    # been consumed to compute block i — its on-disk copy is already present
    # from the __setitem__ write, so keeping it in memory serves no purpose.
    # The window [fetch_lo, fetch_lo + window - 1] is preserved because those
    # are the blocks the first forward sweep will need immediately.
    _cache = env_right._path is not None
    if _cache:
        _keep_lo = env_right._fetch_lo
        _keep_hi = _keep_lo + env_right._window - 1
    # Stop at fetch_lo: blocks below it are never fetched (e.g. env_right[0]
    # in 2-site mode where fetch_lo == 1), so there is no reason to compute them.
    for i in range(L - 2, env_right._fetch_lo - 1, -1):
        env_right[i] = step_right_env(env_right[i + 1], mps[i + 1], mpo[i + 1])
        if _cache and not (_keep_lo <= i + 1 <= _keep_hi):
            env_right._evict(i + 1)
