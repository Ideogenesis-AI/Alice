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


"""XTRG environment block storage and transfer-matrix update functions.

An environment block `E` is a rank-3 tensor with axes
`(c_bond, a_bond, b_bond)` where `c` is the bond of the compressed MPO C,
`a` is the bond of factor MPO A, and `b` is the bond of factor MPO B.
Two `Environment` instances are used per variational compression run — one
for the left environments and one for the right environments.

Index convention used throughout this module:

- `c`, `d` — left and right bond indices of the compressed MPO C (bra side)
- `a`, `p` — left and right bond indices of factor MPO A
- `b`, `q` — left and right bond indices of factor MPO B
- `r`       — phys_in of A and C (shared)
- `x`       — shared internal physical index (A phys_out = B phys_in)
- `s`       — phys_out of B and C (shared)

Left environment at site `i` (`env_left[i]`) has axes
`(c, a, b)` and accumulates sites `0…i-1`.
`env_left[0]` is the trivial left boundary.

Right environment at site `i` (`env_right[i]`) has axes
`(d, p, q)` and accumulates sites `i+1…L-1`.
`env_right[L-1]` is the trivial right boundary.

The axis structure of left and right environments is semantically identical
— both carry `(C_bond, A_bond, B_bond)` at a chain boundary. Different
letter names are used within einsums to avoid name collisions.
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

from alice.network.thermal import NormalMPO

logger = logging.getLogger(__name__)


class Environment:
    """Environment block storage for one sweep direction with optional disk caching.

    A single class instantiated twice per variational compression run — once
    as `env_left` and once as `env_right`. The `__getitem__` / `__setitem__`
    interface mirrors `MPS` and `MPO` for consistency.

    When `path` is provided, the class maintains a sliding window of
    `window` blocks in memory and caches the rest to disk. Disk writes are
    submitted asynchronously via a `ThreadPoolExecutor` so they overlap with
    the local update step. Prefetch reads are also submitted asynchronously;
    `fetch` waits on them if needed.

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
        Lowest site index that will ever be passed to `fetch`. Defaults
        to `0`.
    fetch_hi:
        Highest site index that will ever be passed to `fetch`. Defaults
        to `L-1`.
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
        fetch_range = max(1, self._fetch_hi - self._fetch_lo + 1)
        self._window: int = min(window, fetch_range)

        # Sweep direction: +1 = forward (increasing index), -1 = backward.
        self._direction: int = 1

        self._executor: Optional[ThreadPoolExecutor] = None
        self._write_futures: Dict[int, Future] = {}
        self._read_futures: Dict[int, Future] = {}

    # ------------------------------------------------------------------
    # Index-style access
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

        Raises
        ------
        IndexError
            If `i` is out of range.
        """
        if not (0 <= i < len(self._blocks)):
            raise IndexError(f"site index {i} out of range for L={len(self._blocks)}")
        self._blocks[i] = E
        self._read_futures.pop(i, None)

        if self._path is not None:
            prev = self._write_futures.pop(i, None)
            if prev is not None:
                prev.result()
            self._write_futures[i] = self._submit(_save, self._path / f"{i:05d}.pt", E)

    # ------------------------------------------------------------------
    # Fetch — sliding-window read with direction tracking
    # ------------------------------------------------------------------

    def fetch(self, i: int) -> Tensor:
        """Return block `i`, loading from disk if evicted, then update the window.

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

        if self._path is not None:
            self._evict(i - self._direction * self._window)
            next_i = i + self._direction
            if not (self._fetch_lo <= next_i <= self._fetch_hi):
                self._direction = -self._direction
            for k in range(1, self._window):
                self._prefetch(i + self._direction * k)

        return block

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Flush all pending I/O and release the background thread."""
        for fut in self._write_futures.values():
            fut.result()
        self._write_futures.clear()
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
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        return self._executor

    def _submit(self, fn, *args):
        if self._async_io:
            return self._get_executor().submit(fn, *args)
        result = fn(*args)
        fut: Future = Future()
        fut.set_result(result)
        return fut

    def _evict(self, i: int) -> None:
        if not (0 <= i < len(self._blocks)):
            return
        fut = self._write_futures.pop(i, None)
        if fut is not None:
            fut.result()
        self._blocks[i] = None

    def _prefetch(self, i: int) -> None:
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

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        n_init = sum(1 for b in self._blocks if b is not None)
        return f"Environment(L={len(self._blocks)}, initialized={n_init})"


# ---------------------------------------------------------------------------
# Module-level I/O helpers
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

def left_env_boundary(mpo_a: NormalMPO, mpo_b: NormalMPO, mpo_c: NormalMPO) -> Tensor:
    """Build the trivial left boundary environment tensor.

    Constructs a rank-3 tensor with all indices of dimension 1 representing
    the left boundary. The axes follow the convention `(c, a, b)` =
    `(C_left, A_left, B_left)`.

    For OBC all left bonds at site 0 are dim-1 (vacuum sector), so the
    returned tensor is effectively the scalar 1 dressed with the correct
    symmetry-sector labels.

    Parameters
    ----------
    mpo_a:
        Factor MPO A.
    mpo_b:
        Factor MPO B.
    mpo_c:
        Compressed MPO C (the one being optimized).

    Returns
    -------
    Tensor
        Rank-3 boundary tensor with axes `(C_left, A_left, B_left)`.
    """
    # identity on C's left bond gives a 2-leg IN/OUT pair (both dim-1 for OBC).
    E = identity(mpo_c[0].indices[0])
    # Retag: axis 0 → C's left itag, axis 1 → B's left itag.
    E.retag([0, 1], [mpo_c[0].itags[0], mpo_b[0].itags[0]])
    # Insert A's left bond at axis 1 (between the C and B axes).
    E.insert_index(1, direction=Direction.OUT, itag=mpo_a[0].itags[0])
    # E axes: (c=C_left, a=A_left, b=B_left)
    return E


def right_env_boundary(mpo_a: NormalMPO, mpo_b: NormalMPO, mpo_c: NormalMPO) -> Tensor:
    """Build the trivial right boundary environment tensor.

    Constructs a rank-3 tensor with all indices of dimension 1 representing
    the right boundary. The axes follow the convention `(d, p, q)` =
    `(C_right, A_right, B_right)`.

    Parameters
    ----------
    mpo_a:
        Factor MPO A.
    mpo_b:
        Factor MPO B.
    mpo_c:
        Compressed MPO C (the one being optimized).

    Returns
    -------
    Tensor
        Rank-3 boundary tensor with axes `(C_right, A_right, B_right)`.
    """
    L = mpo_c.L
    # identity on C's right bond (dim-1 for OBC, direction OUT).
    E = identity(mpo_c[L - 1].indices[1])
    # Retag: axis 0 → C's right itag, axis 1 → B's right itag.
    E.retag([0, 1], [mpo_c[L - 1].itags[1], mpo_b[L - 1].itags[1]])
    # Insert A's right bond at axis 1; direction IN (opposite to the left boundary).
    E.insert_index(1, direction=Direction.IN, itag=mpo_a[L - 1].itags[1])
    # E axes: (d=C_right, p=A_right, q=B_right)
    return E


# ---------------------------------------------------------------------------
# Transfer-matrix update steps (pure functions)
# ---------------------------------------------------------------------------

def step_left_env(E: Tensor, A_i: Tensor, B_i: Tensor, C_i: Tensor) -> Tensor:
    """Absorb one site rightward into the left environment.

    Contracts the current left environment `E`, the factor tensors `A_i` and
    `B_i`, and the conjugate compressed tensor `C_i` to produce the updated
    left environment one site to the right.

    Index convention:

    - `E(c, a, b)`         — `(C_left, A_left, B_left)`
    - `A_i(a, p, r, x)`   — `(A_left, A_right, phys_in, phys_internal)`
    - `B_i(b, q, x, s)`   — `(B_left, B_right, phys_internal_in, phys_out)`
    - `C_i.conj(c, d, r, s)` — `(C_left, C_right, phys_in, phys_out)`
    - output `(d, p, q)`  — `(C_right, A_right, B_right)`

    Parameters
    ----------
    E:
        Current left environment with axes `(C_left, A_left, B_left)`.
    A_i:
        Factor MPO A at site i with axes `(left, right, phys_in, phys_internal)`.
    B_i:
        Factor MPO B at site i with axes `(left, right, phys_internal_in, phys_out)`.
    C_i:
        Compressed MPO C at site i with axes `(left, right, phys_in, phys_out)`.

    Returns
    -------
    Tensor
        Updated left environment with axes `(C_right, A_right, B_right)`.
    """
    # Decomposed into sequential 2-tensor einsums to avoid itag-matching ambiguity
    # when A, B, C share the same bond itag family.
    # Step 1: E(c,a,b) × A_i(a,p,r,x) → (c,b,p,r,x), contracting 'a'.
    t1 = einsum('cab,aprx->cbprx', E, A_i)
    # Step 2: t1(c,b,p,r,x) × B_i(b,q,x,s) → (c,p,r,q,s), contracting 'b','x'.
    t2 = einsum('cbprx,bqxs->cprqs', t1, B_i)
    # Step 3: t2(c,p,r,q,s) × C_i†(c,d,r,s) → (d,p,q), contracting 'c','r','s'.
    return einsum('cprqs,cdrs->dpq', t2, C_i.conj())


def step_right_env(E: Tensor, A_i: Tensor, B_i: Tensor, C_i: Tensor) -> Tensor:
    """Absorb one site leftward into the right environment.

    Contracts the current right environment `E`, the factor tensors `A_i` and
    `B_i`, and the conjugate compressed tensor `C_i` to produce the updated
    right environment one site to the left.

    Index convention:

    - `E(d, p, q)`         — `(C_right, A_right, B_right)`
    - `A_i(a, p, r, x)`   — `(A_left, A_right, phys_in, phys_internal)`
    - `B_i(b, q, x, s)`   — `(B_left, B_right, phys_internal_in, phys_out)`
    - `C_i.conj(c, d, r, s)` — `(C_left, C_right, phys_in, phys_out)`
    - output `(c, a, b)`  — `(C_left, A_left, B_left)`

    Parameters
    ----------
    E:
        Current right environment with axes `(C_right, A_right, B_right)`.
    A_i:
        Factor MPO A at site i with axes `(left, right, phys_in, phys_internal)`.
    B_i:
        Factor MPO B at site i with axes `(left, right, phys_internal_in, phys_out)`.
    C_i:
        Compressed MPO C at site i with axes `(left, right, phys_in, phys_out)`.

    Returns
    -------
    Tensor
        Updated right environment with axes `(C_left, A_left, B_left)`.
    """
    # Decomposed into sequential 2-tensor einsums to avoid itag-matching ambiguity.
    # Step 1: E(d,p,q) × A_i(a,p,r,x) → (d,a,q,r,x), contracting 'p'.
    t1 = einsum('dpq,aprx->daqrx', E, A_i)
    # Step 2: t1(d,a,q,r,x) × B_i(b,q,x,s) → (d,a,r,b,s), contracting 'q','x'.
    t2 = einsum('daqrx,bqxs->darbs', t1, B_i)
    # Step 3: t2(d,a,r,b,s) × C_i†(c,d,r,s) → (c,a,b), contracting 'd','r','s'.
    return einsum('darbs,cdrs->cab', t2, C_i.conj())


# ---------------------------------------------------------------------------
# Bulk initialisation
# ---------------------------------------------------------------------------

def build_left_envs(
    mpo_a: NormalMPO,
    mpo_b: NormalMPO,
    mpo_c: NormalMPO,
    env_left: Environment,
) -> None:
    """Populate all left environment blocks by sweeping left-to-right.

    Starting from the trivial left boundary at site `0`, absorbs each site in
    turn (going rightward) and stores the result in `env_left`. Requires
    `mpo_c` to be in left-canonical form (`center == L-1`), so that each site
    tensor is left-isometric.

    Parameters
    ----------
    mpo_a:
        Factor MPO A.
    mpo_b:
        Factor MPO B.
    mpo_c:
        Left-canonical compressed MPO (`center == L-1`).
    env_left:
        `Environment` instance to populate in-place.

    Raises
    ------
    ValueError
        If `mpo_c.center != L-1`.
    """
    L = mpo_c.L
    if mpo_c.center != L - 1:
        raise ValueError(
            f"build_left_envs requires mpo_c.center == L-1 (= {L - 1}), "
            f"got center={mpo_c.center}"
        )
    env_left[0] = left_env_boundary(mpo_a, mpo_b, mpo_c)
    _cache = env_left._path is not None
    if _cache:
        _keep_hi = env_left._fetch_hi
        _keep_lo = _keep_hi - env_left._window + 1
    for i in range(env_left._fetch_hi):
        env_left[i + 1] = step_left_env(env_left[i], mpo_a[i], mpo_b[i], mpo_c[i])
        if _cache and not (_keep_lo <= i <= _keep_hi):
            env_left._evict(i)


def build_right_envs(
    mpo_a: NormalMPO,
    mpo_b: NormalMPO,
    mpo_c: NormalMPO,
    env_right: Environment,
) -> None:
    """Populate all right environment blocks by sweeping right-to-left.

    Starting from the trivial right boundary at site `L-1`, absorbs each site
    in turn (going leftward) and stores the result in `env_right`. Requires
    `mpo_c` to be in right-canonical form (`center == 0`), so that each site
    tensor is right-isometric.

    Parameters
    ----------
    mpo_a:
        Factor MPO A.
    mpo_b:
        Factor MPO B.
    mpo_c:
        Right-canonical compressed MPO (`center == 0`).
    env_right:
        `Environment` instance to populate in-place.

    Raises
    ------
    ValueError
        If `mpo_c.center != 0`.
    """
    if mpo_c.center != 0:
        raise ValueError(
            f"build_right_envs requires mpo_c.center == 0, got center={mpo_c.center}"
        )
    L = mpo_c.L
    env_right[L - 1] = right_env_boundary(mpo_a, mpo_b, mpo_c)
    _cache = env_right._path is not None
    if _cache:
        _keep_lo = env_right._fetch_lo
        _keep_hi = _keep_lo + env_right._window - 1
    for i in range(L - 2, env_right._fetch_lo - 1, -1):
        env_right[i] = step_right_env(env_right[i + 1], mpo_a[i + 1], mpo_b[i + 1], mpo_c[i + 1])
        if _cache and not (_keep_lo <= i + 1 <= _keep_hi):
            env_right._evict(i + 1)
