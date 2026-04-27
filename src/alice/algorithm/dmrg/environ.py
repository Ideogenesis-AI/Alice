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

from typing import List, Optional

from nicole import Direction, Tensor, einsum, identity

from alice.network import MPS, MPO


class Environment:
    """Environment block storage for one sweep direction.

    A single class instantiated twice per DMRG run — once as `env_left` and
    once as `env_right`. The `__getitem__` / `__setitem__` interface mirrors
    `MPS` and `MPO` for consistency.

    The `fetch` / `cache` methods provide hooks for future disk-based or
    distributed caching strategies. In the current implementation `fetch`
    simply returns the in-memory block and `cache` is a placeholder that
    raises `NotImplementedError`.

    Parameters
    ----------
    L:
        Number of sites in the chain.
    """

    def __init__(self, L: int) -> None:
        self._blocks: List[Optional[Tensor]] = [None] * L

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
            If the block has not been initialised yet.
        """
        if not (0 <= i < len(self._blocks)):
            raise IndexError(f"site index {i} out of range for L={len(self._blocks)}")
        block = self._blocks[i]
        if block is None:
            raise RuntimeError(f"environment block at site {i} has not been initialised")
        return block

    def __setitem__(self, i: int, E: Tensor) -> None:
        """Store the environment block at site `i`.

        Raises
        ------
        IndexError
            If `i` is out of range.
        """
        if not (0 <= i < len(self._blocks)):
            raise IndexError(f"site index {i} out of range for L={len(self._blocks)}")
        self._blocks[i] = E

    # ------------------------------------------------------------------
    # Cache interface (placeholders for future persistence layer)
    # ------------------------------------------------------------------

    def fetch(self, i: int) -> Tensor:
        """Return block `i`, loading from persistent cache if absent.

        Currently identical to `__getitem__`. Future implementations may
        load a serialised block from disk when the in-memory entry is `None`.

        Parameters
        ----------
        i:
            Site index.

        Returns
        -------
        Tensor
            The environment block at site `i`.
        """
        return self[i]

    def cache(self, i: int, **kwargs) -> None:
        """Serialise block `i` to a persistent store.

        Not yet implemented. Future versions will write the block to disk
        (HDF5, memory-mapped arrays, etc.) and may evict it from the
        in-memory store to control peak memory usage.

        Parameters
        ----------
        i:
            Site index of the block to serialise.

        Raises
        ------
        NotImplementedError
            Always, until a persistence backend is provided.
        """
        raise NotImplementedError("disk caching is not yet implemented")

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        n_init = sum(1 for b in self._blocks if b is not None)
        return f"Environment(L={len(self._blocks)}, initialised={n_init})"


# ---------------------------------------------------------------------------
# Boundary constructors
# ---------------------------------------------------------------------------

def left_env_boundary(mps: MPS, mpo: MPO) -> Tensor:
    """Build the trivial left boundary environment tensor.

    Constructs a rank-3 tensor with all indices of dimension 1 representing
    the left boundary. The axes follow the convention
    `(a=bra_left, o=mpo_left, b=ket_left)`.

    The construction mirrors the boundary initialisation in
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
# Bulk initialisation
# ---------------------------------------------------------------------------

def build_right_envs(mps: MPS, mpo: MPO, env_right: Environment) -> None:
    """Populate all right environment blocks by sweeping right-to-left.

    Starting from the trivial right boundary at site `L-1`, absorbs each
    site in turn (going leftward) and stores the value in `env_right`.
    After this call every slot `env_right[0]` … `env_right[L-1]` is filled.

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
    # Initialise the right boundary (site L-1 has a trivial right bond for OBC).
    env_right[L - 1] = right_env_boundary(mps, mpo)
    # Sweep right-to-left: env_right[i] accumulates sites i+1 … L-1.
    for i in range(L - 2, -1, -1):
        env_right[i] = step_right_env(env_right[i + 1], mps[i + 1], mpo[i + 1])
