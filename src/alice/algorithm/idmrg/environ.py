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


"""Unit-cell transfer operators and their regularized fixed points.

An infinite chain has no boundary to start an environment from, so the
boundary environments of a unit cell are fixed points of the transfer
operator that carries one whole cell. This module builds those operators
and solves for the fixed points, and in doing so measures the energy.

Index convention:

- `a, b, c, d` — virtual bond indices of the MPS, bra or ket side
- `o, p, q`    — virtual bond indices of the MPO
- `r, s`       — physical indices
- `x, y`       — the dimension-1 cap axes that stand in for a scalar

The left environment `E(a, o, b)` sits on bond `0` in the `AL` basis and
accumulates every site to its left; the right environment `E(c, p, d)` sits
on the same bond in the `AR` basis and accumulates every site to its right.

Why the fixed point is singular
-------------------------------

The Hamiltonian iMPO is upper triangular with an identity in both corners
(see `alice.network.autoimpo`), so the channels of the left environment
close in order:

- The start channel sees only itself, and its equation `E = T_AL(E)` is
  solved by the identity, because `AL` is left-orthonormal.
- An operator-string channel cannot feed itself: the string advances one
  bond per site and a finite-range term eventually terminates. The strictly
  triangular part is therefore nilpotent and these channels converge in a
  finite number of cells.
- The finish channel sees `E = Y + T_AL(E)`, and `T_AL` has eigenvalue `1`.
  There is no solution: each application adds one more unit cell of energy,
  so the naive iteration grows without bound.

The cure (Zauner-Stauber et al. 2018) is to subtract, at every step, the
component along the offending direction. The subtracted amount is one unit
cell of energy, which is why this module measures the energy rather than
needing a separate contraction for it. Fixing the component to zero also
picks one solution out of the family that differ by a multiple of the
identity, so the environment returned here is well defined.

Nicole has no Krylov solver, so the fixed point is reached by plain
regularized power iteration, converging as the second eigenvalue of the
MPS transfer operator to the power of the number of unit cells absorbed.
A gapped model needs tens of cells. A critical one needs hundreds and may
not get there at all, because its second eigenvalue approaches one as the
bond dimension grows: that is why running out of cells is a warning here
and not an error. This is the accuracy bottleneck of the whole package,
and the thing a Krylov solve would fix.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from nicole import Direction, Index, Tensor, einsum, identity
from nicole import deserialize as _deserialize_tensor
from nicole import serialize as _serialize_tensor

from alice.network import iMPO, iMPS
from alice.network.autoimpo import boundary_vectors

logger = logging.getLogger(__name__)

# itag for the dimension-1 axis that lets a scalar be carried as a tensor.
# Nicole rejects tensors with a single index, so a quantity contracted down
# to a number is held as a matrix with two trivial axes instead.
_CAP_ITAG = '_cap_'

# Relative tolerance on the imaginary part of the energy. A Hermitian
# Hamiltonian contracted against a complex state leaves an imaginary residue
# at roundoff level; anything larger means the operator is not Hermitian.
_IMAG_TOL = 1e-10


# ---------------------------------------------------------------------------
# Single-site environment steps
# ---------------------------------------------------------------------------

def step_left_env(E: Tensor, M: Tensor, W: Tensor) -> Tensor:
    """Absorb one site rightward into the left environment.

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
        iMPS site tensor with axes `(left_bond, right_bond, physical)`.
    W:
        iMPO site tensor with axes `(left_bond, right_bond, phys_bra, phys_ket)`.

    Returns
    -------
    Tensor
        Updated left environment with axes `(bra_right, mpo_right, ket_right)`.
    """
    return einsum('aob,acr,oprs,bds->cpd', E, M.conj(), W, M)


def step_right_env(E: Tensor, M: Tensor, W: Tensor) -> Tensor:
    """Absorb one site leftward into the right environment.

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
        iMPS site tensor with axes `(left_bond, right_bond, physical)`.
    W:
        iMPO site tensor with axes `(left_bond, right_bond, phys_bra, phys_ket)`.

    Returns
    -------
    Tensor
        Updated right environment with axes `(bra_left, mpo_left, ket_left)`.
    """
    return einsum('cpd,acr,oprs,bds->aob', E, M.conj(), W, M)


# ---------------------------------------------------------------------------
# Scalar extraction
# ---------------------------------------------------------------------------

def _real_scalar(t: Tensor, label: str) -> float:
    """Read the real number carried by a tensor whose every axis is trivial.

    Mirrors the boundary reader in `alice.network.observe`, including the
    Bridge normalization weight that is not unity for a generic symmetry
    group. A tensor with no block at all is an exactly vanishing scalar,
    which happens whenever every symmetry sector of the contraction is
    annihilated.

    Parameters
    ----------
    t:
        Tensor holding at most one block of one element.
    label:
        Name of the quantity, used in the error message.

    Returns
    -------
    float
        The real part of the scalar, weighted.

    Raises
    ------
    ValueError
        If the imaginary part is too large to be roundoff.
    """
    if not t.data:
        return 0.0

    key, block = next(iter(t.data.items()))
    weight = 1.0 if t.intw is None else float(t.intw[key].weights[0, 0].real)
    z = complex(block.item()) * weight
    if abs(z.imag) > _IMAG_TOL * max(abs(z), 1.0):
        raise ValueError(
            f"{label} = {z!r} has a non-negligible imaginary part; the "
            "Hamiltonian is not Hermitian on this state"
        )
    return z.real


def _capped_identity(idx: Index, itag: str, direction: Direction) -> Tensor:
    """Build an identity on `idx` with a trivial axis wedged in the middle.

    The trivial axis is what lets a boundary vector be attached: the
    selectors from `boundary_vectors` carry a cap of their own, and
    contracting the two caps leaves a clean 3-index environment.

    Parameters
    ----------
    idx:
        Bond index the identity acts on.
    itag:
        itag worn by both bond axes, so a later `einsum` pairs them.
    direction:
        Direction of the inserted cap, chosen to oppose the cap of the
        selector it will meet.

    Returns
    -------
    Tensor
        3-index tensor with axes `(bond, cap, bond)`.
    """
    delta = identity(idx)
    delta.retag([0, 1], [itag, itag])
    delta.insert_index(1, direction=direction, itag=_CAP_ITAG)
    return delta


# ---------------------------------------------------------------------------
# MPS transfer operators
# ---------------------------------------------------------------------------

def transfer_left(rho: Tensor, tensors: Sequence[Tensor]) -> Tensor:
    """Carry a bond matrix rightward through one unit cell.

    Computes the sum over physical indices of `A.conj() rho A` for each site
    in turn. For left-orthonormal tensors the identity is a fixed point of
    this map, which is what makes the start channel of the left environment
    trivial.

    Parameters
    ----------
    rho:
        2-index tensor on the left bond of `tensors[0]`.
    tensors:
        Unit cell of MPS tensors with axes `(left, right, physical)`.

    Returns
    -------
    Tensor
        2-index tensor on the right bond of the last site, which for a unit
        cell is the bond it started from.
    """
    for t in tensors:
        rho = einsum('ab,acr,bdr->cd', rho, t.conj(), t)
    return rho


def transfer_right(rho: Tensor, tensors: Sequence[Tensor]) -> Tensor:
    """Carry a bond matrix leftward through one unit cell.

    The mirror of `transfer_left`. For right-orthonormal tensors the
    identity is a fixed point.

    Parameters
    ----------
    rho:
        2-index tensor on the right bond of the last site.
    tensors:
        Unit cell of MPS tensors with axes `(left, right, physical)`.

    Returns
    -------
    Tensor
        2-index tensor on the left bond of `tensors[0]`.
    """
    for t in reversed(tensors):
        rho = einsum('cd,acr,bdr->ab', rho, t.conj(), t)
    return rho


def left_density(imps: iMPS, bond: int = 0) -> Tensor:
    """Return the density matrix dual to the left environment.

    This is `C C'` on the given bond, the fixed point of `transfer_left`
    reached from the right, and the object a left environment is traced
    against to produce a number. It has unit trace because `canonical`
    normalizes the bond matrices.

    Parameters
    ----------
    imps:
        Gauge-fixed infinite MPS.
    bond:
        Bond position, taken modulo the unit-cell size.

    Returns
    -------
    Tensor
        2-index tensor in the `AL` basis.
    """
    bond_matrix = imps.bond_matrix(bond)
    return einsum('ab,cb->ac', bond_matrix, bond_matrix.conj())


def right_density(imps: iMPS, bond: int = 0) -> Tensor:
    """Return the density matrix dual to the right environment.

    This is `C' C` on the given bond, the fixed point of `transfer_right`
    reached from the left, and the mirror of `left_density`.

    Parameters
    ----------
    imps:
        Gauge-fixed infinite MPS.
    bond:
        Bond position, taken modulo the unit-cell size.

    Returns
    -------
    Tensor
        2-index tensor in the `AR` basis.
    """
    bond_matrix = imps.bond_matrix(bond)
    return einsum('ab,ac->bc', bond_matrix, bond_matrix.conj())


# ---------------------------------------------------------------------------
# MPO transfer operators
# ---------------------------------------------------------------------------

def cell_left_env(E: Tensor, tensors: Sequence[Tensor], impo: iMPO) -> Tensor:
    """Absorb one whole unit cell into the left environment.

    Applies `step_left_env` once per site. The itags of the infinite
    containers repeat from one cell to the next, so the environment comes
    back wearing the names it started with and the map can be iterated
    without any retagging.

    Parameters
    ----------
    E:
        Left environment with axes `(bra, mpo, ket)` on bond `0`.
    tensors:
        Unit cell of MPS tensors, normally the left isometries.
    impo:
        Hamiltonian iMPO of the same unit cell.

    Returns
    -------
    Tensor
        Left environment on bond `0` of the next cell.
    """
    for i, t in enumerate(tensors):
        E = step_left_env(E, t, impo[i])
    return E


def cell_right_env(E: Tensor, tensors: Sequence[Tensor], impo: iMPO) -> Tensor:
    """Absorb one whole unit cell into the right environment.

    The mirror of `cell_left_env`, applying `step_right_env` from the last
    site backwards.

    Parameters
    ----------
    E:
        Right environment with axes `(bra, mpo, ket)` on bond `0`.
    tensors:
        Unit cell of MPS tensors, normally the right isometries.
    impo:
        Hamiltonian iMPO of the same unit cell.

    Returns
    -------
    Tensor
        Right environment on bond `0` of the previous cell.
    """
    for i in range(len(tensors) - 1, -1, -1):
        E = step_right_env(E, tensors[i], impo[i])
    return E


# ---------------------------------------------------------------------------
# Regularized fixed points
# ---------------------------------------------------------------------------

def _channel_energy(E: Tensor, projector: Tensor, density: Tensor) -> float:
    """Trace one channel of an environment against the density matrix.

    Parameters
    ----------
    E:
        Environment with axes `(bra, mpo, ket)`.
    projector:
        2-index selector picking the channel, with axes `(mpo, cap)`.
    density:
        3-index density matrix with axes `(ket, cap, bra)`.

    Returns
    -------
    float
        The energy carried by that channel.
    """
    channel = einsum('aob,oy->ayb', E, projector)
    return _real_scalar(einsum('ayb,bxa->yx', channel, density), 'energy')


def _fixed_point(
    seed: Tensor,
    divergent: Tensor,
    projector: Tensor,
    density: Tensor,
    absorb: Callable[[Tensor], Tensor],
    tol: float,
    maxiter: int,
    side: str,
) -> Tuple[Tensor, float]:
    """Run the regularized power iteration shared by both environments.

    Absorbing a cell adds one cell of Hamiltonian. Everything converges
    except the component along `divergent`, which grows by the energy of
    that cell every time. Subtracting that growth bounds the iteration and
    hands back the energy for free, and fixing the component to zero is
    what makes the returned environment unique.

    Parameters
    ----------
    seed:
        Environment with the entering channel set to the identity and
        everything else zero.
    divergent:
        Unit environment along the channel that cannot converge.
    projector:
        Selector reading that same channel back out.
    density:
        Density matrix dual to this side, with a cap axis already inserted.
    absorb:
        Callable absorbing one unit cell.
    tol:
        Convergence threshold on the change in the environment.
    maxiter:
        Iteration budget, counted in unit cells.
    side:
        `'left'` or `'right'`, used in messages only.

    Returns
    -------
    tuple
        `(environment, energy)` with the energy counted per unit cell.
        Returned even if the budget ran out, with a warning logged.
    """
    E = seed
    change = float('inf')
    for step in range(1, maxiter + 1):
        absorbed = absorb(E)
        energy = _channel_energy(absorbed, projector, density)
        absorbed = absorbed - divergent * energy

        change = float((absorbed - E).norm())
        E = absorbed
        if change < tol:
            logger.debug(
                "%s environment converged after %d cells, energy %.12g",
                side, step, energy,
            )
            return E, energy

    # The best estimate is returned rather than raised on, because the
    # models this is aimed at are the ones that cannot meet a tight
    # tolerance: regularized power iteration converges as the second
    # eigenvalue of the MPS transfer operator, which approaches one at a
    # critical point. Discarding a nearly converged environment, and with
    # it a whole run, would be the worse failure.
    logger.warning(
        "%s environment did not converge in %d unit cells (residual %.3e, "
        "tolerance %.3e); continuing with the best estimate. Raise maxiter "
        "or loosen tol to silence this.",
        side, maxiter, change, tol,
    )
    return E, energy


def _check_cell(imps: iMPS, impo: iMPO) -> None:
    """Raise unless the state and the Hamiltonian share a unit cell."""
    if imps.N != impo.N:
        raise ValueError(
            f"iMPS and iMPO must share a unit cell, got N={imps.N} and "
            f"N={impo.N}"
        )


def left_environment(
    imps: iMPS,
    impo: iMPO,
    *,
    tol: float = 1e-12,
    maxiter: int = 200,
) -> Tuple[Tensor, float]:
    """Solve for the left boundary environment of a unit cell.

    The environment lands on bond `0` in the `AL` basis and represents the
    whole half-infinite chain to its left, with the energy of that half
    chain subtracted off. Without the subtraction there would be nothing to
    return, since a half-infinite chain has infinite energy.

    Parameters
    ----------
    imps:
        Gauge-fixed infinite MPS. Call `canonical` first.
    impo:
        Hamiltonian iMPO over the same unit cell.
    tol:
        Convergence threshold on the change in the environment.
    maxiter:
        Iteration budget, counted in unit cells.

    Returns
    -------
    tuple
        `(E, energy)`, the environment with axes `(bra, mpo, ket)` and the
        energy of one unit cell. Running out of cells is reported as a
        warning, not an error, and the best estimate is returned.

    Raises
    ------
    ValueError
        If the unit cells disagree, or if `imps` is not gauge fixed.
    """
    _check_cell(imps, impo)
    isometries = imps.left_isometries()

    start, finish = boundary_vectors(impo, 0)
    # `start` arrives as (cap, mpo) because it was built to cap the left end
    # of a window; here it meets an environment instead, so the cap goes last
    # like every other selector in this module.
    start.permute([1, 0], in_place=True)

    delta = _capped_identity(
        isometries[0].indices[0], isometries[0].itags[0], Direction.OUT,
    )
    density = left_density(imps)
    density.insert_index(1, direction=Direction.IN, itag=_CAP_ITAG)

    return _fixed_point(
        seed=einsum('ayb,oy->aob', delta, start),
        divergent=einsum('ayb,oy->aob', delta, finish.conj()),
        projector=finish,
        density=density,
        absorb=lambda E: cell_left_env(E, isometries, impo),
        tol=tol,
        maxiter=maxiter,
        side='left',
    )


def right_environment(
    imps: iMPS,
    impo: iMPO,
    *,
    tol: float = 1e-12,
    maxiter: int = 200,
) -> Tuple[Tensor, float]:
    """Solve for the right boundary environment of a unit cell.

    The mirror of `left_environment`: the chain to the right of bond `0`,
    in the `AR` basis, with its energy subtracted. The two environments
    describe disjoint halves of the same chain and must report the same
    energy per unit cell, which is the cheapest available check that both
    are right.

    Parameters
    ----------
    imps:
        Gauge-fixed infinite MPS. Call `canonical` first.
    impo:
        Hamiltonian iMPO over the same unit cell.
    tol:
        Convergence threshold on the change in the environment.
    maxiter:
        Iteration budget, counted in unit cells.

    Returns
    -------
    tuple
        `(E, energy)`, the environment with axes `(bra, mpo, ket)` and the
        energy of one unit cell. Running out of cells is reported as a
        warning, not an error, and the best estimate is returned.

    Raises
    ------
    ValueError
        If the unit cells disagree, or if `imps` is not gauge fixed.
    """
    _check_cell(imps, impo)
    isometries = imps.right_isometries()
    last = imps.N - 1

    start, finish = boundary_vectors(impo, 0)
    start.permute([1, 0], in_place=True)

    delta = _capped_identity(
        isometries[last].indices[1], isometries[last].itags[1], Direction.IN,
    )
    density = right_density(imps)
    density.insert_index(1, direction=Direction.OUT, itag=_CAP_ITAG)

    # Read right to left the automaton runs backwards, so the roles of the
    # two boundary channels swap: a right environment enters at finish and
    # diverges at start.
    return _fixed_point(
        seed=einsum('ayb,oy->aob', delta, finish),
        divergent=einsum('ayb,oy->aob', delta, start.conj()),
        projector=start,
        density=density,
        absorb=lambda E: cell_right_env(E, isometries, impo),
        tol=tol,
        maxiter=maxiter,
        side='right',
    )


def energy_density(
    imps: iMPS,
    impo: iMPO,
    *,
    tol: float = 1e-12,
    maxiter: int = 200,
) -> float:
    """Return the energy per site of an infinite MPS.

    A convenience wrapper over `left_environment` for callers that want the
    number and not the environment.

    Parameters
    ----------
    imps:
        Gauge-fixed infinite MPS.
    impo:
        Hamiltonian iMPO over the same unit cell.
    tol:
        Convergence threshold on the change in the environment.
    maxiter:
        Iteration budget, counted in unit cells.

    Returns
    -------
    float
        Energy per site, the energy of one unit cell divided by its size.
    """
    _, energy = left_environment(imps, impo, tol=tol, maxiter=maxiter)
    return energy / imps.N


# ---------------------------------------------------------------------------
# Block storage over the unit cell
# ---------------------------------------------------------------------------

def _save(fp: Path, block: Tensor) -> None:
    """Serialize and write a single environment block to `fp`."""
    torch.save(_serialize_tensor(block), fp)


def _load(fp: Path) -> Tensor:
    """Load and deserialize a single environment block from `fp`."""
    return _deserialize_tensor(torch.load(fp, weights_only=True))


class Environment:
    """Environment blocks for the bonds of a unit cell, optionally on disk.

    Block `k` sits on bond `k` and holds everything on one side of it, so
    a cell of `N` sites has `N` of them and the index wraps: bond `N` is
    bond `0` on the next copy of the cell. This is bond indexing, not site
    indexing, which is what makes the wrap work out.

    A block scales as `chi^2 * w`. On a quasi-1D cylinder the unit cell is
    one column of the lattice, so `N` can be dozens while `chi` and `w`
    both grow with the circumference, and holding all `N` at once is what
    exhausts memory. With `path` set, the class keeps `window` blocks in
    memory and the rest on disk, so the footprint stops depending on `N`.

    The sweep direction is an explicit setting rather than something
    inferred. A cyclic index has no boundary to detect a turn at, and the
    only pass that exists today runs one way, so having the caller say
    which way it is going is both simpler and the hook a backward pass
    would use.

    Parameters
    ----------
    N:
        Number of sites in the unit cell, and so of bonds.
    path:
        Directory for cache files, each block written as `{k:05d}.pt`.
        `None` keeps every block in memory and makes `evict` a no-op,
        since there would be nowhere to read a dropped block back from.
    async_io:
        If `True` (default), reads and writes are submitted to a
        background thread so they overlap with the Davidson solves. No
        effect when `path` is `None`.
    window:
        Number of blocks to keep in memory: the one being read plus
        `window - 1` prefetched ahead. Clamped to `N`. No effect when
        `path` is `None`.
    direction:
        `+1` when the caller reads blocks in increasing bond order, `-1`
        for decreasing. Decides which neighbors are prefetched and which
        block falls out of the window.
    """

    def __init__(
        self,
        N: int,
        path: Optional[Path] = None,
        *,
        async_io: bool = True,
        window: int = 2,
        direction: int = 1,
    ) -> None:
        if N < 1:
            raise ValueError(f"a unit cell needs at least one bond, got N={N}")

        self._blocks: List[Optional[Tensor]] = [None] * N
        self._path: Optional[Path] = path
        self._async_io: bool = async_io and path is not None
        # A window as wide as the cell means nothing is ever dropped, which
        # is also what the modular arithmetic below degenerates to.
        self._window: int = max(1, min(window, N))
        self._direction: int = direction

        self._executor: Optional[ThreadPoolExecutor] = None
        self._write_futures: Dict[int, Future] = {}
        self._read_futures: Dict[int, Future] = {}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def window(self) -> int:
        """Number of blocks the sliding window keeps in memory."""
        return self._window

    @property
    def cached(self) -> bool:
        """Whether blocks are backed by disk and so may be evicted."""
        return self._path is not None

    @property
    def direction(self) -> int:
        """Bond order the caller reads in: `+1` increasing, `-1` decreasing."""
        return self._direction

    @direction.setter
    def direction(self, value: int) -> None:
        if value not in (1, -1):
            raise ValueError(f"direction must be +1 or -1, got {value}")
        self._direction = value

    @property
    def n_resident(self) -> int:
        """Number of blocks currently held in memory."""
        return sum(1 for block in self._blocks if block is not None)

    def is_resident(self, k: int) -> bool:
        """Whether bond `k` is held in memory rather than only on disk."""
        return self._blocks[k] is not None

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        return (
            f"Environment(N={len(self._blocks)}, resident={self.n_resident}, "
            f"cached={self.cached})"
        )

    # ------------------------------------------------------------------
    # Reading and writing
    # ------------------------------------------------------------------

    def __getitem__(self, k: int) -> Tensor:
        """Return the block on bond `k`, without touching the window.

        Resolves from memory, from a prefetch already in flight, or from
        disk, in that order. Use this when reading out of the order the
        window was set up for; use `fetch` to read along it.

        Raises
        ------
        IndexError
            If `k` is out of range.
        RuntimeError
            If the block is neither in memory nor on disk.
        """
        if not (0 <= k < len(self._blocks)):
            raise IndexError(f"bond index {k} out of range for N={len(self._blocks)}")

        block = self._blocks[k]
        if block is not None:
            return block
        if k in self._read_futures:
            block = self._read_futures.pop(k).result()
            self._blocks[k] = block
            return block
        if self._path is not None:
            fp = self._path / f"{k:05d}.pt"
            if fp.exists():
                block = _load(fp)
                self._blocks[k] = block
                return block
        raise RuntimeError(f"environment block on bond {k} has not been initialized")

    def __setitem__(self, k: int, E: Tensor) -> None:
        """Store the block on bond `k` and schedule its write to disk.

        Any prefetch still in flight for this bond is dropped, since the
        block just written supersedes whatever is on disk. A previous write
        to the same slot is waited on first, so two writes never race for
        one file.

        Raises
        ------
        IndexError
            If `k` is out of range.
        """
        if not (0 <= k < len(self._blocks)):
            raise IndexError(f"bond index {k} out of range for N={len(self._blocks)}")

        self._blocks[k] = E
        self._read_futures.pop(k, None)

        if self._path is not None:
            previous = self._write_futures.pop(k, None)
            if previous is not None:
                previous.result()
            self._write_futures[k] = self._submit(_save, self._path / f"{k:05d}.pt", E)

    def fetch(self, k: int) -> Tensor:
        """Return the block on bond `k` and slide the window past it.

        After resolving the block, in this order: the block that has
        dropped off the back of the window is evicted, then the next
        `window - 1` blocks ahead are scheduled for prefetch. Both indices
        wrap around the cell. Doing the eviction first keeps the two
        disjoint, so a block is never dropped and immediately asked for
        again.

        A window as wide as the cell evicts nothing, since the block that
        would fall out is the one just read.

        Parameters
        ----------
        k:
            Bond index.

        Returns
        -------
        Tensor
            The block on bond `k`.
        """
        block = self[k]

        if self._path is not None:
            N = len(self._blocks)
            stale = (k - self._direction * self._window) % N
            if stale != k:
                self.evict(stale)
            for step in range(1, self._window):
                self._prefetch((k + self._direction * step) % N)

        return block

    def evict(self, k: int) -> None:
        """Drop the in-memory copy of bond `k`, leaving the disk copy.

        A no-op when the blocks are not disk-backed, since there would be
        nowhere to read the block back from, and when `k` holds nothing.
        Waits for a write still in flight, so the block is safely on disk
        before memory lets go of it.
        """
        if self._path is None:
            return
        if not (0 <= k < len(self._blocks)):
            return
        pending = self._write_futures.pop(k, None)
        if pending is not None:
            pending.result()
        self._blocks[k] = None

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Flush pending I/O and release the background thread.

        Waits on every outstanding write, drains the prefetches, and closes
        the executor. A no-op in substance when `path` is `None`.
        """
        for future in self._write_futures.values():
            future.result()
        self._write_futures.clear()
        for future in self._read_futures.values():
            try:
                future.result()
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
        """Return the shared executor, created on first use."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        return self._executor

    def _submit(self, fn: Callable, *args) -> Future:
        """Run `fn` on the executor, or now if asynchronous I/O is off."""
        if self._async_io:
            return self._get_executor().submit(fn, *args)
        future: Future = Future()
        future.set_result(fn(*args))
        return future

    def _prefetch(self, k: int) -> None:
        """Schedule an asynchronous load of bond `k`.

        Skipped when the block is already in memory, a read is already in
        flight, or nothing has been written to disk for it yet.
        """
        if self._path is None:
            return
        if self._blocks[k] is not None:
            return
        if k in self._read_futures:
            return
        fp = self._path / f"{k:05d}.pt"
        if not fp.exists():
            return
        self._read_futures[k] = self._submit(_load, fp)
