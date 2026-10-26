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


"""Unit-cell transfer operators of an infinite MPS and MPO.

An infinite chain has no boundary to start an environment from, so the
boundary environments of a unit cell are fixed points of the transfer
operator that carries one whole cell. This module builds those operators
and carries a tensor through them.

Index convention, as in `alice.algorithm.dmrg.environ`:

- `a, b, c, d` — virtual bond indices of the MPS, bra or ket side
- `o, p, q`    — virtual bond indices of the MPO
- `r, s`       — physical indices

The left environment `E(a, o, b)` sits on bond `0` in the `AL` basis and
accumulates every site to its left; the right environment `E(c, p, d)` sits
on the same bond in the `AR` basis and accumulates every site to its right.
"""

from __future__ import annotations

import logging
from typing import Sequence

from nicole import Tensor, einsum

from alice.algorithm.dmrg.environ import step_left_env, step_right_env
from alice.network import iMPO, iMPS

logger = logging.getLogger(__name__)

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

    Applies `step_left_env` once per site. The wrap-around itags of the
    infinite containers mean the environment comes back wearing the names it
    started with, so the map can be iterated without any retagging.

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
