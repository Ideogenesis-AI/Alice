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


"""Two-site BUG bond update and odd/even parity sweeps.

A single bond update is the rank-adaptive basis-update-and-Galerkin step: bring
the orthogonality center onto the active bond (an exact, truncation-free move),
contract the two neighbouring MPS tensors into a two-site block, apply the bond
gate, and split the block back with a truncated SVD that adapts the bond
dimension.

The chain Hamiltonian splits into two commuting groups — even bonds (left-site
index 0, 2, 4, …) and odd bonds (1, 3, 5, …). Gates within one group act on
disjoint site pairs, so a parity sweep applies them exactly; the Trotter error
lives only between the two groups. This is the same even/odd BUG sweep used for
the domain-wall XX chain.

Index conventions match `alice.network`: a two-site block has axes
`(left, right, phys_i, phys_{i+1})` and an MPS site tensor has axes
`(left, right, phys)`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from nicole import Tensor, decomp, einsum

from alice.network import MPS

from .gate import apply_bond_gate, retag_gate_for_bond


def _build_theta(m_i: Tensor, m_i1: Tensor) -> Tensor:
    """Contract two neighbouring MPS tensors into a two-site block.

    Parameters
    ----------
    m_i:
        Site tensor at *i* with axes `(left, right, phys)`.
    m_i1:
        Site tensor at *i+1* with axes `(left, right, phys)`; its left bond
        shares the itag of `m_i`'s right bond.

    Returns
    -------
    Tensor
        Two-site block with axes `(left, right, phys_i, phys_{i+1})`.
    """
    return einsum('abr,bcs->acrs', m_i, m_i1)


def _split_bond(theta: Tensor, itag: str, trunc: Optional[dict]) -> Tuple[Tensor, Tensor]:
    """Truncated-SVD split of a two-site block into two MPS tensors.

    Decomposes `theta` across the `(left, phys_i)` vs `(right, phys_{i+1})`
    bipartition. The left tensor is left-isometric and the right tensor carries
    the singular values, leaving the orthogonality center on the right site. The
    kept bond dimension is set by `trunc`, giving the rank adaptation.

    Parameters
    ----------
    theta:
        Two-site block with axes `(left, right, phys_i, phys_{i+1})`.
    itag:
        itag assigned to the new internal bond.
    trunc:
        Truncation options forwarded to `decomp` (`nkeep`, `thresh`), or `None`.

    Returns
    -------
    Tensor
        Left-isometric tensor with axes `(left, bond, phys_i)`.
    Tensor
        Right tensor (carrying singular values) with axes
        `(bond, right, phys_{i+1})`.
    """
    left, right = decomp(theta, axes=[0, 2], mode='UR', trunc=trunc)
    left.retag(2, itag)
    right.retag(0, itag)
    # decomp returns the left factor as (left, phys_i, bond); reorder to MPS layout.
    left.permute([0, 2, 1], in_place=True)
    return left, right


def _augmented_dim(theta: Tensor) -> int:
    """Return the proposed (pre-truncation) bond dimension of a two-site block.

    This is the dimension of the smaller side of the `(left, phys_i)` vs
    `(right, phys_{i+1})` bipartition — the augmented working space the BUG step
    proposes before the truncated split discards the negligible directions.
    Since the physical dimension is `d`, it is roughly `d` times the incoming
    bond dimension, i.e. the basis augmentation of the step.

    Parameters
    ----------
    theta:
        Two-site block with axes `(left, right, phys_i, phys_{i+1})`.

    Returns
    -------
    int
        Proposed augmented bond dimension at this bond.
    """
    left, right, phys_i, phys_j = theta.indices
    return min(left.dim * phys_i.dim, right.dim * phys_j.dim)


def gate_bond(mps: MPS, i: int, gate: Tensor, trunc: Optional[dict]) -> int:
    """Apply one bond gate to sites *(i, i+1)* of `mps`, in place.

    Moves the orthogonality center onto site *i* without truncation, contracts
    the two-site block, applies the gate, and splits the new block with truncation.
    Performing the center move truncation-free keeps the split — and only the
    split — responsible for the rank adaptation. After the call
    `mps.center == i + 1`.

    Parameters
    ----------
    mps:
        State to update in place.
    i:
        Left site of the bond; the gate acts on sites *i* and *i+1*.
    gate:
        Two-site gate from :func:`alice.algorithm.two_site_bug.gate.exp_bond_gate`.
    trunc:
        Truncation options forwarded to the SVD split.

    Returns
    -------
    int
        Proposed augmented bond dimension at this bond, before truncation
        (see :func:`_augmented_dim`).
    """
    mps.canonical(i, trunc=None)
    phys_itags = (mps[i].itags[2], mps[i + 1].itags[2])
    theta = _build_theta(mps[i], mps[i + 1])
    theta = apply_bond_gate(theta, retag_gate_for_bond(gate, phys_itags))
    augmented = _augmented_dim(theta)
    mps[i], mps[i + 1] = _split_bond(theta, mps._bond_itag(i + 1), trunc)
    mps._center = i + 1
    return augmented


def parity_bonds(length: int, parity: str) -> List[int]:
    """Return the left-site indices of all bonds in one commuting group.

    Parameters
    ----------
    length:
        Number of sites `L`.
    parity:
        `'even'` for bonds with an even left-site index (0, 2, 4, …) or `'odd'`
        for bonds with an odd left-site index (1, 3, 5, …).

    Returns
    -------
    list of int
        Left-site indices of the bonds in the requested group, in increasing
        order.

    Raises
    ------
    ValueError
        If `parity` is not `'even'` or `'odd'`.
    """
    if parity == 'even':
        return list(range(0, length - 1, 2))
    if parity == 'odd':
        return list(range(1, length - 1, 2))
    raise ValueError(f"parity must be 'even' or 'odd', got {parity!r}")


def parity_sweep(
    mps: MPS,
    gates: List[Optional[Tensor]],
    parity: str,
    trunc: Optional[dict],
) -> int:
    """Apply every bond gate of one commuting group to `mps`, in place.

    Bonds of the chosen parity act on disjoint site pairs, so the group is an
    exact factor of the Trotter step. Bonds whose gate is `None` (no Hamiltonian
    term) are skipped.

    Parameters
    ----------
    mps:
        State to update in place.
    gates:
        Per-bond gates of length `L - 1`; entry *b* acts on bond *(b, b+1)*.
    parity:
        `'even'` or `'odd'` — selects the commuting bond group.
    trunc:
        Truncation options forwarded to each bond split.

    Returns
    -------
    int
        Largest proposed augmented bond dimension over the bonds of this group
        (0 if the group has no active bonds).
    """
    augmented = 0
    for i in parity_bonds(mps.L, parity):
        if gates[i] is not None:
            augmented = max(augmented, gate_bond(mps, i, gates[i], trunc))
    return augmented
