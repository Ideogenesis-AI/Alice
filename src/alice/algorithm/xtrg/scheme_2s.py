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


"""2-site variational MPO-MPO compression update.

The 2-site update optimises a two-site bond tensor

    Θ[c, d, r, s, u, v]

representing the product of `C_i` and `C_{i+1}` (in the canonical frame of
C) and splits it via SVD to obtain a new pair `(C_i, C_{i+1})` with a
controlled bond dimension.

Index convention (shared with `environ.py`):

- `c` — C left bond (left boundary of site i)
- `d` — C right bond (right boundary of site i+1)
- `r` — phys_in  of site i
- `s` — phys_out of site i
- `u` — phys_in  of site i+1
- `v` — phys_out of site i+1

Θ axes: `(c=C_left_i, d=C_right_{i+1}, r=phys_in_i, s=phys_out_i,
          u=phys_in_{i+1}, v=phys_out_{i+1})`

The contraction order in `local_update_2s` uses two intermediate partial
contractions to avoid the O(χ⁵) bottleneck of a naive 6-tensor einsum:

    L = einsum('cab, aprx, bqxs -> cpqrs', E_left, A_i, B_i)   # O(χ³ d³)
    R = einsum('def, peuy, qfyv -> dpquv', E_right, A_j, B_j)  # O(χ³ d³)
    Θ = einsum('cpqrs, dpquv -> cdrsuv', L, R)                  # O(χ⁴ d⁴), bottleneck
"""

from __future__ import annotations

from typing import Optional, Tuple

from nicole import Tensor, decomp, einsum, merge_axes
from nicole.decomp import svd


def local_update_2s(
    E_left: Tensor,
    A_i: Tensor,
    B_i: Tensor,
    A_j: Tensor,
    B_j: Tensor,
    E_right: Tensor,
) -> Tensor:
    """Compute the optimal 2-site bond tensor Θ for sites (i, i+1).

    Builds the 2-site right-hand side of the Frobenius-norm optimisation via
    two partial contractions followed by a join, which avoids intermediate
    tensors of size O(χ⁵):

        L = einsum('cab, aprx, bqxs -> cpqrs', E_left, A_i, B_i)
        R = einsum('def, peuy, qfyv -> dpquv', E_right, A_j, B_j)
        Θ = einsum('cpqrs, dpquv -> cdrsuv', L, R)

    Parameters
    ----------
    E_left:
        Left environment at boundary i with axes `(c, a, b)`.
    A_i:
        Factor MPO A at site i with axes `(a, p, r, x)`.
    B_i:
        Factor MPO B at site i with axes `(b, q, x, s)`.
    A_j:
        Factor MPO A at site j=i+1 with axes `(p, e, u, y)`.
    B_j:
        Factor MPO B at site j=i+1 with axes `(q, f, y, v)`.
    E_right:
        Right environment at boundary i+2 with axes `(d, e, f)`.

    Returns
    -------
    Tensor
        Bond tensor Θ with axes `(c, d, r, s, u, v)`.
    """
    # Left partial: E_left(c,a,b) × A_i(a,p,r,x) × B_i(b,q,x,s) → L(c,p,q,r,s).
    # Decomposed into two 2-tensor einsums to avoid itag-matching ambiguity.
    t1 = einsum('cab,aprx->cbprx', E_left, A_i)         # contract 'a'
    L = einsum('cbprx,bqxs->cpqrs', t1, B_i)             # contract 'b','x'

    # Right partial: E_right(d,e,f) × A_j(p,e,u,y) × B_j(q,f,y,v) → R(d,p,q,u,v).
    t2 = einsum('def,peuy->dfpuy', E_right, A_j)          # contract 'e'
    R = einsum('dfpuy,qfyv->dpquv', t2, B_j)              # contract 'f','y'

    # Join: contract L(c,p,q,r,s) × R(d,p,q,u,v) over 'p','q' → Θ(c,d,r,s,u,v).
    return einsum('cpqrs,dpquv->cdrsuv', L, R)


def split_forward(
    theta: Tensor,
    itag: str,
    trunc: Optional[dict],
) -> Tuple[Tensor, Tensor]:
    """SVD-split Θ for a forward (left-to-right) sweep step.

    Decomposes Θ as `C_i · C_{i+1}` where `C_i` is left-isometric and
    `C_{i+1}` carries the singular values. The bipartition is
    `(c, r, s) | (d, u, v)`.

    Parameters
    ----------
    theta:
        Bond tensor with axes `(c, d, r, s, u, v)`.
    itag:
        itag for the new internal bond between the two output tensors.
    trunc:
        Truncation options forwarded to `decomp` (`nkeep`, `thresh`).
        Pass `None` for no truncation.

    Returns
    -------
    Tensor
        Left-isometric `C_i` with axes `(c, new_bond, r, s)`.
    Tensor
        `C_{i+1}` (carries singular values) with axes `(new_bond, d, u, v)`.
    """
    # Decompose with axes [0, 2, 3] = (c, r, s) on the U side.
    # mode='UR' → U is left-isometric, R = S·Vh carries the singular values.
    # U unmerged axes: (c, r, s, new_bond).
    # R axes: (new_bond, d, u, v).
    U, R = decomp(theta, axes=[0, 2, 3], mode='UR', trunc=trunc)
    # Retag the new bond in both tensors.
    U.retag(3, itag)
    R.retag(0, itag)
    # Permute U: (c, r, s, new_bond) → (c, new_bond, r, s) = MPO convention.
    U.permute([0, 3, 1, 2], in_place=True)
    return U, R


def split_backward(
    theta: Tensor,
    itag: str,
    trunc: Optional[dict],
) -> Tuple[Tensor, Tensor]:
    """SVD-split Θ for a backward (right-to-left) sweep step.

    Decomposes Θ as `C_i · C_{i+1}` where `C_{i+1}` is right-isometric and
    `C_i` carries the singular values. The bipartition is `(c, r, s) | (d, u, v)`.

    Parameters
    ----------
    theta:
        Bond tensor with axes `(c, d, r, s, u, v)`.
    itag:
        itag for the new internal bond between the two output tensors.
    trunc:
        Truncation options forwarded to `decomp` (`nkeep`, `thresh`).
        Pass `None` for no truncation.

    Returns
    -------
    Tensor
        `C_i` (carries singular values) with axes `(c, new_bond, r, s)`.
    Tensor
        Right-isometric `C_{i+1}` with axes `(new_bond, d, u, v)`.
    """
    # mode='LV' → L = U·S carries the singular values, V is right-isometric.
    # L unmerged axes: (c, r, s, new_bond).
    # V axes: (new_bond, d, u, v).
    L, V = decomp(theta, axes=[0, 2, 3], mode='LV', trunc=trunc)
    L.retag(3, itag)
    V.retag(0, itag)
    # Permute L: (c, r, s, new_bond) → (c, new_bond, r, s) = MPO convention.
    L.permute([0, 3, 1, 2], in_place=True)
    return L, V


def discarded_weight(theta: Tensor, trunc: Optional[dict]) -> float:
    """Compute the discarded weight for an SVD split of Θ under `trunc`.

    Merges axes `[0, 2, 3]` of Θ (the same bipartition used by `split_forward`
    and `split_backward`) and performs an SVD with `requires_info=True` to
    read back `info["discarded_weight"]`.

    Parameters
    ----------
    theta:
        Bond tensor with axes `(c, d, r, s, u, v)`.
    trunc:
        Truncation options (`nkeep`, `thresh`). Pass `None` for no truncation
        (returns `0.0`).

    Returns
    -------
    float
        Sum of squared singular values discarded by `trunc`. Zero when `trunc`
        is `None` or nothing is cut.
    """
    if trunc is None:
        return 0.0
    merged, _ = merge_axes(theta, [0, 2, 3], merged_tag='_dw_merged_')
    *_, info = svd(merged, axis=0, trunc=trunc, requires_info=True)
    return float(info['discarded_weight'])
