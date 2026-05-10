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


"""2-site DMRG update scheme.

This module contains all operations specific to the 2-site update:

- `build_bulk`: contract two adjacent site tensors into a 4-index bond tensor Θ,
  used as the initial guess for the Davidson solver.
- `matvec_2s`: apply the 2-site effective Hamiltonian
  `H_eff = E_left ⊗ W_i ⊗ W_{i+1} ⊗ E_right` to Θ.
- `split_forward` / `split_backward`: SVD-split the optimised Θ back into two
  site tensors, replacing the `mps.canonical` call used in 1-site DMRG.
- `optimize_2site`: pure function that runs the Davidson solver on Θ.

Index convention (shared with `environ.py` and `scheme_1s.py`)
--------------------------------------------------------------
- `a, b, c`     — virtual bond indices of the MPS (bra or ket side)
- `o, p, q`     — virtual bond indices of the MPO
- `r, s`        — physical indices of site i  (bra r, ket s)
- `u, v`        — physical indices of site i+1 (bra u, ket v)

Bond tensor Θ axes: `(a=ket_left_i, c=ket_right_{i+1}, s=phys_ket_i, v=phys_ket_{i+1})`

2-site effective Hamiltonian layout:

    E_left [a, o, b]    — (bra_left,  mpo_left,   ket_left)
    Θ      [b, d, s, v] — (ket_left,  ket_right,  phys_ket_i, phys_ket_{i+1})
    W_i    [o, p, r, s] — (mpo_left,  mpo_right,  phys_bra_i, phys_ket_i)
    W_{i+1}[p, q, u, v] — (mpo_right, mpo_right2, phys_bra_{i+1}, phys_ket_{i+1})
    E_right[c, q, d]    — (bra_right, mpo_right2, ket_right)
    output [a, c, r, u] — (bra_left, bra_right, phys_bra_i, phys_bra_{i+1})
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

from nicole import Tensor, decomp, einsum, merge_axes
from nicole.decomp import svd

from .davidson import davidson


def build_bulk(M_i: Tensor, M_i1: Tensor) -> Tensor:
    """Contract two adjacent MPS tensors into a 2-site bond tensor Θ.

    Θ is used as the initial guess for the Davidson solver in `optimize_2site`.
    After Davidson converges, the optimised Θ is split back into two site
    tensors via `split_forward` or `split_backward`.

    Parameters
    ----------
    M_i:
        MPS tensor at site i with axes `(ket_left, ket_right, phys_ket)`.
    M_i1:
        MPS tensor at site i+1 with axes `(ket_left, ket_right, phys_ket)`.
        Its left bond must share the itag of `M_i`'s right bond.

    Returns
    -------
    Tensor
        Bond tensor Θ with axes
        `(ket_left_i, ket_right_{i+1}, phys_ket_i, phys_ket_{i+1})`.
    """
    # Contract over the shared internal bond b (right bond of M_i = left bond of M_i1).
    # 'abr,bcs->acrs': a=ket_left_i, b=internal (contracted), c=ket_right_{i+1},
    #                   r=phys_ket_i, s=phys_ket_{i+1}.
    return einsum('abr,bcs->acrs', M_i, M_i1)


def matvec_2s(
    theta: Tensor,
    W_i: Tensor,
    W_i1: Tensor,
    E_left: Tensor,
    E_right: Tensor,
) -> Tensor:
    """Apply the 2-site effective Hamiltonian H_eff to the bond tensor Θ.

    Computes `H_eff|Θ⟩` as a sequence of pairwise contractions in left-to-right
    order, keeping intermediate sizes as small as possible for a generic 4-index
    theta. Nicole's `einsum` contracts multi-tensor equations strictly left to
    right, so the order below is exact:

    1. `E_left(a,o,b) × theta(b,d,s,v)` over b → `(a,o,d,s,v)`
    2. `× W_i(o,p,r,s)` over `(o,s)` → `(a,p,d,r,v)`
    3. `× W_{i+1}(p,q,u,v)` over `(p,v)` → `(a,q,d,r,u)`
    4. `× E_right(c,q,d)` over `(q,d)` → `(a,c,r,u)`

    Parameters
    ----------
    theta:
        Bond tensor with axes `(ket_left, ket_right, phys_ket_i, phys_ket_{i+1})`.
    W_i:
        MPO tensor at site i with axes `(mpo_left, mpo_right, phys_bra_i, phys_ket_i)`.
    W_i1:
        MPO tensor at site i+1 with axes
        `(mpo_left, mpo_right, phys_bra_{i+1}, phys_ket_{i+1})`.
    E_left:
        Left environment with axes `(bra_left, mpo_left, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_right, ket_right)`.

    Returns
    -------
    Tensor
        `H_eff|Θ⟩` with axes `(bra_left, bra_right, phys_bra_i, phys_bra_{i+1})`,
        identical in shape to Θ.
    """
    # Contract left to right: E_left × theta × W_i × W_{i+1} × E_right.
    # Nicole's einsum performs multi-tensor equations strictly left to right,
    # so this single call is identical to the four sequential pairwise calls.
    return einsum('aob,bdsv,oprs,pquv,cqd->acru', E_left, theta, W_i, W_i1, E_right)


def split_forward(
    theta: Tensor,
    itag: str,
    trunc: Optional[dict],
) -> Tuple[Tensor, Tensor]:
    """SVD-split Θ for a forward (left-to-right) sweep step.

    Decomposes Θ as `M_i · M_{i+1}` where `M_i` is left-isometric and
    `M_{i+1}` carries the singular values. The bond dimension of the new
    internal index is controlled by `trunc`.

    This replaces the `mps.canonical(i + 1)` call used in 1-site DMRG.

    Parameters
    ----------
    theta:
        Bond tensor with axes `(ket_left, ket_right, phys_i, phys_{i+1})`.
    itag:
        itag for the new internal bond between the two output tensors.
    trunc:
        Truncation options forwarded to `decomp` (`nkeep`, `thresh`).
        Pass `None` for no truncation.

    Returns
    -------
    Tensor
        Left-isometric `M_i` with axes `(ket_left, new_bond, phys_i)`.
    Tensor
        `M_{i+1}` (carries singular values) with axes
        `(new_bond, ket_right, phys_{i+1})`.
    """
    # Decompose with axes [0, 2] (ket_left, phys_i) on the U side.
    # mode='UR' → U is left-isometric, R = S·Vh carries the singular values.
    # U_unmerged axes after unmerging: (ket_left, phys_i, new_bond).
    # R axes: (new_bond, ket_right, phys_{i+1}).
    U, R = decomp(theta, axes=[0, 2], mode='UR', trunc=trunc)
    # Retag the new bond in both tensors to the caller-supplied itag.
    U.retag(2, itag)
    R.retag(0, itag)
    # Permute U from (ket_left, phys_i, new_bond) → (ket_left, new_bond, phys_i)
    # to match the MPS axis convention (left, right, phys).
    U.permute([0, 2, 1], in_place=True)

    return U, R


def split_backward(
    theta: Tensor,
    itag: str,
    trunc: Optional[dict],
) -> Tuple[Tensor, Tensor]:
    """SVD-split Θ for a backward (right-to-left) sweep step.

    Decomposes Θ as `M_i · M_{i+1}` where `M_{i+1}` is right-isometric and
    `M_i` carries the singular values.

    This replaces the `mps.canonical(i - 1)` call used in 1-site DMRG.

    Parameters
    ----------
    theta:
        Bond tensor with axes `(ket_left, ket_right, phys_i, phys_{i+1})`.
    itag:
        itag for the new internal bond between the two output tensors.
    trunc:
        Truncation options forwarded to `decomp` (`nkeep`, `thresh`).
        Pass `None` for no truncation.

    Returns
    -------
    Tensor
        `M_i` (carries singular values) with axes `(ket_left, new_bond, phys_i)`.
    Tensor
        Right-isometric `M_{i+1}` with axes `(new_bond, ket_right, phys_{i+1})`.
    """
    # mode='LV' → L = U·S carries the singular values, V is right-isometric.
    # L_unmerged axes: (ket_left, phys_i, new_bond).
    # V axes: (new_bond, ket_right, phys_{i+1}).
    L, V = decomp(theta, axes=[0, 2], mode='LV', trunc=trunc)
    L.retag(2, itag)
    V.retag(0, itag)
    # Permute L from (ket_left, phys_i, new_bond) → (ket_left, new_bond, phys_i).
    L.permute([0, 2, 1], in_place=True)

    return L, V


def discarded_weight(theta: Tensor, trunc: Optional[dict]) -> float:
    """Compute the discarded weight for an SVD split of Θ under `trunc`.

    Merges axes 0 and 2 of `theta` (the same bipartition used by `split_forward`
    and `split_backward`) and performs a second SVD with `requires_info=True` to
    read back `info["discarded_weight"]`. This is an intentional second SVD;
    it is called only once per full sweep at the center bond, so the overhead is
    negligible relative to the Davidson optimisation cost.

    Parameters
    ----------
    theta:
        Bond tensor with axes `(ket_left, ket_right, phys_i, phys_{i+1})`.
    trunc:
        Truncation options (`nkeep`, `thresh`) — must match what is passed to
        `split_forward` / `split_backward` so the discarded weight is consistent.

    Returns
    -------
    float
        Sum of all singular values discarded by `trunc` across all charge
        sectors. Zero when `trunc` is `None` or nothing is cut.
    """
    merged, _ = merge_axes(theta, [0, 2], merged_tag='_dw_merged_')
    *_, info = svd(merged, axis=0, trunc=trunc, requires_info=True)

    return float(info['discarded_weight'])


def optimize_2site(
    M_i: Tensor,
    M_i1: Tensor,
    W_i: Tensor,
    W_i1: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    davidson_opts: dict,
) -> Tuple[float, Tensor, float]:
    """Find the optimal 2-site bond tensor via the Davidson eigensolver.

    Forms the bond tensor Θ = M_i ⊗ M_{i+1} as the initial guess, then runs
    Davidson to minimise the 2-site Rayleigh quotient. Returns the optimised
    Θ ready to be split by `split_forward` or `split_backward`.

    This is a pure function: it takes tensors and returns tensors without
    mutating any MPS state.

    Parameters
    ----------
    M_i:
        MPS tensor at site i with axes `(ket_left, ket_right, phys_ket)`.
    M_i1:
        MPS tensor at site i+1 with axes `(ket_left, ket_right, phys_ket)`.
    W_i:
        MPO tensor at site i with axes `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    W_i1:
        MPO tensor at site i+1 with axes `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    E_left:
        Left environment with axes `(bra_left, mpo_left, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_right, ket_right)`.
    davidson_opts:
        Keyword arguments forwarded to `davidson`: `max_iter`, `tol`,
        `max_subspace`.

    Returns
    -------
    float
        Variational energy estimate (lowest Ritz value).
    Tensor
        Optimised bond tensor `theta_opt` with axes
        `(ket_left, ket_right, phys_ket_i, phys_ket_{i+1})`.
    float
        Final Davidson residual norm at convergence (or at exit if not converged).
    """
    theta0 = build_bulk(M_i, M_i1)
    mv = partial(matvec_2s, E_left=E_left, W_i=W_i, W_i1=W_i1, E_right=E_right)
    energy, theta_opt, davidson_error = davidson(mv, theta0, **davidson_opts)

    return energy, theta_opt, davidson_error
