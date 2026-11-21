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


"""2-site TgTRG update scheme.

This module contains the operations specific to the 2-site update:

- `build_bulk`: contract two adjacent density-matrix tensors into a 6th-order
  bond tensor Θ.
- `matvec_2s` / `evolve_2site`: apply `exp(-tau·H_eff)` to Θ, where
  `H_eff = E_left ⊗ W_i ⊗ W_{i+1} ⊗ E_right`.
- `split_forward` / `split_backward`: SVD-split the cooled Θ back into two
  site tensors, truncating the new bond under `trunc`.
- `discarded_weight`: diagnose how much the truncation threw away.

The 2-site scheme has no zero-site step. Its back-evolution is a *1-site*
evolution of the site the split just loaded with singular values, so this
module imports nothing of its own for that purpose — the sweep calls
`scheme_1s.evolve_1site` with a negated increment.

This is the scheme a cooling run normally wants. The initial state is
`ρ(τ₀)` from a short Taylor series, whose bond dimension is that of the
Hamiltonian and far below what a low-temperature density matrix needs; only
the 2-site update can grow the bond on the way down. The price is a
truncation error the 1-site scheme does not incur.

Index convention (shared with `environ.py` and `scheme_1s.py`)
--------------------------------------------------------------
- `a, b, c, d`  — virtual bond indices of ρ (bra or ket side)
- `o, p, q`     — virtual bond indices of the Hamiltonian MPO
- `r, x`        — `phys_in` of site i, bra and ket
- `s`           — `phys_out` of site i, a spectator
- `u, y`        — `phys_in` of site i+1, bra and ket
- `v`           — `phys_out` of site i+1, a spectator

Bond tensor Θ axes:
`(ket_left_i, ket_right_{i+1}, phys_in_i, phys_out_i, phys_in_{i+1},
phys_out_{i+1})`, written `(b, d, x, s, y, v)` below.

2-site effective Hamiltonian layout:

    E_left [a, o, b]          — (bra_left,  mpo_left,   ket_left)
    Θ      [b, d, x, s, y, v] — the bond tensor above
    W_i    [o, p, r, x]       — (mpo_left,  mpo_right,  phys_bra_i, phys_ket_i)
    W_{i+1}[p, q, u, y]       — (mpo_right, mpo_right2, phys_bra, phys_ket)
    E_right[c, q, d]          — (bra_right, mpo_right2, ket_right)
    output [a, c, r, s, u, v] — same shape as Θ

Both `phys_out` axes pass through untouched: the Hamiltonian multiplies ρ
from one side only, so half the doubled physical space is a spectator. That,
and the resulting 6th-order Θ, is the only structural difference from the
2-site TDVP scheme.

As in `environ.py`, ρ and the Hamiltonian share the `W{i}` bond itag family,
so every contraction is a chain of explicit two-tensor `einsum` calls.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

from nicole import Tensor, decomp, einsum, merge_axes
from nicole.decomp import svd

from .krylov import lanczos_expv


def build_bulk(M_i: Tensor, M_i1: Tensor) -> Tensor:
    """Contract two adjacent density-matrix tensors into a bond tensor Θ.

    Θ is the object the 2-site effective Hamiltonian acts on. After cooling
    it is split back into two site tensors via `split_forward` or
    `split_backward`.

    Parameters
    ----------
    M_i:
        Density-matrix tensor at site i with axes
        `(ket_left, ket_right, phys_in, phys_out)`.
    M_i1:
        Density-matrix tensor at site i+1, same layout. Its left bond must
        share the itag of `M_i`'s right bond.

    Returns
    -------
    Tensor
        Bond tensor Θ with axes `(ket_left_i, ket_right_{i+1}, phys_in_i,
        phys_out_i, phys_in_{i+1}, phys_out_{i+1})`.
    """
    # Contract over the shared internal bond (right bond of M_i, left bond
    # of M_i1).
    return einsum('abxs,bdyv->adxsyv', M_i, M_i1)


def matvec_2s(
    theta: Tensor,
    W_i: Tensor,
    W_i1: Tensor,
    E_left: Tensor,
    E_right: Tensor,
) -> Tensor:
    """Apply the 2-site effective Hamiltonian H_eff to the bond tensor Θ.

    Parameters
    ----------
    theta:
        Bond tensor with the axis layout `build_bulk` returns.
    W_i:
        Hamiltonian tensor at site i with axes
        `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    W_i1:
        Hamiltonian tensor at site i+1, same layout.
    E_left:
        Left environment with axes `(bra_left, mpo_left, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_right2, ket_right)`.

    Returns
    -------
    Tensor
        `H_eff|Θ⟩`, identical in shape to Θ.
    """
    # Eliminate the virtual bonds first, one environment at a time, so no
    # intermediate carries both ρ bonds and both MPO bonds at once.
    # Step 1: E_left(a,o,b) × Θ(b,d,x,s,y,v) → (a,o,d,x,s,y,v), over 'b'.
    t1 = einsum('aob,bdxsyv->aodxsyv', E_left, theta)
    # Step 2: × W_i(o,p,r,x) → (a,d,s,y,v,p,r), over 'o','x'.
    t2 = einsum('aodxsyv,oprx->adsyvpr', t1, W_i)
    # Step 3: × W_{i+1}(p,q,u,y) → (a,d,s,v,r,q,u), over 'p','y'.
    t3 = einsum('adsyvpr,pquy->adsvrqu', t2, W_i1)
    # Step 4: × E_right(c,q,d) → (a,c,r,s,u,v), over 'd','q'.
    return einsum('adsvrqu,cqd->acrsuv', t3, E_right)


def evolve_2site(
    M_i: Tensor,
    M_i1: Tensor,
    W_i: Tensor,
    W_i1: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    tau: float,
    krylov_opts: dict,
) -> Tuple[Tensor, float, float]:
    """Cool two adjacent sites under the 2-site effective Hamiltonian.

    Forms Θ from the two site tensors and applies `exp(-tau·H_eff)` to it by
    Lanczos. The caller splits the returned Θ. This is a pure function: it
    takes tensors and returns tensors without mutating any density matrix.

    Parameters
    ----------
    M_i:
        Density-matrix tensor at site i with axes
        `(ket_left, ket_right, phys_in, phys_out)`.
    M_i1:
        Density-matrix tensor at site i+1, same layout.
    W_i:
        Hamiltonian tensor at site i.
    W_i1:
        Hamiltonian tensor at site i+1.
    E_left:
        Left environment with axes `(bra_left, mpo_left, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_right2, ket_right)`.
    tau:
        Imaginary-time increment. Pass a negative value to heat.
    krylov_opts:
        Keyword arguments forwarded to `lanczos_expv`: `max_krylov`, `tol`.

    Returns
    -------
    Tensor
        Cooled bond tensor Θ, with the extremal-eigenvalue factor divided
        out.
    float
        `log` of the factor divided out, for the caller to accumulate.
    float
        Krylov truncation error estimate for this step.
    """
    theta = build_bulk(M_i, M_i1)
    mv = partial(matvec_2s, W_i=W_i, W_i1=W_i1, E_left=E_left, E_right=E_right)

    return lanczos_expv(mv, theta, tau, **krylov_opts)


def split_forward(
    theta: Tensor,
    itag: str,
    trunc: Optional[dict],
) -> Tuple[Tensor, Tensor]:
    """SVD-split Θ for a forward (left-to-right) sweep step.

    Decomposes Θ as `M_i · M_{i+1}` with `M_i` left-isometric and `M_{i+1}`
    carrying the singular values. The 1-site back-evolution then acts on
    `M_{i+1}`.

    Parameters
    ----------
    theta:
        Bond tensor with the axis layout `build_bulk` returns.
    itag:
        itag for the new internal bond between the two output tensors.
    trunc:
        Truncation options forwarded to `decomp` (`nkeep`, `thresh`). Pass
        `None` for no truncation.

    Returns
    -------
    Tensor
        Left-isometric `M_i` with axes `(ket_left, new_bond, phys_in,
        phys_out)`.
    Tensor
        `M_{i+1}` carrying the singular values, with axes `(new_bond,
        ket_right, phys_in, phys_out)`.
    """
    # The bipartition keeps site i's two physical axes on the left.
    # mode='UR' → U is left-isometric, R = S·Vh carries the singular values.
    # U unmerged axes: (ket_left, phys_in, phys_out, new_bond).
    U, R = decomp(theta, axes=[0, 2, 3], mode='UR', trunc=trunc)
    U.retag(3, itag)
    R.retag(0, itag)
    # (ket_left, phys_in, phys_out, new_bond) → MPO axis order.
    U.permute([0, 3, 1, 2], in_place=True)

    return U, R


def split_backward(
    theta: Tensor,
    itag: str,
    trunc: Optional[dict],
) -> Tuple[Tensor, Tensor]:
    """SVD-split Θ for a backward (right-to-left) sweep step.

    Mirror image of `split_forward`: `M_{i+1}` comes out right-isometric and
    `M_i` carries the singular values, so the 1-site back-evolution acts on
    `M_i`.

    Parameters
    ----------
    theta:
        Bond tensor with the axis layout `build_bulk` returns.
    itag:
        itag for the new internal bond between the two output tensors.
    trunc:
        Truncation options forwarded to `decomp` (`nkeep`, `thresh`). Pass
        `None` for no truncation.

    Returns
    -------
    Tensor
        `M_i` carrying the singular values, with axes `(ket_left, new_bond,
        phys_in, phys_out)`.
    Tensor
        Right-isometric `M_{i+1}` with axes `(new_bond, ket_right, phys_in,
        phys_out)`.
    """
    # mode='LV' → L = U·S carries the singular values, V is right-isometric.
    L, V = decomp(theta, axes=[0, 2, 3], mode='LV', trunc=trunc)
    L.retag(3, itag)
    V.retag(0, itag)
    L.permute([0, 3, 1, 2], in_place=True)

    return L, V


def discarded_weight(theta: Tensor, trunc: Optional[dict]) -> float:
    """Compute the discarded weight for an SVD split of Θ under `trunc`.

    Merges axes `[0, 2, 3]` of Θ — the same bipartition the splits use — and
    performs a second SVD with `requires_info=True` to read back
    `info['discarded_weight']`. The extra SVD is intentional; the sweep calls
    it once per cooling step, at the mid-chain bond where the spectrum is
    widest, so its cost is negligible next to the Krylov evolution.

    Parameters
    ----------
    theta:
        Bond tensor with the axis layout `build_bulk` returns.
    trunc:
        Truncation options (`nkeep`, `thresh`) — must match what the splits
        are given, or the number reported describes a different cut than the
        one performed.

    Returns
    -------
    float
        Sum of the *squared* singular values discarded by `trunc` across all
        charge sectors, relative to the total — so for a normalized Θ it is
        `1 - ‖kept‖²`. Zero when `trunc` is `None` or nothing is cut.
    """
    if trunc is None:
        return 0.0
    merged, _ = merge_axes(theta, [0, 2, 3], merged_tag='_dw_merged_')
    *_, info = svd(merged, axis=0, trunc=trunc, requires_info=True)

    return float(info['discarded_weight'])
