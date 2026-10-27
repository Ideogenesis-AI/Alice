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


"""1-site VUMPS update scheme.

The variational uniform MPS algorithm (Zauner-Stauber, Vanderstraeten,
Fishman, Verstraete and Haegeman, PRB 97, 045145 (2018)) optimizes one
position of the unit cell with two eigenproblems and a gauge step that
reconciles them:

- `optimize_center` solves `H_AC AC = e AC` for the center tensor of a
  site, with `H_AC` the sandwich of the left environment, the iMPO tensor
  and the right environment.
- `optimize_bond` solves `H_C C = e C` for the bond matrix, with `H_C` the
  same sandwich minus the iMPO tensor.
- `gauge_tensors` turns the two answers back into a left- and a
  right-orthonormal tensor.

Nothing is ever split or truncated. `AC` and `C` are solved independently,
against environments that are the state itself repeated, so away from a
fixed point the two describe slightly different states. The size of that
disagreement is the convergence measure, vanishing exactly at a stationary
point of the energy on the uniform manifold.

Gauge reconciliation
--------------------
`AL` is the minimizer of `|AC - AL C|` over left-orthonormal tensors,
which is `U_l(AC) U_l(C)^dag` where `U_l(X)` is the orthonormal polar
factor of `X`. The two polar factors are taken separately, because the
polar decomposition of the product `AC C^dag` would square the condition
number of `C`, which is ill-conditioned by construction, since its small
singular values are the ones the bond dimension is about to stop
resolving.

Index convention (shared with `environ.py`)
--------------------------------------------
- `a, b, c, d` — virtual bond indices of the iMPS (bra or ket side)
- `o, p`       — virtual bond indices of the iMPO
- `r, s`       — physical indices

Effective Hamiltonian contraction layout:

    E_left [a, o, b]    — (bra_left,  mpo_left,  ket_left)
    W      [o, p, r, s] — (mpo_left,  mpo_right, phys_bra, phys_ket)
    E_right[c, p, d]    — (bra_right, mpo_right, ket_right)
    AC     [b, d, s]    — (ket_left,  ket_right, phys_ket)
    output [a, c, r]    — (bra_left,  bra_right, phys_bra)
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

from nicole import Tensor, decomp, einsum

from .davidson import davidson

# itag worn by the inner index of a polar decomposition, which is summed
# over immediately and never reaches a stored tensor.
_POLAR_ITAG = '_polar_'


# ---------------------------------------------------------------------------
# Effective Hamiltonians
# ---------------------------------------------------------------------------

def matvec_center(AC: Tensor, W: Tensor, E_left: Tensor, E_right: Tensor) -> Tensor:
    """Apply the center-site effective Hamiltonian `H_AC` to `AC`.

    Parameters
    ----------
    AC:
        Center site tensor with axes `(ket_left, ket_right, phys_ket)`.
    W:
        iMPO site tensor with axes `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    E_left:
        Left environment on the site's left bond, axes `(bra, mpo, ket)`.
    E_right:
        Right environment on the site's right bond, axes `(bra, mpo, ket)`.

    Returns
    -------
    Tensor
        `H_AC|AC>`, with the same axis layout as `AC`.
    """
    # Contracting E_left against AC first eliminates a virtual bond before
    # the iMPO index is introduced, which keeps the intermediates small.
    return einsum('aob,bds,oprs,cpd->acr', E_left, AC, W, E_right)


def matvec_bond(C: Tensor, E_left: Tensor, E_right: Tensor) -> Tensor:
    """Apply the bond effective Hamiltonian `H_C` to `C`.

    `H_C` is `H_AC` with the iMPO tensor removed, so the two environments
    meet directly on their iMPO index. Both must sit on the same bond.

    Parameters
    ----------
    C:
        Bond matrix with axes `(ket_left, ket_right)`.
    E_left:
        Left environment on that bond, axes `(bra, mpo, ket)`.
    E_right:
        Right environment on that bond, axes `(bra, mpo, ket)`.

    Returns
    -------
    Tensor
        `H_C|C>`, with the same axis layout as `C`.
    """
    return einsum('aob,bd,cod->ac', E_left, C, E_right)


# ---------------------------------------------------------------------------
# Local eigenproblems
# ---------------------------------------------------------------------------

def optimize_center(
    AC: Tensor,
    W: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    davidson_opts: dict,
) -> Tuple[float, Tensor, float]:
    """Find the optimal center site tensor via the Davidson eigensolver.

    A pure function: it takes tensors and returns tensors without mutating
    any state.

    Parameters
    ----------
    AC:
        Current center site tensor, used as the Davidson starting vector.
    W:
        iMPO site tensor.
    E_left:
        Left environment on the site's left bond.
    E_right:
        Right environment on the site's right bond.
    davidson_opts:
        Keyword arguments forwarded to `davidson`: `max_iter`, `tol`,
        `max_subspace`.

    Returns
    -------
    energy:
        Lowest Ritz value. Regularized along with the environments, so it
        carries no absolute scale.
    center:
        Optimized center tensor, normalized.
    error:
        Final Davidson residual norm.
    """
    mv = partial(matvec_center, W=W, E_left=E_left, E_right=E_right)
    return davidson(mv, AC, **davidson_opts)


def optimize_bond(
    C: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    davidson_opts: dict,
) -> Tuple[float, Tensor, float]:
    """Find the optimal bond matrix via the Davidson eigensolver.

    Parameters
    ----------
    C:
        Current bond matrix, used as the Davidson starting vector.
    E_left:
        Left environment on that bond.
    E_right:
        Right environment on that bond.
    davidson_opts:
        Keyword arguments forwarded to `davidson`.

    Returns
    -------
    energy:
        Lowest Ritz value, regularized as in `optimize_center`.
    bond:
        Optimized bond matrix, normalized.
    error:
        Final Davidson residual norm.
    """
    mv = partial(matvec_bond, E_left=E_left, E_right=E_right)
    return davidson(mv, C, **davidson_opts)


# ---------------------------------------------------------------------------
# Gauge reconciliation
# ---------------------------------------------------------------------------

def _polar_left(t: Tensor, axes) -> Tensor:
    """Return the orthonormal polar factor of `t`, isometric over `axes`.

    For `t = U S Vh` grouped with `axes` on the left, this is `U Vh`: the
    orthonormal matrix closest to `t`. Discarding `S` is the whole point,
    since `S` is what carries the conditioning.
    """
    u, _, vh = decomp(t, axes=axes, mode='SVD', flow='<<', itag=_POLAR_ITAG)
    if len(u.indices) == 3:
        # AC grouped as (left, phys | right): u is (left, phys, inner).
        return einsum('arb,bc->acr', u, vh)
    if len(vh.indices) == 3:
        # AC grouped as (left | right, phys): vh is (inner, right, phys).
        return einsum('ab,bcr->acr', u, vh)
    return einsum('ab,bc->ac', u, vh)


def gauge_tensors(
    AC: Tensor,
    C_left: Tensor,
    C_right: Tensor,
) -> Tuple[Tensor, Tensor, float]:
    """Turn a center tensor and its two bond matrices into canonical forms.

    Implements the VUMPS gauge step

        AL = U_l(AC) U_l(C_right)^dag
        AR = U_r(C_left)^dag U_r(AC)

    where `U_l` and `U_r` are orthonormal polar factors taken with the
    grouping that makes the factor left- or right-isometric.

    Parameters
    ----------
    AC:
        Optimized center tensor for the site, axes `(left, right, phys)`.
    C_left:
        Optimized bond matrix on the site's left bond.
    C_right:
        Optimized bond matrix on the site's right bond.

    Returns
    -------
    left_iso:
        Left-orthonormal tensor `AL` for the site.
    right_iso:
        Right-orthonormal tensor `AR` for the site.
    error:
        `|AC - AL C_right|`, the local VUMPS gradient norm. It is zero
        exactly when the two eigenproblems agree, which is the fixed point
        the algorithm is looking for.
    """
    polar_ac_l = _polar_left(AC, [0, 2])
    polar_ac_r = _polar_left(AC, 0)
    polar_c_l = _polar_left(C_left, 0)
    polar_c_r = _polar_left(C_right, 0)

    left_iso = einsum('acr,bc->abr', polar_ac_l, polar_c_r.conj())
    right_iso = einsum('ab,acr->bcr', polar_c_l.conj(), polar_ac_r)

    error = float((AC - einsum('abr,bc->acr', left_iso, C_right)).norm())
    return left_iso, right_iso, error
