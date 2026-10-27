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


"""2-site VUMPS update scheme.

The one-site update solves for a center tensor on a bond whose index it
cannot change, so a run is confined to the bond dimension its initial
cell happened to carry. The two-site update lifts that limit: it solves
a pair of neighboring sites at once and then splits the answer, and the
SVD that splits it is free to choose how wide the bond between them
should be.

- `matvec_two_site` applies the effective Hamiltonian of a site pair,
  the same sandwich as in the one-site scheme with a second iMPO tensor
  threaded through it.
- `optimize_two_site` solves it for the lowest eigenpair.
- `split_two_site` cuts the answer back into two site tensors and the
  bond matrix between them, truncating as it goes.
- `discarded_weight` reports what that truncation threw away.

Why the split needs no inverse
------------------------------
The SVD hands back `U`, `S` and `Vh` on the new bond together, and the
three are exactly the pieces the canonical form is made of. Writing the
new bond matrix as `C = S` and the new center tensors as `U S` and
`S Vh`, the gauge step of the one-site scheme reproduces `U` and `Vh`
unchanged, since the orthonormal polar factor of `U S` is `U` and that
of `S` is the identity. So the left-orthonormal tensor is exact on the
side that was just solved, and only the other side carries a gradient.
Nothing has to be divided by a bond matrix.

Only the bond between the two sites changes. The outer bonds of the pair
keep the indices they had, which is what lets a pass walk the cell one
pair at a time and still leave a cell that tiles.

Index convention (shared with `environ.py` and `scheme_1s.py`)
--------------------------------------------------------------
- `a, b, c, d` — virtual bond indices of the iMPS (bra or ket side)
- `o, p, q`    — virtual bond indices of the iMPO
- `r, s, u, v` — physical indices

Two-site contraction layout:

    E_left [a, o, b]    — (bra_left,  mpo_left,   ket_left)
    W_left [o, p, r, s] — (mpo_left,  mpo_middle, phys_bra, phys_ket)
    W_right[p, q, u, v] — (mpo_middle, mpo_right, phys_bra, phys_ket)
    E_right[c, q, d]    — (bra_right, mpo_right,  ket_right)
    AC2    [b, d, s, v] — (ket_left,  ket_right,  phys_ket, phys_ket)
    output [a, c, r, u] — (bra_left,  bra_right,  phys_bra, phys_bra)
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

from nicole import Tensor, decomp, einsum, merge_axes
from nicole.decomp import svd

from .davidson import davidson

# itag worn by the bipartition axis of the extra SVD that measures the
# discarded weight. It never reaches a stored tensor.
_WEIGHT_ITAG = '_discarded_'


# ---------------------------------------------------------------------------
# Effective Hamiltonian
# ---------------------------------------------------------------------------

def matvec_two_site(
    AC2: Tensor,
    W_left: Tensor,
    W_right: Tensor,
    E_left: Tensor,
    E_right: Tensor,
) -> Tensor:
    """Apply the two-site effective Hamiltonian to `AC2`.

    Parameters
    ----------
    AC2:
        Two-site center tensor with axes `(ket_left, ket_right, phys_left,
        phys_right)`.
    W_left:
        iMPO tensor of the left site of the pair.
    W_right:
        iMPO tensor of the right site of the pair.
    E_left:
        Left environment on the pair's left bond, axes `(bra, mpo, ket)`.
    E_right:
        Right environment on the pair's right bond, axes `(bra, mpo, ket)`.

    Returns
    -------
    Tensor
        `H_AC2|AC2>`, with the same axis layout as `AC2`.
    """
    # Nicole contracts a multi-tensor equation strictly left to right, and
    # this order never holds more than one iMPO index and one open virtual
    # bond at a time.
    return einsum(
        'aob,bdsv,oprs,pquv,cqd->acru', E_left, AC2, W_left, W_right, E_right,
    )


# ---------------------------------------------------------------------------
# Local eigenproblem
# ---------------------------------------------------------------------------

def optimize_two_site(
    AC2: Tensor,
    W_left: Tensor,
    W_right: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    davidson_opts: dict,
) -> Tuple[float, Tensor, float]:
    """Find the optimal two-site center tensor via the Davidson eigensolver.

    A pure function: it takes tensors and returns tensors without mutating
    any state.

    Parameters
    ----------
    AC2:
        Current two-site center tensor, used as the Davidson starting
        vector.
    W_left:
        iMPO tensor of the left site of the pair.
    W_right:
        iMPO tensor of the right site of the pair.
    E_left:
        Left environment on the pair's left bond.
    E_right:
        Right environment on the pair's right bond.
    davidson_opts:
        Keyword arguments forwarded to `davidson`: `max_iter`, `tol`,
        `max_subspace`.

    Returns
    -------
    energy:
        Lowest Ritz value. Regularized along with the environments, so it
        carries no absolute scale.
    center:
        Optimized two-site tensor, normalized.
    error:
        Final Davidson residual norm.
    """
    mv = partial(
        matvec_two_site,
        W_left=W_left,
        W_right=W_right,
        E_left=E_left,
        E_right=E_right,
    )
    return davidson(mv, AC2, **davidson_opts)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_two_site(
    AC2: Tensor,
    itag: str,
    trunc: Optional[dict] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Split a two-site tensor into two site tensors and the bond between.

    The bipartition puts the left bond and the left physical axis on one
    side, which is the same cut `AL[k] C[k+1] = C[k] AR[k]` is written
    across. The three factors come back ready to use as they are: `U` is
    the new left-orthonormal tensor of the left site, `Vh` the new
    right-orthonormal tensor of the right site, and `S` the bond matrix
    between them, already diagonal and so already in the Schmidt basis
    that `iMPS.set_canonical` expects.

    Parameters
    ----------
    AC2:
        Two-site center tensor with axes `(ket_left, ket_right, phys_left,
        phys_right)`.
    itag:
        itag for the new bond, which is the itag the cell already uses for
        the bond between the two sites.
    trunc:
        Truncation options forwarded to `decomp` (`nkeep`, `thresh`). Pass
        `None` to keep every singular value the bipartition admits, which
        multiplies the bond dimension by the physical dimension.

    Returns
    -------
    left_iso:
        Left-orthonormal tensor with axes `(ket_left, new_bond, phys_left)`.
    bond:
        Diagonal bond matrix on the new bond.
    right_iso:
        Right-orthonormal tensor with axes `(new_bond, ket_right,
        phys_right)`.
    """
    # '<<' sends the new bond out of the left factor, matching the
    # direction convention the gauge iteration gives AL and AR.
    u, s, vh = decomp(AC2, axes=[0, 2], mode='SVD', flow='<<', itag=itag, trunc=trunc)
    # decomp unmerges U back to (ket_left, phys_left, new_bond); the iMPS
    # layout wants the bond on axis 1.
    u.permute([0, 2, 1], in_place=True)
    return u, s, vh


def discarded_weight(AC2: Tensor, trunc: Optional[dict] = None) -> float:
    """Return the weight `split_two_site` would discard from `AC2`.

    Repeats the bipartition of `split_two_site` and asks for the
    truncation report that `decomp` does not surface. The second
    decomposition is deliberate: it runs once per bond per iteration,
    against a Davidson solve that has already cost many applications of
    the effective Hamiltonian.

    Parameters
    ----------
    AC2:
        Two-site center tensor, as passed to `split_two_site`.
    trunc:
        The same truncation options, so the number describes the split
        that was actually made.

    Returns
    -------
    float
        Relative sum of the squared singular values dropped, in `[0, 1)`.
        Zero when `trunc` is `None` or nothing is cut.
    """
    merged, _ = merge_axes(AC2, [0, 2], merged_tag=_WEIGHT_ITAG)
    *_, info = svd(merged, axis=0, trunc=trunc, requires_info=True)
    return float(info['discarded_weight'])
