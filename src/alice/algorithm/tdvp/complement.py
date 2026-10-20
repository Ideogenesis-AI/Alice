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


"""CBE complement isometry for 1-site-plus TDVP.

Implements the controlled bond expansion (CBE) step used by the 1-site-plus
scheme. Before each 1-site evolution, a complement isometry is computed from a
cheap, factored approximation of the 2-site gradient: H is applied
independently from the left and right to a truncated 2-site tensor; the current
(kept) subspace is projected out from each half; the resulting
"doubly-discarded" outer product is decomposed by SVD to yield new basis
vectors that expand the MPS bond. Those directions are what the subsequent
1-site exponential can flow into, so the bond adapts as in the 2-site scheme
while every local update stays 1-site sized.

The construction follows DMRG's `'1sp'` scheme (`alice.algorithm.dmrg.complement`)
and reuses its factored structure, but three points differ because TDVP
integrates rather than minimizes. In DMRG the expanded center tensor is only an
initial guess that Davidson overwrites, and an approximate effective
Hamiltonian still leaves the energy variational; here both feed straight into
`exp(-i dt H_eff)`.

Exact state preservation
------------------------
The expansion must not move the state. Concatenating complement vectors onto
*both* tensors would, since `Θ = M_i ⊗ M_i1` picks up the extra outer product.
Only the neighbor is expanded by `oplus`, and the center tensor is then *filled
in* exactly by projecting Θ onto the expanded neighbor basis,
`M_i_exp = Θ · M_i1_exp†`. Because the expanded basis contains the old one,
this reproduces Θ exactly; it is the TDVP counterpart of the exact fill-in in
`alice.algorithm.xtrg.complement`. With a right-isometric `M_i1_exp` the result
is the old center tensor padded by zero columns, and writing it as a projection
rather than a zero padding keeps it exact even where the complement is not
perfectly orthogonal to the kept space.

Exact expanded environment
--------------------------
The environment facing the expanded bond is built by a single `step_*_env` call
on the expanded neighbor, which keeps the cross blocks between `M_i1` and the
complement. Assembling it block-diagonally from the two pieces separately, as
DMRG does, drops those blocks: cheap for an initial guess, but a perturbed
`H_eff` would perturb the trajectory itself.

Orthonormal projectors
----------------------
Each bond end projects out its own kept space, and the two ends call for
different tensors. On the neighbor side the site tensor itself is used: it is
exactly isometric there, and its row (or column) space is the full space the
complement has to avoid for the expanded neighbor to stay isometric. On the
center side a 1-site split of the center tensor supplies an orthonormal basis
of the same space the center tensor spans, since `M M†` is a projector only for
an isometry and in a TDVP sweep the center tensor is not one. DMRG uses the raw
site tensor at both ends, which costs it nothing because Davidson overwrites
the center tensor anyway.

What must *not* be used on the center side is the isometric factor of Θ, even
though it is orthonormal and spans the center tensor's column space. Untruncated
it spans more: `decomp` keeps every sector at full dimension regardless of rank,
so that factor can cover the entire (bond, physical) space and project the
complement away entirely, leaving Θ'_disc at roundoff where a real complement
exists. The split of the center tensor alone cannot overreach that way, its
internal bond being the bond itself.

Index convention (shared with `environ.py`, `scheme_1s.py`, `scheme_2s.py`):

- `a, b, c, d` — virtual bond indices of the MPS (bra or ket side).
- `α`          — internal reduced bond of the truncated 2-site tensor.
- `o, p, q`    — virtual bond indices of the MPO.
- `r, s`       — physical indices of site i (bra r, ket s).
- `u, v`       — physical indices of site i+1 (bra u, ket v).

Tensor shapes used in this module:

- `left_half  [a, α, p, r]` — (bra_left, internal, mpo_mid, phys_bra_i)
- `right_half [α, p, c, u]` — (internal, mpo_mid, bra_right, phys_bra_{i+1})
- `N_L        [a, k, r]`    — complement vectors for the right bond of site i.
- `N_R        [k, c, u]`    — complement vectors for the left bond of site i+1.
"""

from __future__ import annotations

from typing import Optional, Tuple

from nicole import Tensor, decomp, einsum, oplus

from .environ import step_left_env, step_right_env
from .scheme_1s import split_backward, split_forward
from .scheme_2s import build_bulk

# Temporary itag assigned to the truncated internal bond in _cheap_factors.
_TRUNC_ITAG: str = '_cbe_trunc_'

# Itag assigned to the internal bond of the center tensor's own split, kept
# distinct from _TRUNC_ITAG so that a projection can carry both at once.
_CENTER_ITAG: str = '_cbe_center_'

# Itag assigned to the complement bond produced by the SVD of Θ'_disc.
_COMPLEMENT_ITAG: str = '_cbe_comp_'

# Relative size below which a discarded part counts as roundoff rather than as a
# genuine complement direction. A saturated bond — one whose dimension already
# equals `dim_outer · d` on either end — has no complement at all, and the
# remainder left by the projection there is pure noise. Keeping such directions
# would be worse than useless: they are not orthogonal to the kept space, so the
# expanded neighbor would stop being isometric and the state would move.
_CBE_TOL: float = 1e-10


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _center_isometry_forward(M_i: Tensor) -> Tensor:
    """Orthonormalize the center tensor's column space for a forward sweep.

    Returns the isometric factor of the 1-site forward split of the center
    tensor, which spans exactly what the center tensor spans over its
    (left bond, physical) indices — the kept space the left end of the bond has
    to project out. The bond tensor the split also produces is discarded; it is
    the non-orthonormal part that makes the center tensor itself unusable here.

    Parameters
    ----------
    M_i:
        Center tensor, axes `(ket_left, ket_right, phys)`.

    Returns
    -------
    Tensor
        Left-isometric tensor with axes `(ket_left, internal, phys)`.
    """
    A, _C = split_forward(M_i, _CENTER_ITAG)
    return A


def _center_isometry_backward(M_i: Tensor) -> Tensor:
    """Orthonormalize the center tensor's row space for a backward sweep.

    Mirror of `_center_isometry_forward`, over the (right bond, physical)
    indices.

    Parameters
    ----------
    M_i:
        Center tensor, axes `(ket_left, ket_right, phys)`.

    Returns
    -------
    Tensor
        Right-isometric tensor with axes `(internal, ket_right, phys)`.
    """
    _C, B = split_backward(M_i, _CENTER_ITAG)
    return B


def _cheap_factors_forward(theta: Tensor, alpha: Optional[int]) -> Tuple[Tensor, Tensor]:
    """Factorize Θ for a forward sweep.

    The pair is a cheap factored form of Θ whose internal bond is at most
    `alpha`, which is what makes the half-applications of H below scale with
    `alpha` instead of the full bond dimension. The direction of the split
    follows the sweep, as in the 2-site scheme.

    Parameters
    ----------
    theta:
        2-site bond tensor with axes
        `(ket_left, ket_right, phys_i, phys_{i+1})`.
    alpha:
        Max bond dimension for the truncated internal bond. `None` keeps all
        singular values (equivalent to using the full 2-site tensor).

    Returns
    -------
    Tensor
        `A` with axes `(ket_left, internal_α, phys_i)`, left-isometric.
    Tensor
        `R = S·Vh` with axes `(internal_α, ket_right, phys_{i+1})`.
    """
    trunc = {'nkeep': alpha} if alpha is not None else None
    # mode='UR' → A is left-isometric and R = S·Vh carries the singular values,
    # the same split the 2-site scheme uses for a forward sweep.
    A, R = decomp(theta, axes=[0, 2], mode='UR', trunc=trunc)

    # A comes back unmerged as (ket_left, phys_i, internal); restore MPS order.
    A.permute([0, 2, 1], in_place=True)
    A.retag(1, _TRUNC_ITAG)
    R.retag(0, _TRUNC_ITAG)

    return A, R


def _cheap_factors_backward(theta: Tensor, alpha: Optional[int]) -> Tuple[Tensor, Tensor]:
    """Factorize Θ for a backward sweep.

    Mirror of `_cheap_factors_forward`: the singular values stay on the left
    factor, so the isometric factor is the one facing the sweep's direction of
    travel.

    Parameters
    ----------
    theta:
        2-site bond tensor with axes
        `(ket_left, ket_right, phys_{i-1}, phys_i)`.
    alpha:
        Max bond dimension for the truncated internal bond. `None` keeps all
        singular values.

    Returns
    -------
    Tensor
        `L = U·S` with axes `(ket_left, internal_α, phys_{i-1})`.
    Tensor
        `B` with axes `(internal_α, ket_right, phys_i)`, right-isometric.
    """
    trunc = {'nkeep': alpha} if alpha is not None else None
    # mode='LV' → L = U·S carries the singular values and B is right-isometric.
    L, B = decomp(theta, axes=[0, 2], mode='LV', trunc=trunc)

    # L comes back unmerged as (ket_left, phys_{i-1}, internal); restore MPS order.
    L.permute([0, 2, 1], in_place=True)
    L.retag(1, _TRUNC_ITAG)
    B.retag(0, _TRUNC_ITAG)

    return L, B


def _left_half(E_left: Tensor, M_trunc: Tensor, W: Tensor) -> Tensor:
    """Compute the left half of H acting on the truncated ket M̃_i.

    Contracts E_left, M̃_i (truncated ket), and W (MPO at site i), leaving the
    internal reduced bond α and MPO bond p free. Together with `_right_half`,
    this allows H·Θ̃ to be reconstructed as
    `einsum('aαpr,αpcu->acru', left_half, right_half)`.

    Parameters
    ----------
    E_left:
        Left environment with axes `(bra_left=a, mpo_left=o, ket_left=b)`.
    M_trunc:
        Truncated ket tensor M̃_i with axes `(ket_left=b, internal_α=α, phys_ket=s)`.
    W:
        MPO tensor at site i with axes `(mpo_left=o, mpo_right=p, phys_bra=r, phys_ket=s)`.

    Returns
    -------
    Tensor
        Left half with axes `(bra_left=a, internal=α, mpo_mid=p, phys_bra=r)`.
    """
    # b (ket_left), o (mpo_left), s (phys_ket) are contracted.
    # Free: a (bra_left), α (internal bond), p (mpo_right), r (phys_bra).
    return einsum('aob,bαs,oprs->aαpr', E_left, M_trunc, W)


def _right_half(E_right: Tensor, M_trunc: Tensor, W: Tensor) -> Tensor:
    """Compute the right half of H acting on the truncated ket M̃_{i+1}.

    Contracts E_right, M̃_{i+1} (truncated ket), and W (MPO at site i+1),
    leaving the internal reduced bond α and MPO bond p free.

    Parameters
    ----------
    E_right:
        Right environment with axes `(bra_right=c, mpo_right=q, ket_right=d)`.
    M_trunc:
        Truncated ket tensor M̃_{i+1} with axes `(internal_α=α, ket_right=d, phys_ket=v)`.
    W:
        MPO tensor at site i+1 with axes `(mpo_left=p, mpo_right=q, phys_bra=u, phys_ket=v)`.

    Returns
    -------
    Tensor
        Right half with axes `(internal=α, mpo_mid=p, bra_right=c, phys_bra=u)`.
    """
    # d (ket_right), q (mpo_right), v (phys_ket) are contracted.
    # Free: α (internal bond), p (mpo_left), c (bra_right), u (phys_bra).
    return einsum('cqd,αdv,pquv->αpcu', E_right, M_trunc, W)


def _project_complement_left(left_half: Tensor, A: Tensor) -> Tensor:
    """Project left_half onto the complement of the kept left subspace.

    `A` is left-isometric (`A† A = I`), so the projector onto its column space
    is `P_L = A A†`, acting on the joint (bra_left, phys_bra) indices of
    left_half. The discarded part is `(I − P_L) left_half`.

    Parameters
    ----------
    left_half:
        Tensor with axes `(bra_left=a, internal=α, mpo_mid=p, phys_bra=r)`.
    A:
        Left-isometric tensor spanning the kept space, axes
        `(ket_left=a, internal=b, phys=r)`.

    Returns
    -------
    Tensor
        Discarded component of left_half, same axes as left_half.
    """
    # proj[a,α,p,r] = Σ_{b,x,z} left_half[x,α,p,z] · A*[x,b,z] · A[a,b,r]
    # left_half and A* share x and z → contract them first (intermediate b·α·p).
    proj = einsum('xαpz,xbz,abr->aαpr', left_half, A.conj(), A)
    return left_half - proj


def _project_complement_right(right_half: Tensor, B: Tensor) -> Tensor:
    """Project right_half onto the complement of the kept right subspace.

    `B` is right-isometric (`B B† = I`), so the projector onto its row space is
    `P_R = B† B`, acting on the joint (bra_right, phys_bra) indices of
    right_half. The discarded part is `(I − P_R) right_half`.

    Parameters
    ----------
    right_half:
        Tensor with axes `(internal=α, mpo_mid=p, bra_right=c, phys_bra=u)`.
    B:
        Right-isometric tensor spanning the kept space, axes
        `(internal=b, ket_right=c, phys=u)`.

    Returns
    -------
    Tensor
        Discarded component of right_half, same axes as right_half.
    """
    # proj[α,p,c,u] = Σ_{b,x,z} right_half[α,p,x,z] · B*[b,x,z] · B[b,c,u]
    # right_half and B* share x and z → contract them first (intermediate b·α·p).
    proj = einsum('αpxz,bxz,bcu->αpcu', right_half, B.conj(), B)
    return right_half - proj


def _has_complement(half: Tensor, half_disc: Tensor) -> bool:
    """Report whether a projected half holds a genuine complement direction.

    Parameters
    ----------
    half:
        Half-application of H before the projection.
    half_disc:
        What the projection left of it.

    Returns
    -------
    bool
        `True` when the remainder is large enough to be signal rather than
        roundoff.
    """
    return half_disc.norm() > _CBE_TOL * half.norm()


def _complement_vectors(
    left_half_disc: Tensor,
    right_half_disc: Tensor,
    k_expand: int,
    flow: str,
) -> Tuple[Tensor, Tensor]:
    """Extract complement vectors from the discarded 2-site outer product.

    Contracts the two discarded halves into the "doubly-discarded" 2-site tensor
    Θ'_disc and decomposes it by truncated SVD. Both singular-vector factors are
    isometric; a sweep expands the neighbor with the one facing it and fills the
    center tensor in exactly, so each direction consumes only one of the two.

    The required `flow` depends on the sweep direction because `theta_disc`
    carries different index directions in each case:

    - Forward sweep: use `flow='<<'`. The index directions of `theta_disc`
      produce N_R with a new-bond direction matching the right-isometric
      neighbor's left bond.
    - Backward sweep: use `flow='>>'`. Directions of `theta_disc` are reversed
      relative to the forward case, so the opposite flow is needed for N_L to
      match the left-isometric neighbor's right bond.

    Parameters
    ----------
    left_half_disc:
        Discarded left half with axes `(bra_left=a, internal=α, mpo_mid=p, phys_bra=r)`.
    right_half_disc:
        Discarded right half with axes `(internal=α, mpo_mid=p, bra_right=c, phys_bra=u)`.
    k_expand:
        Maximum number of complement vectors to extract (nkeep for the SVD).
    flow:
        Arrow-flow string passed to `decomp`. Use `'<<'` for a forward sweep and
        `'>>'` for a backward sweep.

    Returns
    -------
    Tensor
        N_L with axes `(bra_left=a, k=new_bond, phys=r)` — left-isometric, the
        new right-bond vectors for site i.
    Tensor
        N_R with axes `(k=new_bond, bra_right=c, phys=u)` — right-isometric, the
        new left-bond vectors for site i+1.
    """
    # Contract the two discarded halves into Θ'_disc.
    # Axes (a, c, r, u) match the standard 2-site bond tensor convention.
    theta_disc = einsum('aαpr,αpcu->acru', left_half_disc, right_half_disc)

    # SVD with axes=[0, 2] fuses (a, r) as left and (c, u) as right. The
    # threshold drops directions that only exist because `k_expand` asks for
    # more than the complement holds; `nkeep` alone would hand back roundoff
    # vectors, which do not lie in the complement and would spoil the isometry
    # of the expanded neighbor.
    U, _S, Vh = decomp(
        theta_disc, axes=[0, 2], mode='SVD',
        trunc={'nkeep': k_expand, 'thresh': _CBE_TOL * theta_disc.norm()},
        flow=flow, itag=_COMPLEMENT_ITAG,
    )
    # After decomp + unmerge: U has axes (a, r, k); permute to (a, k, r).
    U.permute([0, 2, 1], in_place=True)   # N_L: (a, k, r)
    # Vh has axes (k, c, u). N_R = Vh.
    return U, Vh


def _fill_center_forward(theta: Tensor, M_i1_exp: Tensor) -> Tensor:
    """Represent Θ exactly in the expanded right basis (forward sweep).

    Projects Θ onto the expanded, right-isometric neighbor,
    `M_i_exp = Θ · M_i1_exp†`. Since the expanded basis contains the old one,
    the pair `(M_i_exp, M_i1_exp)` represents the same state as before the
    expansion — the new bond directions simply enter with zero amplitude, ready
    for the 1-site exponential to populate.

    Parameters
    ----------
    theta:
        2-site bond tensor with axes
        `(ket_left, ket_right, phys_i, phys_{i+1})`.
    M_i1_exp:
        Expanded right-isometric tensor at site i+1, axes
        `(ket_left_exp, ket_right, phys)`.

    Returns
    -------
    Tensor
        `M_i_exp` with axes `(ket_left, ket_right_exp, phys)`.
    """
    # c (ket_right) and u (phys_{i+1}) are contracted against the conjugated
    # neighbor; its expanded left bond b becomes the new right bond of site i.
    return einsum('acru,bcu->abr', theta, M_i1_exp.conj())


def _fill_center_backward(theta: Tensor, M_im1_exp: Tensor) -> Tensor:
    """Represent Θ exactly in the expanded left basis (backward sweep).

    Mirror of `_fill_center_forward`: projects Θ onto the expanded,
    left-isometric neighbor at site i-1.

    Parameters
    ----------
    theta:
        2-site bond tensor with axes
        `(ket_left, ket_right, phys_{i-1}, phys_i)`.
    M_im1_exp:
        Expanded left-isometric tensor at site i-1, axes
        `(ket_left, ket_right_exp, phys)`.

    Returns
    -------
    Tensor
        `M_i_exp` with axes `(ket_left_exp, ket_right, phys)`.
    """
    # a (ket_left) and r (phys_{i-1}) are contracted against the conjugated
    # neighbor; its expanded right bond b becomes the new left bond of site i.
    return einsum('acru,abr->bcu', theta, M_im1_exp.conj())


# ---------------------------------------------------------------------------
# Public expansion functions
# ---------------------------------------------------------------------------

def expand_forward(
    M_i: Tensor,
    M_i1: Tensor,
    W_i: Tensor,
    W_i1: Tensor,
    E_left_i: Tensor,
    E_right_i1: Tensor,
    k_expand: int,
    alpha: Optional[int],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute the CBE complement and expand bond (i, i+1) for a forward sweep.

    The orthogonality center sits at site i; `M_i1` is right-isometric. The
    procedure is:

    1. Truncated SVD of Θ = M_i ⊗ M_i1 → cheap factors (bond α).
    2. Factored half-application of H: `left_half` from E_left and the left
       factor; `right_half` from E_right and the right factor.
    3. Discarded-space projection: the left half against the orthonormalized
       center tensor, the right half against the right-isometric `M_i1` itself.
    4. SVD of Θ'_disc → N_R, the new left-bond vectors for M_i1.
    5. Expansion of M_i1 via `oplus`, and exact fill-in of the center tensor.
    6. Expansion of the right environment at site i from E_right[i+1].

    Parameters
    ----------
    M_i:
        Center tensor at site i, axes `(ket_left, ket_right, phys)`.
    M_i1:
        Right-isometric MPS tensor at site i+1, axes `(ket_left, ket_right, phys)`.
    W_i:
        MPO tensor at site i, axes `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    W_i1:
        MPO tensor at site i+1, axes `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    E_left_i:
        Left environment at site i, axes `(bra_left, mpo_left, ket_left)`.
    E_right_i1:
        Right environment at site i+1, axes `(bra_right, mpo_right, ket_right)`.
    k_expand:
        Maximum number of complement vectors to add to the bond.
    alpha:
        Truncated internal bond dimension for Θ̃. `None` keeps the full bond.

    Returns
    -------
    Tensor
        M_i_exp — center tensor in the expanded basis, axes
        `(ket_left, ket_right_exp, phys)`.
    Tensor
        M_i1_exp — expanded M_i1 with axes `(ket_left_exp, ket_right, phys)`.
    Tensor
        E_right_i_exp — expanded right environment at site i, axes
        `(bra_right_exp, mpo_right, ket_right_exp)`.
    """
    # Step 1: cheap factored form of Θ, isometric on the center side.
    theta = build_bulk(M_i, M_i1)
    A, R = _cheap_factors_forward(theta, alpha)

    # Step 2: factored half-application of H.
    lh = _left_half(E_left_i,   A, W_i)
    rh = _right_half(E_right_i1, R, W_i1)

    # Step 3: discarded-space projection. Each end projects against the tensor
    # that spans exactly its own kept space: the orthonormalized center tensor
    # on the left, and M_i1 itself on the right, whose whole row space the
    # complement must miss for the expanded neighbor to stay isometric — which
    # is what makes the fill-in below exact.
    lh_disc = _project_complement_left(lh,  _center_isometry_forward(M_i))
    rh_disc = _project_complement_right(rh, M_i1)

    # A saturated bond has nothing to expand into; leave it as it is and hand
    # back the environment the 1-site step would have used anyway.
    if not (_has_complement(lh, lh_disc) and _has_complement(rh, rh_disc)):
        return M_i, M_i1, step_right_env(E_right_i1, M_i1, W_i1)

    # Step 4: complement vectors from the SVD of Θ'_disc. Only N_R is used: the
    # center tensor is filled in exactly rather than padded with N_L.
    # flow='<<' gives N_R a new-bond direction matching M_i1's left bond.
    _N_L, N_R = _complement_vectors(lh_disc, rh_disc, k_expand, flow='<<')

    # Step 5: expand M_i1's left bond (axis 0), then re-express Θ in that basis.
    M_i1_exp = oplus(M_i1, N_R, axes=[0])
    M_i_exp = _fill_center_forward(theta, M_i1_exp)

    # Step 6: expanded right environment at site i. One step over the expanded
    # neighbor keeps the M_i1-N_R cross blocks, so H_eff stays exact.
    E_right_i_exp = step_right_env(E_right_i1, M_i1_exp, W_i1)

    return M_i_exp, M_i1_exp, E_right_i_exp


def expand_backward(
    M_i: Tensor,
    M_im1: Tensor,
    W_i: Tensor,
    W_im1: Tensor,
    E_left_im1: Tensor,
    E_right_i: Tensor,
    k_expand: int,
    alpha: Optional[int],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute the CBE complement and expand bond (i-1, i) for a backward sweep.

    Mirror of `expand_forward`. The orthogonality center sits at site i and
    `M_im1` is left-isometric, so the complement expands the left bond of the
    center and the right bond of its neighbor:

    1. Truncated SVD of Θ = M_im1 ⊗ M_i → cheap factors (bond α).
    2. Factored half-application of H using E_left[i-1] and E_right[i].
    3. Discarded-space projection: the right half against the orthonormalized
       center tensor, the left half against the left-isometric `M_im1` itself.
    4. SVD of Θ'_disc → N_L, the new right-bond vectors for M_im1.
    5. Expansion of M_im1 via `oplus`, and exact fill-in of the center tensor.
    6. Expansion of the left environment at site i from E_left[i-1].

    Parameters
    ----------
    M_i:
        Center tensor at site i, axes `(ket_left, ket_right, phys)`.
    M_im1:
        Left-isometric MPS tensor at site i-1, axes `(ket_left, ket_right, phys)`.
    W_i:
        MPO tensor at site i, axes `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    W_im1:
        MPO tensor at site i-1, axes `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    E_left_im1:
        Left environment at site i-1, axes `(bra_left, mpo_left, ket_left)`.
    E_right_i:
        Right environment at site i, axes `(bra_right, mpo_right, ket_right)`.
    k_expand:
        Maximum number of complement vectors to add to the bond.
    alpha:
        Truncated internal bond dimension for Θ̃. `None` keeps the full bond.

    Returns
    -------
    Tensor
        M_i_exp — center tensor in the expanded basis, axes
        `(ket_left_exp, ket_right, phys)`.
    Tensor
        M_im1_exp — expanded M_im1 with axes `(ket_left, ket_right_exp, phys)`.
    Tensor
        E_left_i_exp — expanded left environment at site i, axes
        `(bra_left_exp, mpo_left, ket_left_exp)`.
    """
    # Step 1: cheap factored form of Θ = M_im1 ⊗ M_i, isometric on the center side.
    theta = build_bulk(M_im1, M_i)
    L, B = _cheap_factors_backward(theta, alpha)

    # Step 2: factored half-application of H.
    lh = _left_half(E_left_im1, L, W_im1)
    rh = _right_half(E_right_i,  B, W_i)

    # Step 3: discarded-space projection. Mirroring the forward sweep, the left
    # end projects against M_im1, whose column space is the whole kept space
    # the complement has to miss, and the center end against its own tensor
    # orthonormalized.
    lh_disc = _project_complement_left(lh,  M_im1)
    rh_disc = _project_complement_right(rh, _center_isometry_backward(M_i))

    # A saturated bond has nothing to expand into; leave it as it is and hand
    # back the environment the 1-site step would have used anyway.
    if not (_has_complement(lh, lh_disc) and _has_complement(rh, rh_disc)):
        return M_i, M_im1, step_left_env(E_left_im1, M_im1, W_im1)

    # Step 4: complement vectors. Only N_L is used here, for the same reason
    # N_R alone is used in the forward sweep.
    # flow='>>' gives N_L a new-bond direction matching M_im1's right bond.
    N_L, _N_R = _complement_vectors(lh_disc, rh_disc, k_expand, flow='>>')

    # Step 5: expand M_im1's right bond (axis 1), then re-express Θ in that basis.
    M_im1_exp = oplus(M_im1, N_L, axes=[1])
    M_i_exp = _fill_center_backward(theta, M_im1_exp)

    # Step 6: expanded left environment at site i, exact for the same reason as
    # the right environment in the forward sweep.
    E_left_i_exp = step_left_env(E_left_im1, M_im1_exp, W_im1)

    return M_i_exp, M_im1_exp, E_left_i_exp
