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


"""CBE complement isometry for 1-site-plus VUMPS.

Implements the controlled bond expansion (CBE) step the 1-site-plus scheme
runs before each center-tensor solve. A complement isometry is computed from
a cheap, factored approximation of the two-site gradient: the iMPO is applied
independently from the left and the right to a truncated two-site tensor, the
kept subspace is projected out of each half, and the resulting
"doubly-discarded" outer product is decomposed by SVD into new basis vectors
that widen the bond. Those directions are what the one-site eigenproblem that
follows can then fill, so the bond dimension adapts as it does under `'2s'`
while every local solve stays one-site sized.

Why the one-site scheme needs this
----------------------------------
The one-site update ends in a pair of polar decompositions, which discard
nothing and so cannot replace a bond index: a `'1s'` run is confined to the
bond dimension its initial cell happened to carry. The two-site update
escapes that by solving a pair and splitting the answer, at a local problem
`d` times larger on each side. CBE buys the same freedom at the one-site
price, by widening the bond *before* the solve instead of letting a larger
solve discover the width on its own.

What the pass sees
------------------
A forward pass reaches bond `j` with the left-orthonormal tensor of the site
behind it and the center tensor of the site in front, which is exactly the
geometry of a backward sweep in the finite algorithm: the neighbor is on the
left and the center is on the right. So only one of the two complement
factors is ever used here, the one facing the neighbor.

Exact state preservation
------------------------
The expansion must not move the state. Concatenating complement vectors onto
*both* tensors would, since `Θ = AL ⊗ AC` picks up the extra outer product.
Only the neighbor is expanded by `oplus`, and the center tensor is then
*filled in* exactly by projecting Θ onto the expanded neighbor basis,
`AC_exp = AL_exp† · Θ`. Because the expanded basis contains the old one, this
reproduces Θ exactly; with a left-isometric `AL_exp` the answer is the old
center tensor padded by zero rows, and writing it as a projection instead of
as a zero padding keeps it exact even where the complement is not perfectly
orthogonal to the kept space.

Exact expanded environment
--------------------------
The left environment facing the expanded bond is a single `step_left_env`
call on the expanded neighbor, which keeps the cross blocks between `AL` and
the complement. Nothing has to be assembled block-diagonally, and nothing on
the `AR` side of the bond has to be expanded at all: the block being stepped
sits on the bond *behind* the one being widened, which the expansion does not
touch, and the bond matrix comes out of `split_center`, not out of an
eigenproblem of its own.

Orthonormal projectors
----------------------
Each end of the bond projects out its own kept space, and the two ends call
for different tensors. On the neighbor side the tensor itself is used: it is
exactly isometric there, and its column space is the whole space the
complement has to avoid for the expanded neighbor to stay isometric. On the
center side a one-site split of the center tensor supplies an orthonormal
basis of the same space the center tensor spans, since `M M†` is a projector
only for an isometry and the center tensor is not one.

What must *not* be used on the center side is the isometric factor of Θ, even
though it is orthonormal and spans the center tensor's row space.
Untruncated it spans more: `decomp` keeps every sector at full dimension
regardless of rank, so that factor can cover the entire (bond, physical)
space and project the complement away entirely, leaving Θ'_disc at roundoff
where a real complement exists. The split of the center tensor alone cannot
overreach that way, its internal bond being the bond itself.

Index convention (shared with `environ.py`, `scheme_1s.py`, `scheme_2s.py`)
---------------------------------------------------------------------------
- `a, b, c, d` — virtual bond indices of the iMPS (bra or ket side)
- `α`          — internal reduced bond of the truncated two-site tensor
- `o, p, q`    — virtual bond indices of the iMPO
- `r, s`       — physical indices of the neighbor site (bra `r`, ket `s`)
- `u, v`       — physical indices of the center site (bra `u`, ket `v`)

Tensor shapes used in this module:

- `left_half  [a, α, p, r]` — (bra_left, internal, mpo_mid, phys_bra_neighbor)
- `right_half [α, p, c, u]` — (internal, mpo_mid, bra_right, phys_bra_center)
- `N_L        [a, k, r]`    — complement vectors for the neighbor's right bond
"""

from __future__ import annotations

from typing import Optional, Tuple

from nicole import Tensor, decomp, einsum, merge_axes, oplus
from nicole.decomp import svd

from .environ import step_left_env

# itag worn by the truncated internal bond of the cheap two-site factors.
_TRUNC_ITAG: str = '_cbe_trunc_'

# itag worn by the internal bond of the center tensor's own split, kept
# distinct from `_TRUNC_ITAG` so that a projection can carry both at once.
_CENTER_ITAG: str = '_cbe_center_'

# itag worn by the complement bond produced by the SVD of Θ'_disc.
_COMPLEMENT_ITAG: str = '_cbe_comp_'

# itag worn by the bipartition axis of the extra SVD that measures the
# discarded weight. It never reaches a stored tensor.
_WEIGHT_ITAG: str = '_discarded_'

# Relative size below which a complement direction counts as roundoff rather
# than as signal. A saturated bond — one whose dimension already equals
# `dim_outer · d` on either end — has no complement at all, and neither has a
# bond the iMPO cannot yet act across. What the SVD returns there is noise, or
# in the exactly vanishing case an arbitrary unit vector lying in the kept
# space, and either would stop the expanded neighbor from being isometric and
# move the state. The tolerance is taken against `_noise_floor`, never against
# Θ'_disc itself.
_CBE_TOL: float = 1e-10


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _center_isometry_right(AC: Tensor) -> Tensor:
    """Orthonormalize the center tensor's row space.

    Returns the isometric factor of a one-site split of the center tensor,
    which spans exactly what the center tensor spans over its (right bond,
    physical) indices — the kept space the center end of the bond has to
    project out. The bond tensor the split also produces is discarded; it is
    the non-orthonormal part that makes the center tensor itself unusable
    here.

    Parameters
    ----------
    AC:
        Center tensor, axes `(ket_left, ket_right, phys)`.

    Returns
    -------
    Tensor
        Right-isometric tensor with axes `(internal, ket_right, phys)`.
    """
    # rows = left bond only → LV gives the isometry with axes
    # (internal, ket_right, phys), already in the iMPS axis order.
    _C, B = decomp(AC, axes=0, flow='<<', mode='LV')
    B.retag(0, _CENTER_ITAG)
    return B


def _cheap_factors(theta: Tensor, alpha: Optional[int]) -> Tuple[Tensor, Tensor]:
    """Factorize Θ into a cheap pair whose internal bond is at most `alpha`.

    The truncation is the cost-saving measure of the whole scheme: instead of
    applying the iMPO to the full two-site tensor at bond dimension `chi`, the
    SVD reduces the internal bond to at most `alpha`, and the
    half-applications below then scale with `alpha` instead of with `chi`.

    The singular values stay on the left factor, so the isometric one is the
    factor on the center side, which is the side whose kept space the center
    tensor's own split describes.

    Parameters
    ----------
    theta:
        Two-site bond tensor with axes `(ket_left, ket_right, phys_neighbor,
        phys_center)`.
    alpha:
        Largest internal bond dimension kept. `None` keeps every singular
        value, which is equivalent to using the full two-site tensor.

    Returns
    -------
    L:
        `U·S` with axes `(ket_left, internal_α, phys_neighbor)`.
    B:
        Right-isometric factor with axes `(internal_α, ket_right,
        phys_center)`.
    """
    trunc = {'nkeep': alpha} if alpha is not None else None
    # mode='LV' → L = U·S carries the singular values and B is right-isometric.
    L, B = decomp(theta, axes=[0, 2], mode='LV', trunc=trunc)

    # L comes back unmerged as (ket_left, phys_neighbor, internal); restore the
    # iMPS axis order.
    L.permute([0, 2, 1], in_place=True)
    L.retag(1, _TRUNC_ITAG)
    B.retag(0, _TRUNC_ITAG)

    return L, B


def _left_half(E_left: Tensor, L: Tensor, W: Tensor) -> Tensor:
    """Apply the iMPO to the cheap left factor from the left.

    Contracts the left environment, the left factor of Θ and the iMPO tensor
    of the neighbor site, leaving the internal reduced bond and the iMPO bond
    free. Together with `_right_half` this reconstructs `H·Θ̃` as
    `einsum('aαpr,αpcu->acru', left_half, right_half)`.

    Parameters
    ----------
    E_left:
        Left environment on the neighbor's left bond, axes `(bra, mpo, ket)`.
    L:
        Cheap left factor, axes `(ket_left, internal_α, phys_ket)`.
    W:
        iMPO tensor of the neighbor site, axes `(mpo_left, mpo_right,
        phys_bra, phys_ket)`.

    Returns
    -------
    Tensor
        Left half with axes `(bra_left, internal, mpo_mid, phys_bra)`.
    """
    # b (ket_left), o (mpo_left) and s (phys_ket) are contracted; a (bra_left),
    # α (internal), p (mpo_right) and r (phys_bra) stay free.
    return einsum('aob,bαs,oprs->aαpr', E_left, L, W)


def _right_half(E_right: Tensor, B: Tensor, W: Tensor) -> Tensor:
    """Apply the iMPO to the cheap right factor from the right.

    Parameters
    ----------
    E_right:
        Right environment on the center site's right bond, axes
        `(bra, mpo, ket)`.
    B:
        Cheap right factor, axes `(internal_α, ket_right, phys_ket)`.
    W:
        iMPO tensor of the center site, axes `(mpo_left, mpo_right, phys_bra,
        phys_ket)`.

    Returns
    -------
    Tensor
        Right half with axes `(internal, mpo_mid, bra_right, phys_bra)`.
    """
    # d (ket_right), q (mpo_right) and v (phys_ket) are contracted; α
    # (internal), p (mpo_left), c (bra_right) and u (phys_bra) stay free.
    return einsum('cqd,αdv,pquv->αpcu', E_right, B, W)


def _project_complement_left(left_half: Tensor, A: Tensor) -> Tensor:
    """Project the left half onto the complement of the kept left subspace.

    `A` is left-orthonormal (`A† A = I`), so the projector onto its column
    space is `A A†`, acting on the joint (bra_left, phys_bra) indices of the
    left half. What comes back is the part of the half that the projector
    misses.

    Parameters
    ----------
    left_half:
        Tensor with axes `(bra_left, internal, mpo_mid, phys_bra)`.
    A:
        Left-orthonormal tensor spanning the kept space, axes `(ket_left,
        internal, phys)`.

    Returns
    -------
    Tensor
        Discarded component of the left half, with the same axis layout.
    """
    # proj[a,α,p,r] = Σ_{b,x,z} left_half[x,α,p,z] · A*[x,b,z] · A[a,b,r].
    # The half and the conjugate share x and z, so contracting those two first
    # holds an intermediate of size b·α·p.
    proj = einsum('xαpz,xbz,abr->aαpr', left_half, A.conj(), A)
    return left_half - proj


def _project_complement_right(right_half: Tensor, B: Tensor) -> Tensor:
    """Project the right half onto the complement of the kept right subspace.

    `B` is right-orthonormal (`B B† = I`), so the projector onto its row
    space is `B† B`, acting on the joint (bra_right, phys_bra) indices of the
    right half.

    Parameters
    ----------
    right_half:
        Tensor with axes `(internal, mpo_mid, bra_right, phys_bra)`.
    B:
        Right-orthonormal tensor spanning the kept space, axes `(internal,
        ket_right, phys)`.

    Returns
    -------
    Tensor
        Discarded component of the right half, with the same axis layout.
    """
    # proj[α,p,c,u] = Σ_{b,x,z} right_half[α,p,x,z] · B*[b,x,z] · B[b,c,u].
    proj = einsum('αpxz,bxz,bcu->αpcu', right_half, B.conj(), B)
    return right_half - proj


def _discarded_bulk(left_half_disc: Tensor, right_half_disc: Tensor) -> Tensor:
    """Contract the two discarded halves into the doubly-discarded Θ'_disc.

    Parameters
    ----------
    left_half_disc:
        Discarded left half, axes `(bra_left, internal, mpo_mid, phys_bra)`.
    right_half_disc:
        Discarded right half, axes `(internal, mpo_mid, bra_right, phys_bra)`.

    Returns
    -------
    Tensor
        Θ'_disc with axes `(a, c, r, u)`, the two-site layout of
        `scheme_2s`.
    """
    return einsum('aαpr,αpcu->acru', left_half_disc, right_half_disc)


def _noise_floor(left_half: Tensor, right_half: Tensor) -> float:
    """Absolute size below which a complement direction is cancellation noise.

    Θ'_disc is a difference of quantities of size `‖lh‖ · ‖rh‖`, so that
    product, not Θ'_disc, sets the scale at which its own entries stop being
    meaningful. Anything the SVD reports below this floor was manufactured by
    the subtraction and carries no direction: such vectors are not orthogonal
    to the kept space, and in measured cases overlap it completely.

    The floor is deliberately shared by `_has_complement` and
    `_complement_vectors`. A relative threshold cannot do this job — taken
    against Θ'_disc it shrinks with exactly the quantity whose smallness is
    the thing to detect, which lets a bond whose complement is pure noise pass
    every test and expand on garbage.

    Parameters
    ----------
    left_half:
        Left half-application of the iMPO, before the projection.
    right_half:
        Right half-application of the iMPO, before the projection.

    Returns
    -------
    float
        Absolute threshold on `‖Θ'_disc‖` and on its singular values.
    """
    return _CBE_TOL * left_half.norm() * right_half.norm()


def _has_complement(theta_disc: Tensor, floor: float) -> bool:
    """Report whether Θ'_disc holds a genuine complement direction.

    The test has to be made on Θ'_disc, not on the two discarded halves
    separately. Each half can keep a sizable remainder while their
    contraction cancels it: the surviving sectors of one need not pair with
    those of the other, and then the doubly-discarded tensor vanishes even
    though neither projection removed much. That happens wherever the iMPO
    cannot yet act across the bond, as on a product state.

    It matters because the SVD cannot report it. Handed an exactly zero
    block, `decomp` returns a unit-norm singular vector anyway. The vectors
    that come back lie in the kept space, not in its complement, and
    `oplus`-ing them would duplicate a kept direction, destroy the neighbor's
    isometry, and move the state.

    Parameters
    ----------
    theta_disc:
        The doubly-discarded tensor built from the projected halves.
    floor:
        Absolute threshold from `_noise_floor`.

    Returns
    -------
    bool
        `True` when Θ'_disc is large enough to be signal, not roundoff.
    """
    return theta_disc.norm() > floor


def _complement_vectors(theta_disc: Tensor, k_expand: int, floor: float) -> Tensor:
    """Extract the neighbor's new bond directions from Θ'_disc.

    Decomposes Θ'_disc by truncated SVD and keeps the left singular vectors,
    which are the directions the neighbor's right bond is widened with. The
    right singular vectors are not used: the center tensor is filled in
    exactly against the expanded neighbor instead, so each direction is
    consumed once.

    `flow='<<'` gives the new bond a direction matching the neighbor's right
    bond, the pass always meeting the bond with the neighbor on its left.

    Parameters
    ----------
    theta_disc:
        Doubly-discarded tensor with axes `(a, c, r, u)`, from
        `_discarded_bulk`. Must hold a complement — see `_has_complement`.
    k_expand:
        Largest number of complement vectors extracted, as `nkeep` for the
        SVD.
    floor:
        Absolute threshold from `_noise_floor`. Singular values below it are
        dropped, so the new bond can come back narrower than `k_expand` — or
        empty, which the caller has to read as no complement.

    Returns
    -------
    Tensor
        `N_L` with axes `(bra_left, k, phys)`, left-orthonormal: the new
        right-bond vectors for the neighbor.
    """
    # axes=[0, 2] fuses (a, r) as the row space and (c, u) as the column one.
    # `nkeep` caps the expansion; `thresh` keeps it honest, since a bond
    # whose complement holds fewer than `k_expand` directions still has
    # singular values to report and they are cancellation noise, lying in the
    # kept space and not in the complement. Asking for `k_expand` directions
    # where fewer exist is the ordinary case, not an edge case.
    N_L, _S, _Vh = decomp(
        theta_disc, axes=[0, 2], mode='SVD',
        trunc={'nkeep': k_expand, 'thresh': floor},
        flow='<<', itag=_COMPLEMENT_ITAG,
    )
    # decomp unmerges the left factor back to (a, r, k); the iMPS layout wants
    # the bond on axis 1.
    N_L.permute([0, 2, 1], in_place=True)
    return N_L


def _fill_center(theta: Tensor, left_iso_exp: Tensor) -> Tensor:
    """Represent Θ exactly in the expanded neighbor basis.

    Projects Θ onto the expanded, left-orthonormal neighbor,
    `AC_exp = AL_exp† · Θ`. Since the expanded basis contains the old one,
    the pair `(AL_exp, AC_exp)` describes the same state as before the
    expansion — the new bond directions simply enter with zero amplitude,
    ready for the one-site eigenproblem to fill.

    Parameters
    ----------
    theta:
        Two-site bond tensor with axes `(ket_left, ket_right, phys_neighbor,
        phys_center)`.
    left_iso_exp:
        Expanded left-orthonormal neighbor, axes `(ket_left, ket_right_exp,
        phys)`.

    Returns
    -------
    Tensor
        `AC_exp` with axes `(ket_left_exp, ket_right, phys)`.
    """
    # a (ket_left) and r (phys_neighbor) are contracted against the conjugated
    # neighbor; its expanded right bond b becomes the new left bond of the
    # center tensor.
    return einsum('acru,abr->bcu', theta, left_iso_exp.conj())


# ---------------------------------------------------------------------------
# Bond expansion
# ---------------------------------------------------------------------------

def expand_bond(
    left_iso: Tensor,
    center: Tensor,
    W_neighbor: Tensor,
    W_center: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    k_expand: int,
    alpha: Optional[int],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Widen the bond between a left-orthonormal tensor and a center tensor.

    The bond expanded is the one the two share: the right bond of
    `left_iso` and the left bond of `center`. The procedure is

    1. truncated SVD of `Θ = left_iso ⊗ center` into cheap factors of
       internal bond `alpha`;
    2. factored half-application of the iMPO from each side;
    3. projection of each half onto the complement of its own kept space;
    4. SVD of Θ'_disc into the new bond directions `N_L`;
    5. expansion of `left_iso` by `oplus`, and exact fill-in of `center`;
    6. one step of the left environment over the expanded neighbor.

    Parameters
    ----------
    left_iso:
        Left-orthonormal tensor of the site behind the bond, axes `(ket_left,
        ket_right, phys)`.
    center:
        Center tensor of the site in front of the bond, axes `(ket_left,
        ket_right, phys)`.
    W_neighbor:
        iMPO tensor of the neighbor site.
    W_center:
        iMPO tensor of the center site.
    E_left:
        Left environment on the neighbor's left bond, which the expansion
        does not touch.
    E_right:
        Right environment on the center site's right bond, which the
        expansion does not touch either.
    k_expand:
        Largest number of complement directions added to the bond.
    alpha:
        Internal bond dimension of the cheap two-site factors. `None` keeps
        the full bond.

    Returns
    -------
    left_iso_exp:
        Neighbor with its right bond widened, still left-orthonormal.
    center_exp:
        Center tensor re-expressed on the widened bond, describing the same
        state as `center` did.
    E_left_exp:
        Left environment on the widened bond, exact for `left_iso_exp`.
    """
    # Step 1: cheap factored form of Θ, isometric on the center side.
    theta = einsum('abr,bcs->acrs', left_iso, center)
    L, B = _cheap_factors(theta, alpha)

    # Step 2: factored half-application of the iMPO.
    lh = _left_half(E_left, L, W_neighbor)
    rh = _right_half(E_right, B, W_center)

    # Step 3: discarded-space projection. Each end projects against the tensor
    # that spans exactly its own kept space: the neighbor itself on the left,
    # whose whole column space the complement must miss for the expanded
    # neighbor to stay isometric, which makes the fill-in exact, and the
    # orthonormalized center tensor on the right.
    lh_disc = _project_complement_left(lh, left_iso)
    rh_disc = _project_complement_right(rh, _center_isometry_right(center))
    theta_disc = _discarded_bulk(lh_disc, rh_disc)

    # A bond with no complement — saturated, or simply out of the iMPO's reach
    # from the current state — has nothing to expand into; leave it as it is
    # and hand back the environment the one-site step would have used anyway.
    floor = _noise_floor(lh, rh)
    if not _has_complement(theta_disc, floor):
        return left_iso, center, step_left_env(E_left, left_iso, W_neighbor)

    # Step 4: new bond directions from the SVD of Θ'_disc.
    N_L = _complement_vectors(theta_disc, k_expand, floor)

    # Everything the SVD offered sat below the floor, which leaves no
    # direction worth adding even though Θ'_disc as a whole cleared it.
    if N_L.indices[1].dim == 0:
        return left_iso, center, step_left_env(E_left, left_iso, W_neighbor)

    # Step 5: widen the neighbor's right bond (axis 1), then re-express Θ in
    # that basis.
    left_iso_exp = oplus(left_iso, N_L, axes=[1])
    center_exp = _fill_center(theta, left_iso_exp)

    # Step 6: left environment on the widened bond. One step over the expanded
    # neighbor keeps the cross blocks between it and the complement, so the
    # effective Hamiltonian of the solve that follows is exact.
    E_left_exp = step_left_env(E_left, left_iso_exp, W_neighbor)

    return left_iso_exp, center_exp, E_left_exp


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_center(
    AC: Tensor,
    itag: str,
    trunc: Optional[dict] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Split a center tensor across its left bond, truncating as it goes.

    The bipartition puts the left bond alone on one side, which is the cut
    the widened bond sits on, so this is the step that cuts it back down. The
    three factors come back ready to use as they are: `U` rotates the
    neighbor onto the truncated bond, `S` is the bond matrix, already
    diagonal and so already in the Schmidt basis `iMPS.set_canonical`
    expects, and `Vh` is the right-orthonormal tensor of the site.

    The gauge step of the one-site scheme reproduces those factors unchanged,
    since the orthonormal polar factor of `S` is the identity and that of
    `AL·S` is `AL`. So the side that was just solved carries no gradient, and
    nothing has to be divided by a bond matrix.

    Parameters
    ----------
    AC:
        Center tensor with axes `(ket_left, ket_right, phys)`, its left bond
        the widened one.
    itag:
        itag for the truncated bond, which is the itag the cell already uses
        for this bond.
    trunc:
        Truncation options forwarded to `decomp` (`nkeep`, `thresh`). `None`
        keeps the widened bond whole, so the expansion is permanent and not
        merely a wider space to solve in.

    Returns
    -------
    rotation:
        `U` with axes `(widened_bond, new_bond)`, to be absorbed into the
        neighbor.
    bond:
        Diagonal bond matrix on the new bond.
    right_iso:
        Right-orthonormal tensor with axes `(new_bond, ket_right, phys)`.
    """
    # '<<' sends the new bond out of the left factor, the same direction
    # convention `split_two_site` and the gauge iteration both follow.
    return decomp(AC, axes=0, mode='SVD', flow='<<', itag=itag, trunc=trunc)


def discarded_weight(AC: Tensor, trunc: Optional[dict] = None) -> float:
    """Return the weight `split_center` would discard from `AC`.

    Repeats the bipartition of `split_center` and asks for the truncation
    report that `decomp` does not surface. The second decomposition is
    deliberate: it runs once per bond per iteration, against a Davidson solve
    that has already cost many applications of the effective Hamiltonian.

    Parameters
    ----------
    AC:
        Center tensor, as passed to `split_center`.
    trunc:
        The same truncation options, so the number describes the split that
        was actually made.

    Returns
    -------
    float
        Relative sum of the squared singular values dropped, in `[0, 1)`.
        Zero when `trunc` is `None` or nothing is cut.
    """
    merged, _ = merge_axes(AC, [1, 2], merged_tag=_WEIGHT_ITAG)
    *_, info = svd(merged, axis=0, trunc=trunc, requires_info=True)
    return float(info['discarded_weight'])
