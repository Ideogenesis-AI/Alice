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


"""CBE complement isometry for 1-site-plus TgTRG.

Implements the controlled bond expansion (CBE) step the 1-site-plus cooling
scheme is built around. Before each local propagation, a complement isometry
is computed from a cheap, factored approximation of the 2-site gradient: `H`
is applied independently from the left and from the right to a truncated
2-site tensor; the current (kept) subspace is projected out of each half; and
the resulting "doubly-discarded" outer product is decomposed by SVD into new
basis vectors that expand the bond of `ρ`. Those directions are what the
subsequent 1-site propagator can flow into, so the bond adapts as it does
under the 2-site scheme while every local exponential stays 1-site sized.

The construction is the one in `alice.algorithm.tdvp.complement`, carried
from a real-time MPS to a cooling density matrix. Both are projector
splittings of the same kind, and the expansion feeds straight into a local
propagator in each, so the reasoning transfers whole; only the index layout
changes.

The spectator physical axis
---------------------------
`ρ` is read as a super-MPS over the doubled physical space, so every site
tensor carries two physical axes, and `H` meets only one of them —
`(Hρ)_{r,s} = Σ_x H_{r,x} ρ_{x,s}`, as everywhere else in this package. The
half-applications of `H` below therefore contract `phys_in` and carry
`phys_out` through untouched.

The complement, by contrast, is sought over the *full* local space, `(bond,
phys_in, phys_out)`. That is the space the bond has to span for `ρ` to
represent what cooling turns it into, and the space the subsequent 1-site
propagator moves within. Searching only the half `H` touches would blind the
expansion to exactly the direction the spectator carries.

Exact state preservation
------------------------
The expansion must not move `ρ`. Concatenating complement vectors onto *both*
tensors would, since `Θ = M_i ⊗ M_i1` picks up the extra outer product. Only
the neighbor is expanded by `oplus`, and the center tensor is then *filled in*
exactly by projecting Θ onto the expanded neighbor basis,
`M_i_exp = Θ · M_i1_exp†`. Because the expanded basis contains the old one,
this reproduces Θ exactly; it is the same exact fill-in as in
`alice.algorithm.xtrg.complement`. With a right-isometric `M_i1_exp` the
outcome is the old center tensor padded by zero columns, and writing it as a
projection instead of a zero padding keeps it exact even where the complement
is not perfectly orthogonal to the kept space.

Preservation matters more here than it does in real time. A cooling run reads
`log Z(β)` off `ρ` at every step, and an expansion that moved the state would
move the free energy with it, at every bond of every sweep.

It also takes one step the real-time scheme does without. The projection is
exact against an isometric neighbor, and the stacked neighbor is isometric
only as far as the complement is orthogonal to the kept space — which, out of
a near-identity `ρ(τ₀)`, it is only to about `1e-10`. `_regauge_right` and
`_regauge_left` re-decompose the stack to recover an exact isometry before
the fill-in; see their docstrings for where the lost digits come from.

Exact expanded environment
--------------------------
The environment facing the expanded bond is built by a single `step_*_env`
call on the expanded neighbor, which keeps the cross blocks between `M_i1`
and the complement. Assembling it block-diagonally from the two pieces
separately, as DMRG does, drops those blocks: cheap for an initial guess, but
a perturbed `H_eff` would perturb the cooling trajectory itself.

Orthonormal projectors
----------------------
Each bond end projects out its own kept space, and the two ends call for
different tensors. On the neighbor side the site tensor itself is used: it is
exactly isometric there, and its row (or column) space is the full space the
complement has to avoid for the expanded neighbor to stay isometric. On the
center side a 1-site split of the center tensor supplies an orthonormal basis
of the same space the center tensor spans, since `M M†` is a projector only
for an isometry and in a cooling sweep the center tensor is not one.

What must *not* be used on the center side is the isometric factor of Θ, even
though it is orthonormal and spans the center tensor's column space.
Untruncated it spans more: `decomp` keeps every sector at full dimension
regardless of rank, so that factor can cover the entire (bond, physical)
space and project the complement away entirely, leaving Θ'_disc at roundoff
where a real complement exists. The split of the center tensor alone cannot
overreach that way, its internal bond being the bond itself.

Index convention (shared with `environ.py`, `scheme_1s.py`, `scheme_2s.py`):

- `a, b, c, d` — virtual bond indices of `ρ` (bra or ket side).
- `α`          — internal reduced bond of the truncated 2-site tensor.
- `o, p, q`    — virtual bond indices of the Hamiltonian MPO.
- `r, x`       — `phys_in` of site i, bra and ket.
- `s`          — `phys_out` of site i, a spectator.
- `u, y`       — `phys_in` of site i+1, bra and ket.
- `v`          — `phys_out` of site i+1, a spectator.

Tensor shapes used in this module:

- `left_half  [a, α, p, r, s]` — (bra_left, internal, mpo_mid, phys_in, phys_out)
- `right_half [α, p, c, u, v]` — (internal, mpo_mid, bra_right, phys_in, phys_out)
- `N_L        [a, k, r, s]`    — complement vectors for the right bond of site i.
- `N_R        [k, c, u, v]`    — complement vectors for the left bond of site i+1.

As in `environ.py`, `ρ` and the Hamiltonian share the `W{i}` bond itag
family, so every contraction below is a chain of explicit two-tensor
`einsum` calls.
"""

from __future__ import annotations

from typing import Optional, Tuple

from nicole import Tensor, decomp, einsum, oplus

from .environ import step_left_env, step_right_env
from .scheme_1s import split_backward, split_forward
from .scheme_2s import build_bulk

# Temporary itag assigned to the truncated internal bond in the cheap factors.
_TRUNC_ITAG: str = '_cbe_trunc_'

# Itag assigned to the internal bond of the center tensor's own split, kept
# distinct from _TRUNC_ITAG so that a projection can carry both at once.
_CENTER_ITAG: str = '_cbe_center_'

# Itag assigned to the complement bond produced by the SVD of Θ'_disc.
_COMPLEMENT_ITAG: str = '_cbe_comp_'

# Relative size below which a complement direction counts as roundoff rather
# than as signal. A saturated bond — one whose dimension already equals
# `dim_outer · d²` on either end — has no complement at all, and neither has a
# bond `H` cannot yet act across. What the SVD returns there is noise, or in
# the exactly vanishing case an arbitrary unit vector lying in the kept space,
# and either would stop the expanded neighbor from being isometric and move
# `ρ`. The tolerance is taken against `_noise_floor`, never against Θ'_disc
# itself.
_CBE_TOL: float = 1e-10


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _center_isometry_forward(M_i: Tensor) -> Tensor:
    """Orthonormalize the center tensor's column space for a forward sweep.

    Returns the isometric factor of the 1-site forward split of the center
    tensor, which spans exactly what the center tensor spans over its (left
    bond, phys_in, phys_out) indices — the kept space the left end of the
    bond has to project out. The bond tensor the split also produces is
    discarded; it is the non-orthonormal part that makes the center tensor
    itself unusable here.

    Parameters
    ----------
    M_i:
        Center tensor, axes `(ket_left, ket_right, phys_in, phys_out)`.

    Returns
    -------
    Tensor
        Left-isometric tensor with axes
        `(ket_left, internal, phys_in, phys_out)`.
    """
    A, _C = split_forward(M_i, _CENTER_ITAG)

    return A


def _center_isometry_backward(M_i: Tensor) -> Tensor:
    """Orthonormalize the center tensor's row space for a backward sweep.

    Mirror of `_center_isometry_forward`, over the (right bond, phys_in,
    phys_out) indices.

    Parameters
    ----------
    M_i:
        Center tensor, axes `(ket_left, ket_right, phys_in, phys_out)`.

    Returns
    -------
    Tensor
        Right-isometric tensor with axes
        `(internal, ket_right, phys_in, phys_out)`.
    """
    _C, B = split_backward(M_i, _CENTER_ITAG)

    return B


def _cheap_factors_forward(theta: Tensor, alpha: Optional[int]) -> Tuple[Tensor, Tensor]:
    """Factorize Θ for a forward sweep.

    The pair is a cheap factored form of Θ whose internal bond is at most
    `alpha`, so the half-applications of `H` below scale with `alpha` instead
    of the full bond dimension. The direction of the split follows the sweep,
    as in the 2-site scheme.

    Parameters
    ----------
    theta:
        2-site bond tensor with the axis layout `build_bulk` returns.
    alpha:
        Max bond dimension for the truncated internal bond. `None` keeps all
        singular values (equivalent to using the full 2-site tensor).

    Returns
    -------
    Tensor
        `A` with axes `(ket_left, internal_α, phys_in, phys_out)`,
        left-isometric.
    Tensor
        `R = S·Vh` with axes `(internal_α, ket_right, phys_in, phys_out)`.
    """
    trunc = {'nkeep': alpha} if alpha is not None else None
    # mode='UR' → A is left-isometric and R = S·Vh carries the singular
    # values, the same split the 2-site scheme uses for a forward sweep. The
    # bipartition keeps both physical axes of site i on the left.
    A, R = decomp(theta, axes=[0, 2, 3], mode='UR', trunc=trunc)

    # A comes back unmerged as (ket_left, phys_in, phys_out, internal);
    # restore the density-matrix axis order.
    A.permute([0, 3, 1, 2], in_place=True)
    A.retag(1, _TRUNC_ITAG)
    R.retag(0, _TRUNC_ITAG)

    return A, R


def _cheap_factors_backward(theta: Tensor, alpha: Optional[int]) -> Tuple[Tensor, Tensor]:
    """Factorize Θ for a backward sweep.

    Mirror of `_cheap_factors_forward`: the singular values stay on the left
    factor, so the isometric factor is the one facing the sweep's direction
    of travel.

    Parameters
    ----------
    theta:
        2-site bond tensor with the axis layout `build_bulk` returns, over
        sites i-1 and i.
    alpha:
        Max bond dimension for the truncated internal bond. `None` keeps all
        singular values.

    Returns
    -------
    Tensor
        `L = U·S` with axes `(ket_left, internal_α, phys_in, phys_out)`.
    Tensor
        `B` with axes `(internal_α, ket_right, phys_in, phys_out)`,
        right-isometric.
    """
    trunc = {'nkeep': alpha} if alpha is not None else None
    # mode='LV' → L = U·S carries the singular values and B is right-isometric.
    L, B = decomp(theta, axes=[0, 2, 3], mode='LV', trunc=trunc)

    L.permute([0, 3, 1, 2], in_place=True)
    L.retag(1, _TRUNC_ITAG)
    B.retag(0, _TRUNC_ITAG)

    return L, B


def _left_half(E_left: Tensor, M_trunc: Tensor, W: Tensor) -> Tensor:
    """Compute the left half of `H` acting on the truncated ket M̃_i.

    Contracts `E_left`, M̃_i (truncated ket), and `W` (the Hamiltonian at site
    i), leaving the internal reduced bond α and the MPO bond p free. Together
    with `_right_half`, this allows `H·Θ̃` to be reconstructed as
    `einsum('aαprs,αpcuv->acrsuv', left_half, right_half)`.

    Parameters
    ----------
    E_left:
        Left environment with axes `(bra_left=a, mpo_left=o, ket_left=b)`.
    M_trunc:
        Truncated ket tensor M̃_i with axes
        `(ket_left=b, internal_α=α, phys_in=x, phys_out=s)`.
    W:
        Hamiltonian tensor at site i with axes
        `(mpo_left=o, mpo_right=p, phys_bra=r, phys_ket=x)`.

    Returns
    -------
    Tensor
        Left half with axes
        `(bra_left=a, internal=α, mpo_mid=p, phys_in=r, phys_out=s)`.
    """
    # Step 1: E_left(a,o,b) × M̃(b,α,x,s) → (a,o,α,x,s), contracting 'b'.
    t1 = einsum('aob,bαxs->aoαxs', E_left, M_trunc)
    # Step 2: t1(a,o,α,x,s) × W(o,p,r,x) → (a,α,p,r,s), contracting 'o','x'.
    return einsum('aoαxs,oprx->aαprs', t1, W)


def _right_half(E_right: Tensor, M_trunc: Tensor, W: Tensor) -> Tensor:
    """Compute the right half of `H` acting on the truncated ket M̃_{i+1}.

    Contracts `E_right`, M̃_{i+1} (truncated ket), and `W` (the Hamiltonian at
    site i+1), leaving the internal reduced bond α and the MPO bond p free.

    Parameters
    ----------
    E_right:
        Right environment with axes
        `(bra_right=c, mpo_right=q, ket_right=d)`.
    M_trunc:
        Truncated ket tensor M̃_{i+1} with axes
        `(internal_α=α, ket_right=d, phys_in=y, phys_out=v)`.
    W:
        Hamiltonian tensor at site i+1 with axes
        `(mpo_left=p, mpo_right=q, phys_bra=u, phys_ket=y)`.

    Returns
    -------
    Tensor
        Right half with axes
        `(internal=α, mpo_mid=p, bra_right=c, phys_in=u, phys_out=v)`.
    """
    # Step 1: E_right(c,q,d) × M̃(α,d,y,v) → (c,q,α,y,v), contracting 'd'.
    t1 = einsum('cqd,αdyv->cqαyv', E_right, M_trunc)
    # Step 2: t1(c,q,α,y,v) × W(p,q,u,y) → (α,p,c,u,v), contracting 'q','y'.
    return einsum('cqαyv,pquy->αpcuv', t1, W)


def _project_complement_left(left_half: Tensor, A: Tensor) -> Tensor:
    """Project `left_half` onto the complement of the kept left subspace.

    `A` is left-isometric (`A† A = I`), so the projector onto its column space
    is `P_L = A A†`, acting on the joint (bra_left, phys_in, phys_out) indices
    of `left_half`. The discarded part is `(I − P_L) left_half`.

    Parameters
    ----------
    left_half:
        Tensor with axes `(bra_left=a, internal=α, mpo_mid=p, phys_in=r,
        phys_out=s)`.
    A:
        Left-isometric tensor spanning the kept space, axes
        `(ket_left=a, internal=b, phys_in=r, phys_out=s)`.

    Returns
    -------
    Tensor
        Discarded component of `left_half`, same axes as `left_half`.
    """
    # proj[a,α,p,r,s] = Σ_{b,x,z,w} left_half[x,α,p,z,w] · A*[x,b,z,w] · A[a,b,r,s]
    # left_half and A* share x, z and w → contract them first, which keeps the
    # intermediate at b·α·p.
    t1 = einsum('xαpzw,xbzw->αpb', left_half, A.conj())
    proj = einsum('αpb,abrs->aαprs', t1, A)

    return left_half - proj


def _project_complement_right(right_half: Tensor, B: Tensor) -> Tensor:
    """Project `right_half` onto the complement of the kept right subspace.

    `B` is right-isometric (`B B† = I`), so the projector onto its row space
    is `P_R = B† B`, acting on the joint (bra_right, phys_in, phys_out)
    indices of `right_half`. The discarded part is `(I − P_R) right_half`.

    Parameters
    ----------
    right_half:
        Tensor with axes `(internal=α, mpo_mid=p, bra_right=c, phys_in=u,
        phys_out=v)`.
    B:
        Right-isometric tensor spanning the kept space, axes
        `(internal=b, ket_right=c, phys_in=u, phys_out=v)`.

    Returns
    -------
    Tensor
        Discarded component of `right_half`, same axes as `right_half`.
    """
    # proj[α,p,c,u,v] = Σ_{b,x,z,w} right_half[α,p,x,z,w] · B*[b,x,z,w] · B[b,c,u,v]
    t1 = einsum('αpxzw,bxzw->αpb', right_half, B.conj())
    proj = einsum('αpb,bcuv->αpcuv', t1, B)

    return right_half - proj


def _discarded_bulk(left_half_disc: Tensor, right_half_disc: Tensor) -> Tensor:
    """Contract the two discarded halves into the doubly-discarded Θ'_disc.

    Parameters
    ----------
    left_half_disc:
        Discarded left half with axes `(bra_left=a, internal=α, mpo_mid=p,
        phys_in=r, phys_out=s)`.
    right_half_disc:
        Discarded right half with axes `(internal=α, mpo_mid=p, bra_right=c,
        phys_in=u, phys_out=v)`.

    Returns
    -------
    Tensor
        Θ'_disc with axes `(a, c, r, s, u, v)`, the layout `build_bulk`
        returns.
    """
    return einsum('aαprs,αpcuv->acrsuv', left_half_disc, right_half_disc)


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
    the thing to detect, letting a bond whose complement is pure noise pass
    every test and expand on garbage.

    Parameters
    ----------
    left_half:
        Left half-application of `H`, before the projection.
    right_half:
        Right half-application of `H`, before the projection.

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
    though neither projection removed much. That happens wherever `H` cannot
    yet act across the bond.

    It matters because the SVD cannot report it. Handed an exactly zero block,
    `decomp` returns a unit-norm singular vector anyway. The vectors that come
    back lie in the kept space, not its complement, and `oplus`-ing them would
    duplicate a kept direction, destroy the neighbor's isometry, and move `ρ`.

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


def _complement_vectors(
    theta_disc: Tensor,
    k_expand: int,
    flow: str,
    floor: float,
) -> Tuple[Tensor, Tensor]:
    """Extract complement vectors from the doubly-discarded 2-site tensor.

    Decomposes Θ'_disc by truncated SVD over the same bipartition the 2-site
    splits use, so both physical axes of a site stay with that site's bond.
    Both singular-vector factors are isometric; a sweep expands the neighbor
    with the one facing it and fills the center tensor in exactly, so each
    direction consumes only one of the two.

    The required `flow` depends on the sweep direction because `theta_disc`
    carries different index directions in each case:

    - Forward sweep: use `flow='<<'`. The index directions of `theta_disc`
      produce N_R with a new-bond direction matching the right-isometric
      neighbor's left bond.
    - Backward sweep: use `flow='>>'`. Directions of `theta_disc` are
      reversed relative to the forward case, so the opposite flow is needed
      for N_L to match the left-isometric neighbor's right bond.

    Parameters
    ----------
    theta_disc:
        Doubly-discarded tensor with axes `(a, c, r, s, u, v)`, from
        `_discarded_bulk`. Must hold a complement — see `_has_complement`.
    k_expand:
        Maximum number of complement vectors to extract (nkeep for the SVD).
    flow:
        Arrow-flow string passed to `decomp`. Use `'<<'` for a forward sweep
        and `'>>'` for a backward sweep.
    floor:
        Absolute threshold from `_noise_floor`. Singular values below it are
        dropped, so the new bond can come back smaller than `k_expand` — or
        empty, which the caller must treat as no complement.

    Returns
    -------
    Tensor
        N_L with axes `(bra_left=a, k=new_bond, phys_in=r, phys_out=s)` —
        left-isometric, the new right-bond vectors for site i.
    Tensor
        N_R with axes `(k=new_bond, bra_right=c, phys_in=u, phys_out=v)` —
        right-isometric, the new left-bond vectors for site i+1.
    """
    # `nkeep` caps the expansion; `thresh` keeps it honest, since a bond whose
    # complement holds fewer than `k_expand` directions still has singular
    # values to report and they are cancellation noise, lying in the kept
    # space and not in the complement. Asking for `k_expand` directions where
    # one exists is the ordinary case, not an edge case.
    U, _S, Vh = decomp(
        theta_disc, axes=[0, 2, 3], mode='SVD',
        trunc={'nkeep': k_expand, 'thresh': floor},
        flow=flow, itag=_COMPLEMENT_ITAG,
    )
    # After decomp + unmerge: U has axes (a, r, s, k); permute to (a, k, r, s).
    U.permute([0, 3, 1, 2], in_place=True)   # N_L: (a, k, r, s)
    # Vh has axes (k, c, u, v). N_R = Vh.
    return U, Vh


def _regauge_right(M_exp: Tensor, itag: str) -> Tensor:
    """Restore exact right-isometry of a neighbor just expanded by `oplus`.

    `oplus` stacks the complement onto the kept basis, so the expanded tensor
    is isometric only as far as the two blocks are orthogonal. They are
    orthogonal *by construction* — the rows of Θ'_disc are combinations of
    rows of `rh_disc`, which the projection cleared of the kept space — but
    only to the accuracy Θ'_disc itself carries, and that is where a cooling
    run differs from a real-time one.

    Θ'_disc is a cancellation: its entries are correct to about `eps ‖lh‖
    ‖rh‖` in absolute terms, so the singular vectors it yields lose roughly
    `eps ‖lh‖ ‖rh‖ / ‖Θ'_disc‖` digits of orthogonality. A run cooling out of
    `ρ(τ₀) ≈ 1` meets the bad end of that ratio at once: `H` barely moves a
    near-identity `ρ` off the space it already spans, the ratio reaches `1e-6`
    at `β` of order `τ₀`, and the complement comes back orthogonal to only
    `1e-10`. That is enough to cost the expanded neighbor its isometry, the
    fill-in its exactness, and `log Z` the accuracy the rest of the algorithm
    is careful to keep.

    Re-decomposing the stacked tensor fixes it outright. The new isometry
    spans the same space, since a decomposition spans what it decomposes, and
    that space still contains the old basis — so the fill-in against it stays
    exact whatever the complement's own orthogonality was. One small
    decomposition per bond buys that, against a Krylov propagation on the
    same tensor.

    Parameters
    ----------
    M_exp:
        Expanded neighbor from `oplus`, axes
        `(ket_left_exp, ket_right, phys_in, phys_out)`.
    itag:
        itag to carry on the expanded bond, the one the neighbor had before
        the expansion.

    Returns
    -------
    Tensor
        Right-isometric tensor spanning the same space, same axis layout.
    """
    # The same split `scheme_1s.split_backward` makes, with the bond tensor
    # dropped: it is the non-orthonormal part being removed.
    _C, B = decomp(M_exp, axes=0, flow='<<', mode='LV')
    B.retag(0, itag)

    return B


def _regauge_left(M_exp: Tensor, itag: str) -> Tensor:
    """Restore exact left-isometry of a neighbor just expanded by `oplus`.

    Mirror of `_regauge_right`, over the (right bond, phys_in, phys_out)
    indices.

    Parameters
    ----------
    M_exp:
        Expanded neighbor from `oplus`, axes
        `(ket_left, ket_right_exp, phys_in, phys_out)`.
    itag:
        itag to carry on the expanded bond.

    Returns
    -------
    Tensor
        Left-isometric tensor spanning the same space, same axis layout.
    """
    _C, A = decomp(M_exp, axes=1, flow='<<', mode='LV')
    A.permute([1, 0, 2, 3], in_place=True)
    A.retag(1, itag)

    return A


def _fill_center_forward(theta: Tensor, M_i1_exp: Tensor) -> Tensor:
    """Represent Θ exactly in the expanded right basis (forward sweep).

    Projects Θ onto the expanded, right-isometric neighbor,
    `M_i_exp = Θ · M_i1_exp†`. Since the expanded basis contains the old one,
    the pair `(M_i_exp, M_i1_exp)` represents the same `ρ` as before the
    expansion — the new bond directions simply enter with zero amplitude,
    ready for the 1-site propagator to populate.

    Parameters
    ----------
    theta:
        2-site bond tensor with the axis layout `build_bulk` returns.
    M_i1_exp:
        Expanded right-isometric tensor at site i+1, axes
        `(ket_left_exp, ket_right, phys_in, phys_out)`.

    Returns
    -------
    Tensor
        `M_i_exp` with axes `(ket_left, ket_right_exp, phys_in, phys_out)`.
    """
    # c (ket_right), u and v (both physical axes of site i+1) are contracted
    # against the conjugated neighbor; its expanded left bond b becomes the
    # new right bond of site i.
    return einsum('acrsuv,bcuv->abrs', theta, M_i1_exp.conj())


def _fill_center_backward(theta: Tensor, M_im1_exp: Tensor) -> Tensor:
    """Represent Θ exactly in the expanded left basis (backward sweep).

    Mirror of `_fill_center_forward`: projects Θ onto the expanded,
    left-isometric neighbor at site i-1.

    Parameters
    ----------
    theta:
        2-site bond tensor with the axis layout `build_bulk` returns, over
        sites i-1 and i.
    M_im1_exp:
        Expanded left-isometric tensor at site i-1, axes
        `(ket_left, ket_right_exp, phys_in, phys_out)`.

    Returns
    -------
    Tensor
        `M_i_exp` with axes `(ket_left_exp, ket_right, phys_in, phys_out)`.
    """
    # a (ket_left), r and s (both physical axes of site i-1) are contracted
    # against the conjugated neighbor; its expanded right bond b becomes the
    # new left bond of site i.
    return einsum('acrsuv,abrs->bcuv', theta, M_im1_exp.conj())


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
    2. Factored half-application of `H`: `left_half` from `E_left` and the
       left factor; `right_half` from `E_right` and the right factor.
    3. Discarded-space projection: the left half against the orthonormalized
       center tensor, the right half against the right-isometric `M_i1`
       itself.
    4. SVD of Θ'_disc → N_R, the new left-bond vectors for `M_i1`.
    5. Expansion of `M_i1` via `oplus`, and exact fill-in of the center tensor.
    6. Expansion of the right environment at site i from `E_right[i+1]`.

    Parameters
    ----------
    M_i:
        Center tensor at site i, axes
        `(ket_left, ket_right, phys_in, phys_out)`.
    M_i1:
        Right-isometric tensor of `ρ` at site i+1, same layout.
    W_i:
        Hamiltonian tensor at site i, axes
        `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    W_i1:
        Hamiltonian tensor at site i+1, same layout.
    E_left_i:
        Left environment at site i, axes `(bra_left, mpo_left, ket_left)`.
    E_right_i1:
        Right environment at site i+1, axes
        `(bra_right, mpo_right, ket_right)`.
    k_expand:
        Maximum number of complement vectors to add to the bond.
    alpha:
        Truncated internal bond dimension for Θ̃. `None` keeps the full bond.

    Returns
    -------
    Tensor
        `M_i_exp` — center tensor in the expanded basis, axes
        `(ket_left, ket_right_exp, phys_in, phys_out)`.
    Tensor
        `M_i1_exp` — expanded `M_i1` with axes
        `(ket_left_exp, ket_right, phys_in, phys_out)`.
    Tensor
        `E_right_i_exp` — expanded right environment at site i, axes
        `(bra_right_exp, mpo_right, ket_right_exp)`.
    """
    # Step 1: cheap factored form of Θ, isometric on the center side.
    theta = build_bulk(M_i, M_i1)
    A, R = _cheap_factors_forward(theta, alpha)

    # Step 2: factored half-application of H.
    lh = _left_half(E_left_i, A, W_i)
    rh = _right_half(E_right_i1, R, W_i1)

    # Step 3: discarded-space projection. Each end projects against the tensor
    # that spans exactly its own kept space: the orthonormalized center tensor
    # on the left, and M_i1 itself on the right, whose whole row space the
    # complement must miss for the expanded neighbor to stay isometric, making
    # the fill-in below exact.
    lh_disc = _project_complement_left(lh, _center_isometry_forward(M_i))
    rh_disc = _project_complement_right(rh, M_i1)
    theta_disc = _discarded_bulk(lh_disc, rh_disc)

    # A bond with no complement — saturated, or simply out of H's reach from
    # the current ρ — has nothing to expand into; leave it as it is and hand
    # back the environment the 1-site step would have used anyway.
    floor = _noise_floor(lh, rh)
    if not _has_complement(theta_disc, floor):
        return M_i, M_i1, step_right_env(E_right_i1, M_i1, W_i1)

    # Step 4: complement vectors from the SVD of Θ'_disc. Only N_R is used:
    # the center tensor is filled in exactly instead of padded with N_L.
    # flow='<<' gives N_R a new-bond direction matching M_i1's left bond.
    _N_L, N_R = _complement_vectors(theta_disc, k_expand, flow='<<', floor=floor)

    # Everything the SVD offered sat below the floor, which leaves no
    # direction worth adding even though Θ'_disc as a whole cleared it.
    if N_R.indices[0].dim == 0:
        return M_i, M_i1, step_right_env(E_right_i1, M_i1, W_i1)

    # Step 5: expand M_i1's left bond (axis 0) and re-gauge the stack to an
    # exact isometry, then re-express Θ in that basis.
    M_i1_exp = _regauge_right(oplus(M_i1, N_R, axes=[0]), M_i1.itags[0])
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
    2. Factored half-application of `H` using `E_left[i-1]` and `E_right[i]`.
    3. Discarded-space projection: the right half against the orthonormalized
       center tensor, the left half against the left-isometric `M_im1` itself.
    4. SVD of Θ'_disc → N_L, the new right-bond vectors for `M_im1`.
    5. Expansion of `M_im1` via `oplus`, and exact fill-in of the center
       tensor.
    6. Expansion of the left environment at site i from `E_left[i-1]`.

    Parameters
    ----------
    M_i:
        Center tensor at site i, axes
        `(ket_left, ket_right, phys_in, phys_out)`.
    M_im1:
        Left-isometric tensor of `ρ` at site i-1, same layout.
    W_i:
        Hamiltonian tensor at site i, axes
        `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    W_im1:
        Hamiltonian tensor at site i-1, same layout.
    E_left_im1:
        Left environment at site i-1, axes
        `(bra_left, mpo_left, ket_left)`.
    E_right_i:
        Right environment at site i, axes
        `(bra_right, mpo_right, ket_right)`.
    k_expand:
        Maximum number of complement vectors to add to the bond.
    alpha:
        Truncated internal bond dimension for Θ̃. `None` keeps the full bond.

    Returns
    -------
    Tensor
        `M_i_exp` — center tensor in the expanded basis, axes
        `(ket_left_exp, ket_right, phys_in, phys_out)`.
    Tensor
        `M_im1_exp` — expanded `M_im1` with axes
        `(ket_left, ket_right_exp, phys_in, phys_out)`.
    Tensor
        `E_left_i_exp` — expanded left environment at site i, axes
        `(bra_left_exp, mpo_left, ket_left_exp)`.
    """
    # Step 1: cheap factored form of Θ = M_im1 ⊗ M_i, isometric on the center
    # side.
    theta = build_bulk(M_im1, M_i)
    L, B = _cheap_factors_backward(theta, alpha)

    # Step 2: factored half-application of H.
    lh = _left_half(E_left_im1, L, W_im1)
    rh = _right_half(E_right_i, B, W_i)

    # Step 3: discarded-space projection. Mirroring the forward sweep, the
    # left end projects against M_im1, whose column space is the whole kept
    # space the complement has to miss, and the center end against its own
    # tensor orthonormalized.
    lh_disc = _project_complement_left(lh, M_im1)
    rh_disc = _project_complement_right(rh, _center_isometry_backward(M_i))
    theta_disc = _discarded_bulk(lh_disc, rh_disc)

    # A bond with no complement has nothing to expand into; leave it as it is
    # and hand back the environment the 1-site step would have used anyway.
    floor = _noise_floor(lh, rh)
    if not _has_complement(theta_disc, floor):
        return M_i, M_im1, step_left_env(E_left_im1, M_im1, W_im1)

    # Step 4: complement vectors. Only N_L is used here, for the same reason
    # N_R alone is used in the forward sweep.
    # flow='>>' gives N_L a new-bond direction matching M_im1's right bond.
    N_L, _N_R = _complement_vectors(theta_disc, k_expand, flow='>>', floor=floor)

    # Every direction fell below the floor, as in the forward sweep.
    if N_L.indices[1].dim == 0:
        return M_i, M_im1, step_left_env(E_left_im1, M_im1, W_im1)

    # Step 5: expand M_im1's right bond (axis 1) and re-gauge the stack to an
    # exact isometry, then re-express Θ in that basis.
    M_im1_exp = _regauge_left(oplus(M_im1, N_L, axes=[1]), M_im1.itags[1])
    M_i_exp = _fill_center_backward(theta, M_im1_exp)

    # Step 6: expanded left environment at site i, exact for the same reason
    # as the right environment in the forward sweep.
    E_left_i_exp = step_left_env(E_left_im1, M_im1_exp, W_im1)

    return M_i_exp, M_im1_exp, E_left_i_exp
