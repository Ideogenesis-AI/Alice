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


"""1-site TDVP update scheme, including the zero-site back-evolution.

This module contains all operations specific to the 1-site update:

- `matvec_1s` / `evolve_1site`: apply `exp(-i·dt·H_eff)` to a single site
  tensor, where `H_eff = E_left ⊗ W ⊗ E_right`.
- `matvec_0s` / `evolve_bond`: apply `exp(-i·dt·H_0)` to a bond tensor, where
  `H_0 = E_left ⊗ E_right` closes directly over the shared MPO bond.
- `split_forward` / `split_backward`: peel an isometry off a site tensor,
  leaving the bond tensor that the zero-site step evolves.
- `absorb_forward` / `absorb_backward`: multiply an evolved bond tensor into
  the neighboring site.

The zero-site step has no DMRG analogue and is the piece that makes TDVP a
projector-splitting integrator rather than a sequence of local evolutions. The
tangent-space projector double counts each bond once it is reached from both
sides, and evolving the bond tensor *backwards* in time cancels that excess
exactly. Dropping it does not merely degrade the accuracy order — it destroys
even first-order consistency, so `evolve_bond` is not optional.

The 2-site update scheme lives in `scheme_2s.py`; it reuses `evolve_1site` for
its own back-evolution and only needs its own forward step and splits.

Bond dimensions do not grow under the 1-site scheme. The splits here are
exact (no truncation): the 1-site tangent space cannot grow the bond, so
discarding singular values would lose norm without buying any variational
freedom. Use the 2-site scheme when the bond dimension has to adapt.

Being exact, a split reports the attainable rank, which near a chain edge can
fall below the nominal bond dimension — `dim_left · d` caps it there. The
first sweep therefore trims any such over-sized bond, losing nothing because
those states carried no weight, after which the profile is a fixed point.

That restriction bites hardest on a bond of dimension 1, across which the
scheme cannot transport any correlation at all. A product state has every
bond at dimension 1, so starting a `1s` quench from one freezes the dynamics
entirely, and a state only partly grown by a short `2s` warm-up stays frozen
across whichever bonds have not yet opened — silently, since norm and energy
remain conserved throughout. Quench from a product state with `2s`, or warm up
until every bond has opened before switching.

Index convention (shared with `environ.py`)
--------------------------------------------
- `a, b, c, d` — virtual bond indices of the MPS (bra or ket side)
- `o, p`       — virtual bond indices of the MPO
- `r, s`       — physical indices

1-site effective Hamiltonian layout:

    E_left [a, o, b]    — (bra_left,  mpo_left,  ket_left)
    W      [o, p, r, s] — (mpo_left,  mpo_right, phys_bra, phys_ket)
    E_right[c, p, d]    — (bra_right, mpo_right, ket_right)
    M      [b, d, s]    — (ket_left,  ket_right, phys_ket)
    output [a, c, r]    — (bra_left,  bra_right, phys_bra)  ← same shape as M

Zero-site effective Hamiltonian layout:

    E_left [a, o, b]    — (bra_left,  mpo_bond, ket_left)
    C      [b, d]       — (ket_left,  ket_right)
    E_right[c, o, d]    — (bra_right, mpo_bond, ket_right)
    output [a, c]       — (bra_left,  bra_right)  ← same shape as C

Both environment blocks sit on the same MPS bond and share one MPO bond, so
the zero-site Hamiltonian needs no extra storage: it is built from the very
blocks the 1-site step already uses.

Bond tensor orientation
-----------------------
`C` always has axis 0 facing `E_left` and axis 1 facing `E_right`, regardless
of sweep direction. The split helpers normalize to this orientation so that
`matvec_0s` is direction-agnostic, and the absorb helpers consume it again.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

from nicole import Tensor, contract, decomp, einsum

from .krylov import lanczos_expv


def matvec_1s(M: Tensor, W: Tensor, E_left: Tensor, E_right: Tensor) -> Tensor:
    """Apply the 1-site effective Hamiltonian H_eff to the site tensor M.

    Computes `H_eff|M⟩` as a single four-tensor einsum contraction. Identical
    to the DMRG 1-site `matvec`; it is duplicated here so the TDVP scheme has
    no cross-algorithm dependency.

    Parameters
    ----------
    M:
        Center site tensor with axes `(ket_left, ket_right, phys_ket)`.
    W:
        MPO site tensor with axes `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    E_left:
        Left environment with axes `(bra_left, mpo_left, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_right, ket_right)`.

    Returns
    -------
    Tensor
        `H_eff|M⟩` with axes `(bra_left, bra_right, phys_bra)`, identical
        in shape to `M`.
    """
    # 'aob,bds,oprs,cpd->acr'
    # contracted: b (ket_left), d (ket_right), s (phys_ket), o (mpo_left), p (mpo_right)
    # free:       a (bra_left), c (bra_right), r (phys_bra)
    # Starting with E_left × M eliminates the large MPS virtual bond first,
    # keeping intermediates small.
    return einsum('aob,bds,oprs,cpd->acr', E_left, M, W, E_right)


def matvec_0s(C: Tensor, E_left: Tensor, E_right: Tensor) -> Tensor:
    """Apply the zero-site effective Hamiltonian H_0 to the bond tensor C.

    The two environment blocks are closed against each other over their common
    MPO bond. No MPO site tensor appears: the bond carries no physical index.

    Parameters
    ----------
    C:
        Bond tensor with axes `(ket_left, ket_right)`, axis 0 facing `E_left`.
    E_left:
        Left environment with axes `(bra_left, mpo_bond, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_bond, ket_right)`.

    Returns
    -------
    Tensor
        `H_0|C⟩` with axes `(bra_left, bra_right)`, identical in shape to `C`.
    """
    # 'aob,bd,cod->ac'
    # contracted: b (ket_left), d (ket_right), o (shared MPO bond)
    # free:       a (bra_left), c (bra_right)
    # Contracting C first keeps the intermediate at rank 3 rather than closing
    # the two environments into a full bond-by-bond matrix.
    return einsum('aob,bd,cod->ac', E_left, C, E_right)


def evolve_1site(
    M: Tensor,
    W: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    dt: float,
    krylov_opts: dict,
) -> Tuple[Tensor, float]:
    """Evolve a site tensor under the 1-site effective Hamiltonian.

    Applies `exp(-i·dt·H_eff)` to `M` by Lanczos. This is a pure function: it
    takes tensors and returns tensors without mutating any MPS state. Being
    pure, it is the natural point for future parallel or distributed dispatch.

    Parameters
    ----------
    M:
        Center site tensor with axes `(ket_left, ket_right, phys_ket)`.
    W:
        MPO site tensor with axes `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    E_left:
        Left environment with axes `(bra_left, mpo_left, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_right, ket_right)`.
    dt:
        Time increment. Pass a negative value to evolve backwards.
    krylov_opts:
        Keyword arguments forwarded to `lanczos_expv`: `max_krylov`, `tol`.

    Returns
    -------
    Tensor
        Evolved site tensor with the same axis layout as `M`.
    float
        Krylov truncation error estimate for this step.
    """
    mv = partial(matvec_1s, W=W, E_left=E_left, E_right=E_right)

    return lanczos_expv(mv, M, dt, **krylov_opts)


def evolve_bond(
    C: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    dt: float,
    krylov_opts: dict,
) -> Tuple[Tensor, float]:
    """Evolve a bond tensor under the zero-site effective Hamiltonian.

    Applies `exp(-i·dt·H_0)` to `C` by Lanczos, using the same sign convention
    as `evolve_1site`. The sweep is responsible for passing the negated time
    increment here: the projector splitting requires the bond to run backwards
    relative to the sites it separates.

    Parameters
    ----------
    C:
        Bond tensor with axes `(ket_left, ket_right)`, axis 0 facing `E_left`.
    E_left:
        Left environment with axes `(bra_left, mpo_bond, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_bond, ket_right)`.
    dt:
        Time increment, already signed by the caller.
    krylov_opts:
        Keyword arguments forwarded to `lanczos_expv`: `max_krylov`, `tol`.

    Returns
    -------
    Tensor
        Evolved bond tensor with the same axis layout as `C`.
    float
        Krylov truncation error estimate for this step.
    """
    mv = partial(matvec_0s, E_left=E_left, E_right=E_right)

    return lanczos_expv(mv, C, dt, **krylov_opts)


def split_forward(M: Tensor, itag: str) -> Tuple[Tensor, Tensor]:
    """Peel a left isometry off a site tensor for a forward sweep step.

    Decomposes `M` as `A · C` with `A` left-isometric. The evolved site tensor
    keeps its physical content in `A`, which is frozen for the rest of the
    step, while `C` is handed to `evolve_bond` and then absorbed into the next
    site.

    Parameters
    ----------
    M:
        Evolved site tensor with axes `(ket_left, ket_right, phys)`.
    itag:
        itag for the new bond shared by `A` and `C`.

    Returns
    -------
    Tensor
        Left-isometric `A` with axes `(ket_left, new_bond, phys)`.
    Tensor
        Bond tensor `C` with axes `(new_bond, ket_right)` — axis 0 faces the
        left environment that `A` updates, axis 1 faces the right environment.
    """
    # cols = right bond only → LV gives isometry with axes (new_bond, left, phys).
    # trunc=None: the 1-site scheme never grows a bond, so there is nothing to
    # truncate; any rank below the nominal bond dimension is exact.
    C, A = decomp(M, axes=1, flow='<<', mode='LV', trunc=None)

    A.permute([1, 0, 2], in_place=True)
    A.retag(1, itag)

    # C comes out as (old_right, new_bond); flip to the canonical orientation.
    C.retag(1, itag)
    C.permute([1, 0], in_place=True)

    return A, C


def split_backward(M: Tensor, itag: str) -> Tuple[Tensor, Tensor]:
    """Peel a right isometry off a site tensor for a backward sweep step.

    Mirror image of `split_forward`: decomposes `M` as `C · B` with `B`
    right-isometric.

    Parameters
    ----------
    M:
        Evolved site tensor with axes `(ket_left, ket_right, phys)`.
    itag:
        itag for the new bond shared by `C` and `B`.

    Returns
    -------
    Tensor
        Bond tensor `C` with axes `(ket_left, new_bond)` — axis 0 faces the
        left environment, axis 1 faces the right environment that `B` updates.
    Tensor
        Right-isometric `B` with axes `(new_bond, ket_right, phys)`.
    """
    # rows = left bond only → LV gives isometry with axes (new_bond, right, phys),
    # already in MPS axis order.
    C, B = decomp(M, axes=0, flow='<<', mode='LV', trunc=None)

    B.retag(0, itag)

    # C comes out as (old_left, new_bond), already the canonical orientation.
    C.retag(1, itag)

    return C, B


def absorb_forward(C: Tensor, M_next: Tensor) -> Tensor:
    """Multiply an evolved bond tensor into the site to its right.

    Parameters
    ----------
    C:
        Bond tensor with axes `(new_bond, ket_right)`, as returned by
        `split_forward` and evolved by `evolve_bond`.
    M_next:
        Site tensor to the right with axes `(ket_left, ket_right, phys)`.

    Returns
    -------
    Tensor
        Updated site tensor with axes `(new_bond, ket_right, phys)`.
    """
    # Axis 1 of C is the old bond that M_next still carries.
    return contract(C, M_next, axes=(1, 0))


def absorb_backward(M_prev: Tensor, C: Tensor) -> Tensor:
    """Multiply an evolved bond tensor into the site to its left.

    Parameters
    ----------
    M_prev:
        Site tensor to the left with axes `(ket_left, ket_right, phys)`.
    C:
        Bond tensor with axes `(ket_left, new_bond)`, as returned by
        `split_backward` and evolved by `evolve_bond`.

    Returns
    -------
    Tensor
        Updated site tensor with axes `(ket_left, new_bond, phys)`.
    """
    # Axis 0 of C is the old bond that M_prev still carries. The contraction
    # leaves the new bond last, so restore the MPS axis order.
    M = contract(M_prev, C, axes=(1, 0))
    M.permute([0, 2, 1], in_place=True)

    return M
