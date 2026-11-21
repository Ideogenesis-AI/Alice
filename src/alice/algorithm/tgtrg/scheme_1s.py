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


"""1-site TgTRG update scheme, including the zero-site back-evolution.

This module holds the operations a 1-site-sized cooling update is assembled
from:

- `matvec_1s` / `evolve_1site`: apply `exp(-tau·H_eff)` to a single site
  tensor of the density matrix, where `H_eff = E_left ⊗ W ⊗ E_right`.
- `matvec_0s` / `evolve_bond`: apply `exp(-tau·H_0)` to a bond tensor, where
  `H_0 = E_left ⊗ E_right` closes directly over the shared MPO bond.
- `split_forward` / `split_backward`: peel an isometry off a site tensor,
  leaving the bond tensor that the zero-site step evolves.
- `absorb_forward` / `absorb_backward`: multiply an evolved bond tensor into
  the neighboring site.

The zero-site step is the piece that makes this a projector-splitting
integrator instead of a sequence of local evolutions. The tangent-space
projector double counts each bond once it is reached from both sides, and the
bond tensor is therefore evolved with the *opposite* sign — Eq. (2) of
arXiv:2212.11973 splits into `dA_i/dβ = -H⁽¹⁾_i A_i` and
`dS_i/dβ = +H⁽⁰⁾_i S_i`, and the sweep calls `evolve_bond` with `-tau` to
realize the second. Dropping it does not merely degrade the accuracy order —
it destroys even first-order consistency, so `evolve_bond` is not optional.

The 2-site update scheme lives in `scheme_2s.py`; it reuses `evolve_1site`
for its own back-evolution and only needs its own forward step and splits.

Bond dimensions do not grow under the 1-site scheme, and its splits are
therefore called without truncation: the 1-site tangent space cannot grow a
bond, so discarding singular values would lose weight without buying any
variational freedom. Use the 2-site scheme when the bond dimension has to
adapt, as a cooling run almost always needs it to: `ρ(τ₀)` starts at the
Hamiltonian's own bond dimension and has to grow from there.

Index convention (shared with `environ.py`)
--------------------------------------------
- `a, b, c, d` — virtual bond indices of ρ (bra or ket side)
- `o, p`       — virtual bond indices of the Hamiltonian MPO
- `r, x`       — `phys_in` of the bra ρ and of the ket ρ, the axes `H` joins
- `s`          — `phys_out`, shared by bra and ket ρ and absent from `H`

1-site effective Hamiltonian layout:

    E_left [a, o, b]       — (bra_left,  mpo_left,  ket_left)
    W      [o, p, r, x]    — (mpo_left,  mpo_right, phys_bra, phys_ket)
    E_right[c, p, d]       — (bra_right, mpo_right, ket_right)
    M      [b, d, x, s]    — (ket_left,  ket_right, phys_in, phys_out)
    output [a, c, r, s]    — (bra_left,  bra_right, phys_in, phys_out)
                             ← same shape as M

Zero-site effective Hamiltonian layout:

    E_left [a, o, b]    — (bra_left,  mpo_bond, ket_left)
    C      [b, d]       — (ket_left,  ket_right)
    E_right[c, o, d]    — (bra_right, mpo_bond, ket_right)
    output [a, c]       — (bra_left,  bra_right)  ← same shape as C

Both environment blocks sit on the same ρ bond and share one MPO bond, so the
zero-site Hamiltonian needs no extra storage: it is built from the very
blocks the 1-site step already uses.

Bond tensor orientation
-----------------------
`C` always has axis 0 facing `E_left` and axis 1 facing `E_right`, regardless
of sweep direction. The split helpers normalize to this orientation so that
`matvec_0s` is direction-agnostic, and the absorb helpers consume it again.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

from nicole import Tensor, contract, decomp, einsum

from .krylov import lanczos_expv


def matvec_1s(M: Tensor, W: Tensor, E_left: Tensor, E_right: Tensor) -> Tensor:
    """Apply the 1-site effective Hamiltonian H_eff to the site tensor M.

    `phys_out` never meets the Hamiltonian: it is carried through as a
    spectator, making this the Choi-space effective Hamiltonian of `H ⊗ I`.

    Parameters
    ----------
    M:
        Center site tensor with axes
        `(ket_left, ket_right, phys_in, phys_out)`.
    W:
        Hamiltonian site tensor with axes
        `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    E_left:
        Left environment with axes `(bra_left, mpo_left, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_right, ket_right)`.

    Returns
    -------
    Tensor
        `H_eff|M⟩` with axes `(bra_left, bra_right, phys_in, phys_out)`,
        identical in shape to `M`.
    """
    # Sequential two-tensor contractions, as everywhere in this algorithm: rho
    # and the Hamiltonian share the W{i} bond itag family, so every contracted
    # pair is named explicitly. Starting with E_left × M eliminates the large
    # virtual bond first, keeping intermediates small.
    # Step 1: E_left(a,o,b) × M(b,d,x,s) → (a,o,d,x,s), contracting 'b'.
    t1 = einsum('aob,bdxs->aodxs', E_left, M)
    # Step 2: t1(a,o,d,x,s) × W(o,p,r,x) → (a,d,s,p,r), contracting 'o','x'.
    t2 = einsum('aodxs,oprx->adspr', t1, W)
    # Step 3: t2(a,d,s,p,r) × E_right(c,p,d) → (a,c,r,s), contracting 'd','p'.
    return einsum('adspr,cpd->acrs', t2, E_right)


def matvec_0s(C: Tensor, E_left: Tensor, E_right: Tensor) -> Tensor:
    """Apply the zero-site effective Hamiltonian H_0 to the bond tensor C.

    The two environment blocks are closed against each other over their
    common MPO bond. No Hamiltonian site tensor appears: the bond carries no
    physical index.

    Parameters
    ----------
    C:
        Bond tensor with axes `(ket_left, ket_right)`, axis 0 facing
        `E_left`.
    E_left:
        Left environment with axes `(bra_left, mpo_bond, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_bond, ket_right)`.

    Returns
    -------
    Tensor
        `H_0|C⟩` with axes `(bra_left, bra_right)`, identical in shape to `C`.
    """
    # Contracting C first keeps the intermediate at 3rd order instead of
    # closing the two environments into a full bond-by-bond matrix.
    # Step 1: E_left(a,o,b) × C(b,d) → (a,o,d), contracting 'b'.
    t1 = einsum('aob,bd->aod', E_left, C)
    # Step 2: t1(a,o,d) × E_right(c,o,d) → (a,c), contracting 'o','d'.
    return einsum('aod,cod->ac', t1, E_right)


def evolve_1site(
    M: Tensor,
    W: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    tau: float,
    krylov_opts: dict,
) -> Tuple[Tensor, float, float]:
    """Cool a site tensor under the 1-site effective Hamiltonian.

    Applies `exp(-tau·H_eff)` to `M` by Lanczos. This is a pure function: it
    takes tensors and returns tensors without mutating any density matrix.
    Being pure, it is the natural point for future parallel or distributed
    dispatch.

    Parameters
    ----------
    M:
        Center site tensor with axes
        `(ket_left, ket_right, phys_in, phys_out)`.
    W:
        Hamiltonian site tensor with axes
        `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    E_left:
        Left environment with axes `(bra_left, mpo_left, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_right, ket_right)`.
    tau:
        Imaginary-time increment. Pass a negative value to heat, as the
        back-evolution of the 2-site scheme does.
    krylov_opts:
        Keyword arguments forwarded to `lanczos_expv`: `max_krylov`, `tol`.

    Returns
    -------
    Tensor
        Cooled site tensor with the same axis layout as `M`, with the
        extremal-eigenvalue factor divided out.
    float
        `log` of the factor divided out, for the caller to accumulate.
    float
        Krylov truncation error estimate for this step.
    """
    mv = partial(matvec_1s, W=W, E_left=E_left, E_right=E_right)

    return lanczos_expv(mv, M, tau, **krylov_opts)


def evolve_bond(
    C: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    tau: float,
    krylov_opts: dict,
) -> Tuple[Tensor, float, float]:
    """Cool a bond tensor under the zero-site effective Hamiltonian.

    Applies `exp(-tau·H_0)` to `C` by Lanczos, using the same sign convention
    as `evolve_1site`. The sweep is responsible for passing the negated
    increment here: the projector splitting requires the bond to run
    backwards relative to the sites it separates.

    Parameters
    ----------
    C:
        Bond tensor with axes `(ket_left, ket_right)`, axis 0 facing
        `E_left`.
    E_left:
        Left environment with axes `(bra_left, mpo_bond, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_bond, ket_right)`.
    tau:
        Imaginary-time increment, already signed by the caller.
    krylov_opts:
        Keyword arguments forwarded to `lanczos_expv`: `max_krylov`, `tol`.

    Returns
    -------
    Tensor
        Evolved bond tensor with the same axis layout as `C`, with the
        extremal-eigenvalue factor divided out.
    float
        `log` of the factor divided out, for the caller to accumulate.
    float
        Krylov truncation error estimate for this step.
    """
    mv = partial(matvec_0s, E_left=E_left, E_right=E_right)

    return lanczos_expv(mv, C, tau, **krylov_opts)


def split_forward(
    M: Tensor,
    itag: str,
    trunc: Optional[dict] = None,
) -> Tuple[Tensor, Tensor]:
    """Peel a left isometry off a site tensor for a forward sweep step.

    Decomposes `M` as `A · C` with `A` left-isometric — the `A_i = A_i^L S_i`
    factorization of the paper. The cooled site tensor keeps its physical
    content in `A`, which is frozen for the rest of the step, while `C` is
    handed to `evolve_bond` and then absorbed into the next site.

    Parameters
    ----------
    M:
        Cooled site tensor with axes
        `(ket_left, ket_right, phys_in, phys_out)`.
    itag:
        itag for the new bond shared by `A` and `C`.
    trunc:
        Truncation options forwarded to `decomp` (`nkeep`, `thresh`). `None`
        (the default, and what `'1s'` always passes) splits exactly.

    Returns
    -------
    Tensor
        Left-isometric `A` with axes
        `(ket_left, new_bond, phys_in, phys_out)`.
    Tensor
        Bond tensor `C` with axes `(new_bond, ket_right)` — axis 0 faces the
        left environment that `A` updates, axis 1 faces the right
        environment.
    """
    # cols = right bond only → LV gives the isometry with axes
    # (new_bond, left, phys_in, phys_out).
    C, A = decomp(M, axes=1, flow='<<', mode='LV', trunc=trunc)

    A.permute([1, 0, 2, 3], in_place=True)
    A.retag(1, itag)

    # C comes out as (old_right, new_bond); flip to the canonical orientation.
    C.retag(1, itag)
    C.permute([1, 0], in_place=True)

    return A, C


def split_backward(
    M: Tensor,
    itag: str,
    trunc: Optional[dict] = None,
) -> Tuple[Tensor, Tensor]:
    """Peel a right isometry off a site tensor for a backward sweep step.

    Mirror image of `split_forward`: decomposes `M` as `C · B` with `B`
    right-isometric.

    Parameters
    ----------
    M:
        Cooled site tensor with axes
        `(ket_left, ket_right, phys_in, phys_out)`.
    itag:
        itag for the new bond shared by `C` and `B`.
    trunc:
        Truncation options forwarded to `decomp` (`nkeep`, `thresh`). `None`
        (the default) splits exactly.

    Returns
    -------
    Tensor
        Bond tensor `C` with axes `(ket_left, new_bond)` — axis 0 faces the
        left environment, axis 1 faces the right environment that `B`
        updates.
    Tensor
        Right-isometric `B` with axes
        `(new_bond, ket_right, phys_in, phys_out)`.
    """
    # rows = left bond only → LV gives the isometry with axes
    # (new_bond, right, phys_in, phys_out), already in MPO axis order.
    C, B = decomp(M, axes=0, flow='<<', mode='LV', trunc=trunc)

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
        Site tensor to the right with axes
        `(ket_left, ket_right, phys_in, phys_out)`.

    Returns
    -------
    Tensor
        Updated site tensor with axes
        `(new_bond, ket_right, phys_in, phys_out)`.
    """
    # Axis 1 of C is the old bond that M_next still carries.
    return contract(C, M_next, axes=(1, 0))


def absorb_backward(M_prev: Tensor, C: Tensor) -> Tensor:
    """Multiply an evolved bond tensor into the site to its left.

    Parameters
    ----------
    M_prev:
        Site tensor to the left with axes
        `(ket_left, ket_right, phys_in, phys_out)`.
    C:
        Bond tensor with axes `(ket_left, new_bond)`, as returned by
        `split_backward` and evolved by `evolve_bond`.

    Returns
    -------
    Tensor
        Updated site tensor with axes
        `(ket_left, new_bond, phys_in, phys_out)`.
    """
    # Axis 0 of C is the old bond that M_prev still carries. The contraction
    # leaves the new bond last, so restore the MPO axis order.
    M = contract(M_prev, C, axes=(1, 0))
    M.permute([0, 3, 1, 2], in_place=True)

    return M
