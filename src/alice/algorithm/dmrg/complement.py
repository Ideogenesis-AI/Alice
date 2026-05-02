# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Alice library.
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


"""CBE complement isometry for 1-site-plus DMRG.

Implements the controlled bond expansion (CBE) step used by the 1-site-plus
scheme. Before each 1-site Davidson update, a complement isometry is computed
from a cheap, factored approximation of the 2-site gradient: H is applied
independently from the left and right to a truncated 2-site tensor; the current
(kept) subspace is projected out from each half; the resulting "doubly-discarded"
outer product is decomposed by SVD to yield new basis vectors that expand the MPS
bond.

The truncation in step 1 (`_cheap_factors`) is the key cost-saving measure:
instead of applying H to the full 2-site tensor M[i] ⊗ M[i+1] (bond b), the SVD
is used to reduce the internal bond to at most α ≤ b, giving compact factors
M̃[i] and M̃[i+1]. The subsequent half-applications and projections then scale
with α instead of b.

Index convention (shared with `environ.py`, `scheme_1s.py`, `scheme_2s.py`):

- `a, b, c, d` — virtual bond indices of the MPS (bra or ket side).
- `α`          — internal reduced bond of the truncated 2-site tensor.
- `o, p, q`    — virtual bond indices of the MPO.
- `r, s`       — physical indices of site i (bra r, ket s).
- `u, v`       — physical indices of site i+1 (bra u, ket v).

Tensor shapes used in this module:

- `left_half  [a, α, p, r]` — (bra_left, internal, mpo_mid, phys_bra_i)
- `right_half [α, p, c, u]` — (internal, mpo_mid, bra_right, phys_bra_{i+1})
- `N_L        [a, k, r]`    — complement vectors for M[i] right bond.
- `N_R        [k, c, u]`    — complement vectors for M[i+1] left bond.
"""

from __future__ import annotations

from typing import Optional, Tuple

from nicole import Tensor, einsum, oplus
from nicole import decomp

from .environ import step_left_env, step_right_env
from .scheme_2s import build_bulk, split_forward


# Temporary itag assigned to the truncated internal bond in _cheap_factors.
_TRUNC_ITAG: str = '_cbe_trunc_'

# Itag assigned to the complement bond produced by the SVD of Θ'_disc.
_COMPLEMENT_ITAG: str = '_cbe_comp_'


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cheap_factors(
    M_i: Tensor,
    M_i1: Tensor,
    alpha: Optional[int],
) -> Tuple[Tensor, Tensor]:
    """Form cheap truncated factors of Θ = M_i ⊗ M_i1 via SVD.

    Contracts M_i and M_i1 into the full 2-site bond tensor Θ, then performs a
    truncated SVD keeping at most α singular values. This yields compact factors
    M̃_i (left-isometric) and M̃_i1 (= S·Vh) with internal bond dimension α ≤ b.
    The truncation reduces the cost of the subsequent half-applications of H.

    Parameters
    ----------
    M_i:
        MPS tensor at site i with axes `(ket_left, ket_right, phys)`.
    M_i1:
        MPS tensor at site i+1 with axes `(ket_left, ket_right, phys)`.
    alpha:
        Max bond dimension for the truncated internal bond. `None` keeps all
        singular values (equivalent to using the full 2-site tensor).

    Returns
    -------
    Tensor
        M̃_i with axes `(ket_left, internal_α, phys_i)`, left-isometric.
    Tensor
        M̃_i1 with axes `(internal_α, ket_right_{i+1}, phys_{i+1})`.
    """
    theta = build_bulk(M_i, M_i1)
    trunc = {'nkeep': alpha} if alpha is not None else None
    M_tilde_i, M_tilde_i1 = split_forward(theta, itag=_TRUNC_ITAG, trunc=trunc)
    return M_tilde_i, M_tilde_i1


def _left_half(E_left: Tensor, M_trunc: Tensor, W: Tensor) -> Tensor:
    """Compute the left half of H acting on the truncated ket M̃_i.

    Contracts E_left, M̃_i (truncated ket), and W (MPO at site i), leaving the
    internal reduced bond α and MPO bond p free. Together with `_right_half`,
    this allows H·Θ̃ to be reconstructed as
    `einsum('aαpr,αpcu→acru', left_half, right_half)`.

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


def _project_complement_left(left_half: Tensor, M_i: Tensor) -> Tensor:
    """Project left_half onto the complement of the current M_i subspace.

    M_i is left-isometric (M̂† M̂ = I_b), so the projector onto its column space
    is P_L = M_i M_i†, acting on the joint (bra_left, phys_bra) indices of
    left_half. The discarded part is (I − P_L) left_half.

    Parameters
    ----------
    left_half:
        Tensor with axes `(bra_left=a, internal=α, mpo_mid=p, phys_bra=r)`.
    M_i:
        Original left-isometric site tensor at i with axes `(ket_left=a, ket_right=b, phys=r)`.

    Returns
    -------
    Tensor
        Discarded component of left_half, same axes as left_half.
    """
    # proj[a,α,p,r] = Σ_{b,x,z} left_half[x,α,p,z] · M_i*[x,b,z] · M_i[a,b,r]
    # left_half and M_i* share x and z → contract them first (intermediate b·α·p).
    proj = einsum('xαpz,xbz,abr->aαpr', left_half, M_i.conj(), M_i)
    return left_half - proj


def _project_complement_right(right_half: Tensor, M_i1: Tensor) -> Tensor:
    """Project right_half onto the complement of the current M_i1 subspace.

    M_i1 is right-isometric (M̂ M̂† = I_b), so the projector onto its row space
    is P_R = M_i1† M_i1, acting on the joint (bra_right, phys_bra) indices of
    right_half. The discarded part is (I − P_R) right_half.

    Parameters
    ----------
    right_half:
        Tensor with axes `(internal=α, mpo_mid=p, bra_right=c, phys_bra=u)`.
    M_i1:
        Original right-isometric site tensor at i+1 with axes `(ket_left=b, ket_right=c, phys=u)`.

    Returns
    -------
    Tensor
        Discarded component of right_half, same axes as right_half.
    """
    # proj[α,p,c,u] = Σ_{b,x,z} right_half[α,p,x,z] · M_i1*[b,x,z] · M_i1[b,c,u]
    # right_half and M_i1* share x and z → contract them first (intermediate b·α·p).
    proj = einsum('αpxz,bxz,bcu->αpcu', right_half, M_i1.conj(), M_i1)
    return right_half - proj


def _complement_vectors(
    left_half_disc: Tensor,
    right_half_disc: Tensor,
    k_expand: int,
    flow: str,
) -> Tuple[Tensor, Tensor]:
    """Extract complement vectors from the discarded 2-site outer product.

    Contracts the two discarded halves into the "doubly-discarded" 2-site tensor
    Θ'_disc and decomposes it by truncated SVD. The leading left and right singular
    vectors give the new bond directions to add to M[i] and M[i+1] respectively.

    The required `flow` depends on the sweep direction because `theta_disc` carries
    different index directions in each case:

    - Forward sweep: use `flow='<<'`. The index directions of `theta_disc` produce
      N_L with a new-bond direction matching the center tensor's right bond, and N_R
      with a new-bond direction matching the right-isometric neighbor's left bond.
    - Backward sweep: use `flow='>>'`. Directions of `theta_disc` are reversed
      relative to the forward case, so the opposite flow is needed.

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
        N_L with axes `(bra_left=a, k=new_bond, phys=r)` — new right-bond complement
        vectors for M[i].
    Tensor
        N_R with axes `(k=new_bond, bra_right=c, phys=u)` — new left-bond complement
        vectors for M[i+1].
    """
    # Contract the two discarded halves into Θ'_disc.
    # Axes (a, c, r, u) match the standard 2-site bond tensor convention.
    theta_disc = einsum('aαpr,αpcu->acru', left_half_disc, right_half_disc)

    # SVD with axes=[0, 2] fuses (a, r) as left and (c, u) as right.
    U, _S, Vh = decomp(
        theta_disc, axes=[0, 2], mode='SVD',
        trunc={'nkeep': k_expand}, flow=flow,
        itag=_COMPLEMENT_ITAG,
    )
    # After decomp + unmerge: U has axes (a, r, k); permute to (a, k, r).
    U.permute([0, 2, 1], in_place=True)   # N_L: (a, k, r)
    # Vh has axes (k, c, u). N_R = Vh.
    return U, Vh


def _expand_env_right(
    E_right_i1: Tensor,
    M_i1: Tensor,
    N_R: Tensor,
    W_i1: Tensor,
) -> Tensor:
    """Build the expanded right environment at site i.

    Computes two right-environment blocks via `step_right_env` — one from the
    original M_i1 and one from the complement N_R — and assembles them into a
    block-diagonal expanded environment via `oplus`. Cross terms between M_i1 and
    N_R are omitted; they are small because N_R lies approximately in the
    complement of M_i1's row space.

    Parameters
    ----------
    E_right_i1:
        Right environment at site i+1, axes `(bra_right, mpo_right, ket_right)`.
    M_i1:
        Original right-isometric MPS tensor at i+1, axes `(ket_left, ket_right, phys)`.
    N_R:
        Complement tensor, axes `(new_bond, ket_right, phys)`, acting as the new
        left-bond rows for the expanded M[i+1].
    W_i1:
        MPO tensor at site i+1, axes `(mpo_left, mpo_right, phys_bra, phys_ket)`.

    Returns
    -------
    Tensor
        Expanded right environment at site i with axes
        `(bra_right_exp, mpo_right, ket_right_exp)`, bond dimension b+k.
    """
    # (b, p, b) diagonal block: environment seen by the original M_i1.
    E_orig = step_right_env(E_right_i1, M_i1, W_i1)
    # (k, p, k) diagonal block: environment seen by the complement N_R.
    E_new  = step_right_env(E_right_i1, N_R,  W_i1)
    # Block-diagonal assembly via direct sum on the bra/ket bond axes (0 and 2).
    return oplus(E_orig, E_new, axes=[0, 2])


def _expand_env_left(
    E_left_im1: Tensor,
    M_im1: Tensor,
    N_L: Tensor,
    W_im1: Tensor,
) -> Tensor:
    """Build the expanded left environment at site i.

    Mirror of `_expand_env_right` for the backward sweep direction. Computes
    two left-environment blocks from the original M_im1 and the complement N_L,
    then assembles them block-diagonally via `oplus`.

    Parameters
    ----------
    E_left_im1:
        Left environment at site i-1, axes `(bra_left, mpo_left, ket_left)`.
    M_im1:
        Original left-isometric MPS tensor at site i-1, axes `(ket_left, ket_right, phys)`.
    N_L:
        Complement tensor, axes `(ket_left, new_bond, phys)`, acting as the new
        right-bond columns for the expanded M[i-1].
    W_im1:
        MPO tensor at site i-1, axes `(mpo_left, mpo_right, phys_bra, phys_ket)`.

    Returns
    -------
    Tensor
        Expanded left environment at site i with axes
        `(bra_left_exp, mpo_left, ket_left_exp)`, bond dimension b+k.
    """
    # (b, p, b) diagonal block: environment seen by the original M_im1.
    E_orig = step_left_env(E_left_im1, M_im1, W_im1)
    # (k, p, k) diagonal block: environment seen by the complement N_L.
    E_new  = step_left_env(E_left_im1, N_L,  W_im1)
    # Block-diagonal assembly via direct sum on the bra/ket bond axes (0 and 2).
    return oplus(E_orig, E_new, axes=[0, 2])


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

    The orthogonality center sits at site i (M_i left-isometric, M_i1
    right-isometric). The procedure is:

    1. Truncated SVD of Θ = M_i ⊗ M_i1 → cheap factors M̃_i, M̃_i1 (bond α).
    2. Factored half-application of H: `left_half` from E_left and M̃_i;
       `right_half` from E_right and M̃_i1.
    3. Discarded-space projection of each half using original M_i and M_i1.
    4. SVD of Θ'_disc → N_L (new right-bond vectors for M_i) and
       N_R (new left-bond vectors for M_i1).
    5. Expansion of M_i and M_i1 via `oplus`.
    6. Expansion of the right environment at site i from E_right[i+1].

    Parameters
    ----------
    M_i:
        Left-isometric MPS tensor at site i, axes `(ket_left, ket_right, phys)`.
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
        Maximum number of complement vectors to add per bond end.
    alpha:
        Truncated internal bond dimension for Θ̃. `None` keeps the full bond.

    Returns
    -------
    Tensor
        M_i_exp — expanded M_i with axes `(ket_left, ket_right_exp, phys)`.
    Tensor
        M_i1_exp — expanded M_i1 with axes `(ket_left_exp, ket_right, phys)`.
    Tensor
        E_right_i_exp — expanded right environment at site i, axes
        `(bra_right_exp, mpo_right, ket_right_exp)`.
    """
    # Step 1: cheap truncated factors of Θ.
    M_tilde_i, M_tilde_i1 = _cheap_factors(M_i, M_i1, alpha)

    # Step 2: factored half-application of H.
    lh = _left_half(E_left_i,   M_tilde_i,  W_i)
    rh = _right_half(E_right_i1, M_tilde_i1, W_i1)

    # Step 3: discarded-space projection using the original (non-truncated) tensors.
    lh_disc = _project_complement_left(lh,  M_i)
    rh_disc = _project_complement_right(rh, M_i1)

    # Step 4: complement vectors from SVD of Θ'_disc.
    # flow='<<' produces N_L/N_R with index directions compatible with oplus on
    # the center tensor (right bond = OUT) and right-isometric neighbor (left bond = IN).
    N_L, N_R = _complement_vectors(lh_disc, rh_disc, k_expand, flow='<<')

    # Step 5: expand M_i right bond (axis 1) and M_i1 left bond (axis 0).
    M_i_exp  = oplus(M_i,  N_L, axes=[1])
    M_i1_exp = oplus(M_i1, N_R, axes=[0])

    # Step 6: expanded right environment at site i.
    E_right_i_exp = _expand_env_right(E_right_i1, M_i1, N_R, W_i1)

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

    Mirror of `expand_forward` for backward sweeps. The orthogonality center
    is at site i; M_im1 is left-isometric (from the previous forward sweep) and
    M_i is the active site being updated. The complement expansion targets the
    bond (i-1, i):

    1. Truncated SVD of Θ = M_im1 ⊗ M_i → cheap M̃_im1, M̃_i (bond α).
    2. Factored half-application of H using E_left[i-1] and E_right[i].
    3. Discarded-space projection using original M_im1 and M_i.
    4. SVD of Θ'_disc → N_L (new right-bond vectors for M_im1), N_R (new left-bond for M_i).
    5. Expansion of M_i left bond and M_im1 right bond via `oplus`.
    6. Expansion of the left environment at site i from E_left[i-1].

    Parameters
    ----------
    M_i:
        MPS tensor at site i (right-isometric or center), axes `(ket_left, ket_right, phys)`.
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
        Maximum number of complement vectors to add per bond end.
    alpha:
        Truncated internal bond dimension for Θ̃. `None` keeps the full bond.

    Returns
    -------
    Tensor
        M_i_exp — expanded M_i with axes `(ket_left_exp, ket_right, phys)`.
    Tensor
        M_im1_exp — expanded M_im1 with axes `(ket_left, ket_right_exp, phys)`.
    Tensor
        E_left_i_exp — expanded left environment at site i, axes
        `(bra_left_exp, mpo_left, ket_left_exp)`.
    """
    # Step 1: cheap truncated factors of Θ = M_im1 ⊗ M_i.
    M_tilde_im1, M_tilde_i = _cheap_factors(M_im1, M_i, alpha)

    # Step 2: factored half-application of H.
    lh = _left_half(E_left_im1, M_tilde_im1, W_im1)
    rh = _right_half(E_right_i,  M_tilde_i,  W_i)

    # Step 3: discarded-space projection using the original tensors.
    lh_disc = _project_complement_left(lh,   M_im1)
    rh_disc = _project_complement_right(rh,  M_i)

    # Step 4: complement vectors.
    # N_L: new right-bond vectors for M_im1.
    # N_R: new left-bond vectors for M_i.
    # flow='>>' produces N_L/N_R compatible with oplus on the left-isometric M_im1
    # (right bond = IN) and the center tensor M_i (left bond = OUT).
    N_L, N_R = _complement_vectors(lh_disc, rh_disc, k_expand, flow='>>')

    # Step 5: expand M_i left bond (axis 0) and M_im1 right bond (axis 1).
    M_i_exp   = oplus(M_i,   N_R, axes=[0])
    M_im1_exp = oplus(M_im1, N_L, axes=[1])

    # Step 6: expanded left environment at site i.
    E_left_i_exp = _expand_env_left(E_left_im1, M_im1, N_L, W_im1)

    return M_i_exp, M_im1_exp, E_left_i_exp
