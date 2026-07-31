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


"""CBE complement isometry for 1-site-plus ('1sp') XTRG.

Implements the controlled bond expansion (CBE) step used by the 1s+ scheme
of Zhang & von Delft, arXiv:2510.25022 (Section III), adapted from Alice's
DMRG `'1sp'` scheme to XTRG's linear (non-eigenvalue) least-squares fitting
problem `C ≈ A · B`.

Unlike DMRG's `_cheap_factors` (a single joint SVD of the 2-site ket
`M_i ⊗ M_i1`), the cheap step here (Eq. 13 of the paper) is two *independent*
single-tensor SVDs: one factor-MPO tensor per connector bond (`A_j` for the
A-chain, `B_j` for the B-chain), each compressing only its own bond. This
never forms any object of size O(D²) — the key cost saving over a direct
2-site join.

Index convention (shared with `environ.py`, `scheme_1s.py`, `scheme_2s.py`):

- `a, b, c, d` — virtual bond indices of the compressed MPO C.
- `p, q`       — reduced (or exact) connector bonds of the A- and B-chains.
- `r, s`       — physical indices of site i (phys_in r, phys_out s).
- `u, v`       — physical indices of site j (phys_in u, phys_out v).

Tensor shapes used in this module:

- `left_half  [c, p, q, r, s]` — (C_left, A_connector, B_connector, phys_in, phys_out)
- `right_half [d, p, q, u, v]` — (C_right, A_connector, B_connector, phys_in, phys_out)
- `N_L        [c, k, r, s]`    — complement vectors for C_i's right bond.
- `N_R        [k, d, u, v]`    — complement vectors for C_j's left bond.
"""

from __future__ import annotations

from typing import Optional, Tuple

from nicole import Tensor, contract, decomp, einsum, oplus

from .environ import step_left_env, step_right_env
from .scheme_1s import local_update_1s
from .scheme_2s import left_partial, right_partial


# Itags assigned to the cheaply reduced A- and B-chain connector bonds.
_BOND_ITAG_A: str = '_cbe_bond_a_'
_BOND_ITAG_B: str = '_cbe_bond_b_'

# Itag assigned to the complement bond produced by the SVD of Θ'_disc.
_COMPLEMENT_ITAG: str = '_cbe_comp_'


# ---------------------------------------------------------------------------
# Per-operand cheap bond compression (Eq. 13)
# ---------------------------------------------------------------------------

def _compress_bond(far_tensor: Tensor, axis: int, alpha: int, itag: str) -> Tensor:
    """Return the isometry compressing one connector bond of `far_tensor`.

    Performs a single-tensor truncated SVD of `far_tensor` (Eq. 13),
    treating `axis` (the connector bond shared with the near-side tensor) as
    the left partition and all other axes (own bond, physical legs) as the
    right partition. This SVD depends only on `far_tensor` itself — no
    object of size O(D²) is ever formed, which is what makes the
    compression cheap: cost `O(D³d²)` instead of the `O(D⁴d³)` of a direct
    2-site join.

    Parameters
    ----------
    far_tensor:
        The far-side factor-MPO tensor (`A_j` or `B_j`) whose connector bond
        (shared with the near-side tensor) is compressed.
    axis:
        Axis index of the connector bond within `far_tensor`.
    alpha:
        Maximum bond dimension to keep.
    itag:
        Itag assigned to the reduced bond.

    Returns
    -------
    Tensor
        Isometry with axes `(connector_bond, reduced_bond)`.
    """
    iso, _ = decomp(far_tensor, axes=axis, mode='UR', trunc={'nkeep': alpha}, itag=itag)
    return iso


def _absorb_right(tensor: Tensor, iso: Tensor, axis: int) -> Tensor:
    """Absorb `iso` onto `tensor`'s connector `axis`, replacing it with `iso`'s reduced bond.

    Used for the near-side operand of a connector bond: the resulting
    tensor is `tensor · iso`, i.e. one half of the projector `iso · iso†`
    inserted onto the bond (Eq. 13-14). The reduced axis is placed at the
    same position `axis` occupied originally, preserving `tensor`'s usual
    `(left, right, phys_in, phys_internal)`-style axis convention.

    Parameters
    ----------
    tensor:
        Near-side factor-MPO tensor (`A_i` or `B_i`).
    iso:
        Isometry from `_compress_bond`, axes `(connector_bond, reduced_bond)`.
    axis:
        Axis index of the connector bond within `tensor`.

    Returns
    -------
    Tensor
        `tensor` with axis `axis` replaced by the reduced bond.
    """
    rank = len(tensor.indices)
    perm = list(range(axis)) + [rank - 1] + list(range(axis, rank - 1))
    return contract(tensor, iso, axes=(axis, 0), perm=perm)


def _absorb_left(iso: Tensor, tensor: Tensor, axis: int) -> Tensor:
    """Absorb `iso.conj()` onto `tensor`'s connector `axis`.

    Companion to `_absorb_right`: contracts the conjugate isometry onto the
    far-side operand (the tensor `iso` was itself computed from), giving
    `iso† · tensor`, the other half of the projector `iso · iso†` on the
    connector bond. The reduced axis is placed at the same position `axis`
    occupied originally.

    Parameters
    ----------
    iso:
        Isometry from `_compress_bond`, axes `(connector_bond, reduced_bond)`.
    tensor:
        Far-side factor-MPO tensor (`A_j` or `B_j`) that `iso` was computed
        from.
    axis:
        Axis index of the connector bond within `tensor` (matches the
        `axis` originally passed to `_compress_bond`).

    Returns
    -------
    Tensor
        `tensor` with axis `axis` replaced by the reduced bond.
    """
    rank = len(tensor.indices)
    perm = list(range(axis)) + [rank - 1] + list(range(axis, rank - 1))
    return contract(tensor, iso.conj(), axes=(axis, 0), perm=perm)


def _reduce_operands(
    A_near: Tensor,
    A_far: Tensor,
    B_near: Tensor,
    B_far: Tensor,
    alpha: Optional[int],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cheaply compress the A- and B-chain connector bonds (Eq. 13-14).

    Independently SVDs the far-side tensor of each factor MPO (`A_far`'s own
    axis 0, `B_far`'s own axis 0) and absorbs the resulting isometry (and
    its conjugate) onto both sides of that bond via `_absorb_right` /
    `_absorb_left`. Each SVD is `O(D³d²)` and depends on only one operand;
    no `O(D²)`-scale object is ever formed. `alpha=None` skips compression
    entirely and returns the operands unchanged (falls back to the exact,
    full-bond-dimension halves).

    Parameters
    ----------
    A_near:
        Factor MPO A at the near site, own right bond (axis 1) is the
        connector.
    A_far:
        Factor MPO A at the far site, own left bond (axis 0) is the
        connector.
    B_near:
        Factor MPO B at the near site, own right bond (axis 1) is the
        connector.
    B_far:
        Factor MPO B at the far site, own left bond (axis 0) is the
        connector.
    alpha:
        Maximum reduced bond dimension. `None` returns the operands as-is.

    Returns
    -------
    Tensor
        `A_near_reduced`.
    Tensor
        `A_far_reduced`.
    Tensor
        `B_near_reduced`.
    Tensor
        `B_far_reduced`.
    """
    if alpha is None:
        return A_near, A_far, B_near, B_far

    iso_a = _compress_bond(A_far, axis=0, alpha=alpha, itag=_BOND_ITAG_A)
    iso_b = _compress_bond(B_far, axis=0, alpha=alpha, itag=_BOND_ITAG_B)

    A_near_reduced = _absorb_right(A_near, iso_a, axis=1)
    A_far_reduced = _absorb_left(iso_a, A_far, axis=0)
    B_near_reduced = _absorb_right(B_near, iso_b, axis=1)
    B_far_reduced = _absorb_left(iso_b, B_far, axis=0)

    return A_near_reduced, A_far_reduced, B_near_reduced, B_far_reduced


# ---------------------------------------------------------------------------
# Kept-subspace projection (Eq. 12)
# ---------------------------------------------------------------------------

def _project_complement_left(left_half: Tensor, C_i: Tensor) -> Tensor:
    """Project `left_half` onto the complement of the current `C_i` subspace.

    `C_i` is (approximately) left-isometric — `C_i† C_i = I` on its right
    bond — so the projector onto its `(left, phys_in, phys_out)` column
    space is `P_L = C_i C_i†`. The discarded part is `(I − P_L) left_half`.

    Parameters
    ----------
    left_half:
        Tensor with axes `(c, p, q, r, s)` = (C_left, A_connector,
        B_connector, phys_in, phys_out).
    C_i:
        Compressed-MPO tensor at site i, axes `(c, d, r, s)` = (left, right,
        phys_in, phys_out).

    Returns
    -------
    Tensor
        Discarded component of `left_half`, same axes as `left_half`.
    """
    # proj[c,p,q,r,s] = Σ_{x,z,w,d} left_half[x,p,q,z,w] · C_i*[x,d,z,w] · C_i[c,d,r,s]
    proj = einsum('xpqzw,xdzw,cdrs->cpqrs', left_half, C_i.conj(), C_i)
    return left_half - proj


def _project_complement_right(right_half: Tensor, C_j: Tensor) -> Tensor:
    """Project `right_half` onto the complement of the current `C_j` subspace.

    `C_j` is (approximately) right-isometric — `C_j† C_j = I` on its left
    bond — so the projector onto its `(right, phys_in, phys_out)` row space
    is `P_R = C_j† C_j`. The discarded part is `(I − P_R) right_half`.

    Parameters
    ----------
    right_half:
        Tensor with axes `(d, p, q, u, v)` = (C_right, A_connector,
        B_connector, phys_in, phys_out).
    C_j:
        Compressed-MPO tensor at site j, axes `(d, e, u, v)` = (left, right,
        phys_in, phys_out).

    Returns
    -------
    Tensor
        Discarded component of `right_half`, same axes as `right_half`.
    """
    # proj[d,p,q,u,v] = Σ_{x,y,z,w} right_half[x,p,q,y,z] · C_j*[w,x,y,z] · C_j[w,d,u,v]
    proj = einsum('xpqyz,wxyz,wduv->dpquv', right_half, C_j.conj(), C_j)
    return right_half - proj


# ---------------------------------------------------------------------------
# Complement direction extraction (Eq. 14-15)
# ---------------------------------------------------------------------------

def _join_disc(left_disc: Tensor, right_disc: Tensor) -> Tensor:
    """Join the discarded halves into the doubly-discarded bond tensor Θ'_disc.

    Contracts over the small `(p, q)` connector legs — cheap,
    `O(D²·alpha_a·alpha_b·d⁴)`, subdominant to the exact fill-in step.

    Parameters
    ----------
    left_disc:
        Discarded left half, axes `(c, p, q, r, s)`.
    right_disc:
        Discarded right half, axes `(d, p, q, u, v)`.

    Returns
    -------
    Tensor
        Θ'_disc with axes `(c, d, r, s, u, v)`.
    """
    return einsum('cpqrs,dpquv->cdrsuv', left_disc, right_disc)


def _new_direction_right(left_disc: Tensor, right_disc: Tensor, k_expand: int) -> Tensor:
    """Extract new left-bond complement vectors for the far-side tensor (forward sweep).

    Joins the discarded halves into Θ'_disc, then performs a truncated SVD
    keeping only the `R = S·Vh` factor `N_R` — the new left-bond directions
    for the far-side compressed-MPO tensor `C_j`. The other (left-isometric)
    factor is not needed here: `C_i`'s expansion is obtained exactly via the
    fill-in step instead (Eq. 15/16).

    Parameters
    ----------
    left_disc:
        Discarded left half, axes `(c, p, q, r, s)`.
    right_disc:
        Discarded right half, axes `(d, p, q, u, v)`.
    k_expand:
        Maximum number of complement directions to keep.

    Returns
    -------
    Tensor
        `N_R`, axes `(k, d, u, v)` — new left-bond directions for `C_j`.
    """
    theta_disc = _join_disc(left_disc, right_disc)
    _, N_R = decomp(
        theta_disc, axes=[0, 2, 3], mode='UR', trunc={'nkeep': k_expand},
        itag=_COMPLEMENT_ITAG, flow='<<',
    )
    return N_R


def _new_direction_left(left_disc: Tensor, right_disc: Tensor, k_expand: int) -> Tensor:
    """Extract new right-bond complement vectors for the far-side tensor (backward sweep).

    Mirror of `_new_direction_right`: joins the discarded halves into
    Θ'_disc, then performs a truncated SVD keeping only the `L = U·S`
    factor `N_L` — the new right-bond directions for the far-side
    compressed-MPO tensor `C_i` (site i-1 in a backward sweep).

    Parameters
    ----------
    left_disc:
        Discarded left half, axes `(c, p, q, r, s)`.
    right_disc:
        Discarded right half, axes `(d, p, q, u, v)`.
    k_expand:
        Maximum number of complement directions to keep.

    Returns
    -------
    Tensor
        `N_L`, axes `(c, k, r, s)` — new right-bond directions for `C_i`.
    """
    theta_disc = _join_disc(left_disc, right_disc)
    N_L, _ = decomp(
        theta_disc, axes=[0, 2, 3], mode='LV', trunc={'nkeep': k_expand},
        itag=_COMPLEMENT_ITAG, flow='>>',
    )
    # N_L unmerged axes: (c, r, s, k); permute to (c, k, r, s).
    N_L.permute([0, 3, 1, 2], in_place=True)
    return N_L


# ---------------------------------------------------------------------------
# Exact environment fill-in (Eq. 15/16)
# ---------------------------------------------------------------------------

def _expand_env_right(
    E_right_j: Tensor,
    A_j: Tensor,
    B_j: Tensor,
    C_j: Tensor,
    N_R: Tensor,
) -> Tensor:
    """Build the expanded right environment at site i (forward sweep).

    Computes two right-environment blocks via `step_right_env` — one from
    the original (exact, uncompressed) `A_j`, `B_j`, `C_j` and one from the
    complement `N_R` — and assembles them block-diagonally via `oplus`.
    Only the `C`-bond axis differs between the two blocks (`A_j`, `B_j` are
    shared), so the direct sum is applied to axis 0 only.

    Parameters
    ----------
    E_right_j:
        Right environment at site j, axes `(C_right, A_right, B_right)`.
    A_j:
        Factor MPO A at site j (uncompressed), axes
        `(left, right, phys_in, phys_internal)`.
    B_j:
        Factor MPO B at site j (uncompressed), axes
        `(left, right, phys_internal_in, phys_out)`.
    C_j:
        Compressed MPO C at site j (uncompressed), axes
        `(left, right, phys_in, phys_out)`.
    N_R:
        Complement tensor, axes `(k, right, phys_in, phys_out)`, treated as
        a `C_j`-shaped tensor for `step_right_env`.

    Returns
    -------
    Tensor
        Expanded right environment at site i, axes
        `(C_right_exp, A_right, B_right)`.
    """
    E_orig = step_right_env(E_right_j, A_j, B_j, C_j)
    E_new = step_right_env(E_right_j, A_j, B_j, N_R)
    return oplus(E_orig, E_new, axes=[0])


def _expand_env_left(
    E_left_i: Tensor,
    A_i: Tensor,
    B_i: Tensor,
    C_i: Tensor,
    N_L: Tensor,
) -> Tensor:
    """Build the expanded left environment at site i (backward sweep).

    Mirror of `_expand_env_right`: computes two left-environment blocks from
    the original `A_i`, `B_i`, `C_i` and the complement `N_L`, then
    assembles them block-diagonally via `oplus` on the `C`-bond axis only.

    Parameters
    ----------
    E_left_i:
        Left environment at site i, axes `(C_left, A_left, B_left)`.
    A_i:
        Factor MPO A at site i (uncompressed), axes
        `(left, right, phys_in, phys_internal)`.
    B_i:
        Factor MPO B at site i (uncompressed), axes
        `(left, right, phys_internal_in, phys_out)`.
    C_i:
        Compressed MPO C at site i (uncompressed), axes
        `(left, right, phys_in, phys_out)`.
    N_L:
        Complement tensor, axes `(left, k, phys_in, phys_out)`, treated as a
        `C_i`-shaped tensor for `step_left_env`.

    Returns
    -------
    Tensor
        Expanded left environment at site i, axes
        `(C_left_exp, A_left, B_left)`.
    """
    E_orig = step_left_env(E_left_i, A_i, B_i, C_i)
    E_new = step_left_env(E_left_i, A_i, B_i, N_L)
    return oplus(E_orig, E_new, axes=[0])


# ---------------------------------------------------------------------------
# Public expansion functions
# ---------------------------------------------------------------------------

def expand_forward(
    A_i: Tensor,
    B_i: Tensor,
    C_i: Tensor,
    A_j: Tensor,
    B_j: Tensor,
    C_j: Tensor,
    E_left_i: Tensor,
    E_right_j: Tensor,
    k_expand: int,
    alpha: Optional[int],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute the CBE complement and expand bond (i, i+1) for a forward sweep.

    The orthogonality center sits at site i. The procedure is:

    1. Cheap per-operand bond compression of `A_j`, `B_j` (Eq. 13).
    2. Factored half-contractions: `left_half` from `E_left_i` and the
       (compressed) `A_i`, `B_i`; `right_half` from `E_right_j` and the
       (compressed) `A_j`, `B_j`.
    3. Discarded-space projection of each half using the original `C_i`,
       `C_j` (Eq. 12).
    4. Truncated SVD of the joined discarded halves → `N_R`, the new
       left-bond directions for `C_j` (Eq. 14).
    5. Exact fill-in at site i using the *uncompressed* `A_j`, `B_j`, `C_j`,
       and `N_R` (Eq. 15/16) — a direct contraction, no eigensolver.
    6. Expansion of `C_j` via `oplus`.

    Parameters
    ----------
    A_i:
        Factor MPO A at site i, axes `(left, right, phys_in, phys_internal)`.
    B_i:
        Factor MPO B at site i, axes `(left, right, phys_internal_in, phys_out)`.
    C_i:
        Compressed MPO C at site i, axes `(left, right, phys_in, phys_out)`.
    A_j:
        Factor MPO A at site j=i+1, same axis convention as `A_i`.
    B_j:
        Factor MPO B at site j=i+1, same axis convention as `B_i`.
    C_j:
        Compressed MPO C at site j=i+1, same axis convention as `C_i`.
    E_left_i:
        Left environment at site i, axes `(C_left, A_left, B_left)`.
    E_right_j:
        Right environment at site j, axes `(C_right, A_right, B_right)`.
    k_expand:
        Maximum number of complement vectors to add to the bond.
    alpha:
        Truncated connector-bond dimension for the cheap compression
        (Eq. 13). `None` skips compression (uses the exact operands).

    Returns
    -------
    Tensor
        `C_i_exp` — new `C_i`, axes `(left, right, phys_in, phys_out)`.
    Tensor
        `C_j_exp` — expanded `C_j`, axes `(left_exp, right, phys_in, phys_out)`.
    Tensor
        `E_right_i_exp` — expanded right environment at site i, axes
        `(C_right_exp, A_right, B_right)`.
    """
    A_i_reduced, A_j_reduced, B_i_reduced, B_j_reduced = _reduce_operands(
        A_i, A_j, B_i, B_j, alpha,
    )

    left_half = left_partial(E_left_i, A_i_reduced, B_i_reduced)
    right_half = right_partial(E_right_j, A_j_reduced, B_j_reduced)

    left_disc = _project_complement_left(left_half, C_i)
    right_disc = _project_complement_right(right_half, C_j)

    N_R = _new_direction_right(left_disc, right_disc, k_expand)

    # Exact fill-in at site i using the uncompressed operands.
    E_right_i_exp = _expand_env_right(E_right_j, A_j, B_j, C_j, N_R)
    C_i_exp = local_update_1s(E_left_i, A_i, B_i, E_right_i_exp)
    C_j_exp = oplus(C_j, N_R, axes=[0])

    return C_i_exp, C_j_exp, E_right_i_exp


def expand_backward(
    A_i: Tensor,
    B_i: Tensor,
    C_i: Tensor,
    A_im1: Tensor,
    B_im1: Tensor,
    C_im1: Tensor,
    E_left_im1: Tensor,
    E_right_i: Tensor,
    k_expand: int,
    alpha: Optional[int],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute the CBE complement and expand bond (i-1, i) for a backward sweep.

    Mirror of `expand_forward` for backward sweeps. The orthogonality
    center sits at site i; the exact fill-in happens at site i, and site
    i-1 is only expanded (it will be finalised in a later sweep step). The
    procedure is:

    1. Cheap per-operand bond compression of `A_im1`, `B_im1` (Eq. 13).
    2. Factored half-contractions: `left_half` from `E_left_im1` and the
       (compressed) `A_im1`, `B_im1`; `right_half` from `E_right_i` and the
       (compressed) `A_i`, `B_i`.
    3. Discarded-space projection using the original `C_im1`, `C_i` (Eq. 12).
    4. Truncated SVD of the joined discarded halves → `N_L`, the new
       right-bond directions for `C_im1` (Eq. 14).
    5. Exact fill-in at site i using the *uncompressed* `A_im1`, `B_im1`,
       `C_im1`, and `N_L` (Eq. 15/16).
    6. Expansion of `C_im1` via `oplus`.

    Parameters
    ----------
    A_i:
        Factor MPO A at site i, axes `(left, right, phys_in, phys_internal)`.
    B_i:
        Factor MPO B at site i, axes `(left, right, phys_internal_in, phys_out)`.
    C_i:
        Compressed MPO C at site i, axes `(left, right, phys_in, phys_out)`.
    A_im1:
        Factor MPO A at site i-1, same axis convention as `A_i`.
    B_im1:
        Factor MPO B at site i-1, same axis convention as `B_i`.
    C_im1:
        Compressed MPO C at site i-1, same axis convention as `C_i`.
    E_left_im1:
        Left environment at site i-1, axes `(C_left, A_left, B_left)`.
    E_right_i:
        Right environment at site i, axes `(C_right, A_right, B_right)`.
    k_expand:
        Maximum number of complement vectors to add to the bond.
    alpha:
        Truncated connector-bond dimension for the cheap compression
        (Eq. 13). `None` skips compression (uses the exact operands).

    Returns
    -------
    Tensor
        `C_i_exp` — new `C_i`, axes `(left, right, phys_in, phys_out)`.
    Tensor
        `C_im1_exp` — expanded `C_im1`, axes `(left, right_exp, phys_in, phys_out)`.
    Tensor
        `E_left_i_exp` — expanded left environment at site i, axes
        `(C_left_exp, A_left, B_left)`.
    """
    A_im1_reduced, A_i_reduced, B_im1_reduced, B_i_reduced = _reduce_operands(
        A_im1, A_i, B_im1, B_i, alpha,
    )

    left_half = left_partial(E_left_im1, A_im1_reduced, B_im1_reduced)
    right_half = right_partial(E_right_i, A_i_reduced, B_i_reduced)

    left_disc = _project_complement_left(left_half, C_im1)
    right_disc = _project_complement_right(right_half, C_i)

    N_L = _new_direction_left(left_disc, right_disc, k_expand)

    # Exact fill-in at site i using the uncompressed operands.
    E_left_i_exp = _expand_env_left(E_left_im1, A_im1, B_im1, C_im1, N_L)
    C_i_exp = local_update_1s(E_left_i_exp, A_i, B_i, E_right_i)
    C_im1_exp = oplus(C_im1, N_L, axes=[1])

    return C_i_exp, C_im1_exp, E_left_i_exp
