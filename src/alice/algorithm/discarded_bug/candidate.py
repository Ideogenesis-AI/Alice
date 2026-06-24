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
# Author of code: Madhav Menon.


"""Local two-site update of the discarded-projector BUG integrator.

One rank-adaptive Basis-Update & Galerkin (BUG) step on a single bond, with the
basis growth driven by the **discarded** (orthogonal-complement) projector and
**without** ever forming the augmented overlap matrices ``M``, ``N``.

State on a bond: ``Theta0 = U0 . S0 . V0`` with ``U0`` left-isometric on
``(link_l, site_l)``, ``V0`` right-isometric on ``(site_r, link_r)``, and ``S0``
the center. The effective Hamiltonian enters through the MPO **environments**: the
two-site action is ``H = E_left . W_i . W_{i+1} . E_right`` (the DMRG
:func:`~alice.algorithm.dmrg.scheme_2s.matvec_2s`), so unlike a bare-gate TEBD
update the local generator sees the whole chain through the environments and there
is no Trotter splitting error.

The update (:func:`block_local_update`)
    1. **Evolve the two-site block once** under the two-site effective Hamiltonian,
       ``Theta1 = exp(tau H) Theta0`` (Hermitian, so the Lanczos exponential).
    2. **Grow the frames from** ``Theta1``: the augmented left isometry is
       ``U_aug = qr([colspace(Theta1 | link_l, site_l) | U0])`` and the augmented
       right isometry is ``V_aug = qr([rowspace(Theta1 | link_r, site_r) ; V0])`` —
       the discarded-projector direct sum (the leading ``U0``/``V0`` keep the old
       frame exactly inside; the QR drops dependent columns so a saturated leg gives
       no spurious growth). No overlap matrix ``M``/``N`` is built.
    3. **Project the evolved block** onto the augmented frames for the Galerkin core
       ``S = U_aug+ Theta1 V_aug+`` (the time evolution is already in ``Theta1``;
       there is no separate S-step), then **SVD-truncate** to ``maxdim`` / ``cutoff``
       to set the new (possibly larger) bond rank.

Why grow from the evolved block
    Acting with ``H`` on the two-site window is what creates the new Schmidt
    direction — a domain-wall interface block ``Theta1`` has Schmidt rank 2, so the
    bond *must* grow ``1 -> 2`` in one step. A generator that froze a neighbour at
    the old rank-deficient frame (projecting ``H Theta0`` onto ``V0 V0+``) would
    annihilate exactly that direction, because the new content is orthogonal to the
    old single-state frame. Reading the frames off the full ``Theta1`` keeps the
    physical legs free, so the genuine entanglement growth survives. This is the
    two-site analogue of the reference leaf basis-update (which keeps the leaf's
    physical leg open and only projects the *other* subtrees' bonds).

Forward-only / inverse-free
    A single block evolution and a single truncation, with **no** backward (``-tau``)
    substep and no overlap-matrix inverse — BUG is inverse-free by design. The
    growth and accuracy come entirely from the discarded-projector augmentation and
    the Galerkin core.

Everything stays in the symmetry-blocked Nicole representation (the QR, the SVD,
the direct sum via :func:`nicole.oplus`), so the U(1) charge sectors are respected
throughout — a dense standard-basis step would mix sectors and be rejected.

Index conventions
    * ``U0``    : ``(link_l, site_l, mid_u)`` — left-isometric over ``(link_l, site_l)``
    * ``V0``    : ``(mid_v, link_r, site_r)`` — right-isometric over ``(link_r, site_r)``
    * ``S0``    : ``(mid_u, mid_v)``
    * theta (for ``matvec_2s``) : ``(link_l, link_r, site_l, site_r)``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from nicole import Tensor, conj, contract, decomp, oplus

from ._krylov import tensor_lanczos_expv


# ---------------------------------------------------------------------------
# Bond snapshot
# ---------------------------------------------------------------------------

@dataclass
class BondSnapshot:
    """Canonical two-site window extracted for one local update.

    Attributes
    ----------
    U0:
        Left isometry with axes ``(link_l, site_l, mid_u)``.
    V0:
        Right isometry with axes ``(mid_v, link_r, site_r)``.
    S0:
        Center with axes ``(mid_u, mid_v)``.
    bond_itag:
        itag carried by the internal bond of this two-site window (used to tag
        the new isometries and the truncated bond).
    """

    U0: Tensor
    V0: Tensor
    S0: Tensor
    bond_itag: str


def bond_snapshot(left_core: Tensor, right_core: Tensor, bond_itag: str) -> BondSnapshot:
    """Split two adjacent MPS cores into a canonical ``(U0, S0, V0)`` window.

    Mirrors the Julia ``_canonical_quantum_bond_snapshot``: a QR-like split of the
    left core exposes a left isometry ``U0`` and a left carry, an LQ-like split of
    the right core exposes a right isometry ``V0`` and a right carry, and the two
    carries contract over the shared bond to give the center ``S0``. Both splits
    are computed with Nicole's ``UR`` decomposition (``U``/``V`` isometric, the
    singular values folded into the carry), so the U(1) sectors are preserved.

    Parameters
    ----------
    left_core:
        MPS tensor at site ``i`` with axes ``(link_l, bond, site_l)``.
    right_core:
        MPS tensor at site ``i+1`` with axes ``(bond, link_r, site_r)`` whose left
        bond shares the itag of ``left_core``'s right bond.
    bond_itag:
        itag to assign to the internal ``mid_u`` bond of the left isometry.

    Returns
    -------
    BondSnapshot
        The canonical window ``(U0, S0, V0)`` with the conventions in the module
        docstring.
    """
    # Canonicalise the two-site window by QR/LQ, matching the reference
    # ``_canonical_quantum_bond_snapshot`` (QR of the left core, LQ of the right):
    # the upper-/lower-triangular factors define the gauge that is transported as
    # the orthogonality center moves along the chain.
    #
    # Left core (link_l, bond, site_l): QR separating (link_l, site_l) onto the Q
    # side -> U0 = (link_l, site_l, mid_u) left-isometric, R = (mid_u, bond).
    u0, left_carry = decomp(left_core, axes=[0, 2], mode='QR', itag=bond_itag)
    # Right core (bond, link_r, site_r): QR separating (link_r, site_r) onto the Q
    # side (an LQ of the right core) -> Viso = (link_r, site_r, mid_v) right-iso,
    # R = (mid_v, bond).
    v_iso, right_carry = decomp(right_core, axes=[1, 2], mode='QR', itag=bond_itag + '_v')
    v0 = v_iso.permute([2, 0, 1])  # (mid_v, link_r, site_r)
    # Center S0 = left_carry . right_carry contracted over the shared bond
    # (left_carry axis 1, right_carry axis 1) -> (mid_u, mid_v).
    s0 = contract(left_carry, right_carry, axes=([1], [1]))
    return BondSnapshot(U0=u0, V0=v0, S0=s0, bond_itag=bond_itag)


# ---------------------------------------------------------------------------
# Discarded-projector augmented isometries
# ---------------------------------------------------------------------------

def _augmented_left_isometry(u0: Tensor, k1: Tensor) -> Tuple[Tensor, int]:
    """Grow the left frame by constructing the augmented basis ``[K1 | U0]``.

    This is the rank-adaptive Basis-Update step of the tree/MPS BUG integrator: the
    augmented left isometry spans both the old frame and the freshly evolved ``K1``,

        ``U_aug = orthonormalize([ colspace(K1) | U0 ])``  (over ``(link_l, site_l)``),

    so a new direction is admitted wherever the time-evolved ``K1`` has left
    ``span(U0)``. We **construct the augmented basis** (this concatenation + QR) but
    never the augmented *projectors*: no ``M = U_aug+ U0`` overlap matrix is formed —
    the augmented core is obtained later by projecting the state directly onto the
    augmented frames (see :func:`center_sstep`). The leading ``[K1 | U0]`` ordering
    keeps ``U0`` exactly inside ``U_aug``.

    ``K1`` is QR'd first so its column-space isometry shares the outgoing bond
    direction of ``U0`` before the direct sum; the final QR over ``(link_l, site_l)``
    drops dependent columns (so a saturated ``(link_l, site_l)`` space yields no
    spurious growth) and restores an exact isometry. No augmentation tolerance is
    applied — the QR's machine-precision rank detection sets the admitted directions,
    and the only explicit rank control is the post-S-step SVD truncation.

    Parameters
    ----------
    u0:
        Old left isometry with axes ``(link_l, site_l, mid_u)``.
    k1:
        Integrated K tensor with axes ``(link_l, site_l, mid_v)``.

    Returns
    -------
    Tensor
        Augmented left isometry ``U_aug`` with axes ``(link_l, site_l, mid_aug)``.
    int
        Number of new columns added (``mid_aug - mid_u``).
    """
    old_rank = u0.indices[2].dim
    # colspace(K1): QR over (link_l, site_l) so K1's basis shares U0's bond direction.
    qk, _ = decomp(k1, axes=[0, 1], mode='QR', itag=u0.itags[2])
    # Augmented basis [colspace(K1) | U0], re-orthonormalised by a final QR that drops
    # dependent columns (no growth where (link_l, site_l) is already saturated).
    u_aug, _ = decomp(oplus(qk, u0, axes=2), axes=[0, 1], mode='QR', itag=u0.itags[2])
    return u_aug, u_aug.indices[2].dim - old_rank


def _augmented_right_isometry(v0: Tensor, l1: Tensor) -> Tuple[Tensor, int]:
    """Grow the right frame by constructing the augmented basis ``[L1 ; V0]``.

    Mirror of :func:`_augmented_left_isometry` on the right frame: the augmented
    right isometry spans both the old frame and the evolved ``L1``,

        ``V_aug = orthonormalize([ rowspace(L1) ; V0 ])``  (over ``(link_r, site_r)``),

    constructing the augmented basis (concatenation + QR) but never the augmented
    overlap matrices. ``L1`` is QR'd over ``(link_r, site_r)`` first so its
    row-space isometry shares ``V0``'s bond direction; the final QR drops dependent
    rows (no growth where ``(link_r, site_r)`` is saturated). No augmentation
    tolerance.

    Parameters
    ----------
    v0:
        Old right isometry with axes ``(mid_v, link_r, site_r)``.
    l1:
        Integrated L tensor with axes ``(mid_u, link_r, site_r)``.

    Returns
    -------
    Tensor
        Augmented right isometry ``V_aug`` with axes ``(mid_aug, link_r, site_r)``.
    int
        Number of new rows added.
    """
    old_rank = v0.indices[0].dim
    # rowspace(L1): QR over (link_r, site_r) gives Ql as (link_r, site_r, mid_new);
    # reorder to the right-isometry convention (mid_new, link_r, site_r).
    ql_iso, _ = decomp(l1, axes=[1, 2], mode='QR', itag=v0.itags[0])
    ql = ql_iso.permute([2, 0, 1])
    # Augmented basis [rowspace(L1) ; V0], re-orthonormalised by a final QR that drops
    # dependent rows (no growth where (link_r, site_r) is already saturated).
    v_sum = oplus(ql, v0, axes=0)
    v_aug_iso, _ = decomp(v_sum, axes=[1, 2], mode='QR', itag=v0.itags[0])
    v_aug = v_aug_iso.permute([2, 0, 1])
    return v_aug, v_aug.indices[0].dim - old_rank


# ---------------------------------------------------------------------------
# Local update
# ---------------------------------------------------------------------------

@dataclass
class LocalUpdate:
    """Result of one discarded-BUG local update on a bond.

    Attributes
    ----------
    left_core:
        New left core with axes ``(link_l, kept, site_l)`` (left-isometric).
    right_core:
        New right core with axes ``(kept, link_r, site_r)`` carrying the singular
        values (the orthogonality center after a forward step, or the right
        isometry after a reverse step, depending on the sweep).
    n_new_left:
        Number of directions the K-step added to the left frame.
    n_new_right:
        Number of directions the L-step added to the right frame.
    kept:
        New bond dimension after the SVD truncation.
    svals:
        Kept singular values per charge sector (concatenated, descending).
    """

    left_core: Tensor
    right_core: Tensor
    n_new_left: int
    n_new_right: int
    kept: int
    svals: torch.Tensor


def _two_site_apply(
    theta_left_right_phys: Tensor,
    W_i: Tensor,
    W_i1: Tensor,
    E_left: Tensor,
    E_right: Tensor,
) -> Tensor:
    """Apply the two-site effective Hamiltonian, in the local axis order.

    The local update keeps tensors in ``(link, ..., site)`` order, whereas
    :func:`~alice.algorithm.dmrg.scheme_2s.matvec_2s` expects and returns the
    DMRG bond order ``(link_l, link_r, site_l, site_r)``. This helper permutes in,
    applies ``matvec_2s``, and permutes back, so callers can build theta from the
    snapshot factors without worrying about the DMRG convention.

    Parameters
    ----------
    theta_left_right_phys:
        Bond tensor with axes ``(link_l, site_l, link_r, site_r)``.
    W_i, W_i1:
        MPO tensors at sites ``i`` and ``i+1``.
    E_left, E_right:
        Left/right MPO environments bracketing the two-site window.

    Returns
    -------
    Tensor
        ``H|theta>`` with axes ``(link_l, site_l, link_r, site_r)``.
    """
    from ..dmrg.scheme_2s import matvec_2s

    # (link_l, site_l, link_r, site_r) -> (link_l, link_r, site_l, site_r)
    theta = theta_left_right_phys.permute([0, 2, 1, 3])
    out = matvec_2s(theta, W_i, W_i1, E_left, E_right)   # (link_l, link_r, site_l, site_r)
    return out.permute([0, 2, 1, 3])                     # back to local order


def block_local_update(
    snapshot: BondSnapshot,
    W_i: Tensor,
    W_i1: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    tau: complex,
    *,
    maxdim: int,
    cutoff: float,
    lanczos_tol: float,
    lanczos_maxiter: int,
) -> LocalUpdate:
    """One discarded-BUG local update on a bond, growing the basis from the evolved block.

    This is the rank-adaptive Basis-Update & Galerkin (BUG) step in its faithful
    two-site form. The two-site block is evolved **once** under the two-site
    effective Hamiltonian,

        ``Theta1 = exp(tau * H_2site) . Theta0``     (``H_2site = E_left W_i W_{i+1} E_right``),

    and the augmented frames are read directly off ``Theta1``: the left frame from its
    ``(link_l, site_l)`` column space and the right frame from its ``(link_r, site_r)``
    row space, each direct-summed onto the old frame with the **discarded** projector
    (``U_aug = qr([Theta1_left | U0])`` / ``V_aug = qr([Theta1_right | V0])`` — never an
    ``M``/``N`` overlap matrix). The Galerkin core is then the projection of the already
    evolved block, ``S = U_aug+ Theta1 V_aug+``, which is SVD-truncated to set the new
    bond rank.

    Why grow from the evolved block (and not a frozen-neighbour generator)
        Acting with ``H`` on the two-site window is what creates the new Schmidt
        direction: for a domain-wall product state the interface block ``Theta1`` has
        Schmidt rank 2, so the bond *must* grow ``1 -> 2`` in one step. A K-step that
        froze the right subsystem at the old single-state frame ``V0`` (i.e. projected
        ``H Theta0`` onto ``V0 V0+``) would annihilate exactly that direction, because
        the new right content is orthogonal to ``V0``. Reading the frames off the full
        ``Theta1`` keeps the physical legs free, so the genuine entanglement growth
        survives — this is the two-site analogue of the reference leaf basis-update
        (which keeps the leaf's physical leg open and only projects the *other* subtrees'
        bonds).

    The update is forward-only: a single block evolution and a single truncation, with
    **no** backward ``-tau`` substep. The growth and accuracy come entirely from the
    discarded-projector basis augmentation and the Galerkin core, never from an inverse.

    Parameters
    ----------
    snapshot:
        Canonical ``(U0, S0, V0)`` window from :func:`bond_snapshot`.
    W_i, W_i1:
        MPO tensors at sites ``i`` and ``i+1``.
    E_left, E_right:
        Left/right MPO environments bracketing the two-site window.
    tau:
        Substep generator coefficient (``prefactor * dt``).
    maxdim, cutoff:
        SVD truncation controls for the new bond rank.
    lanczos_tol, lanczos_maxiter:
        Krylov termination tolerance and maximum dimension for the block evolution.

    Returns
    -------
    LocalUpdate
        New left/right cores (``left`` left-isometric, ``right`` carries the singular
        values) and rank-adaptivity diagnostics.
    """
    u0, v0, s0 = snapshot.U0, snapshot.V0, snapshot.S0
    theta0 = contract(contract(u0, s0, axes=([2], [0])), v0, axes=([2], [0]))

    # Evolve the two-site block once under the two-site effective Hamiltonian (Hermitian
    # -> Lanczos exponential). This is the single forward evolution of the BUG step.
    def apply_h(theta: Tensor) -> Tensor:
        return _two_site_apply(theta, W_i, W_i1, E_left, E_right)

    theta1 = tensor_lanczos_expv(apply_h, tau, theta0, maxiter=lanczos_maxiter, tol=lanczos_tol)

    # Augmented LEFT frame from Theta1's (link_l, site_l) column space, discarded-summed
    # onto U0. The QR isolates the column space; _augmented_left_isometry appends U0.
    k_left, _ = decomp(theta1, axes=[0, 1], mode='QR', itag=u0.itags[2])
    u_aug, n_new_left = _augmented_left_isometry(u0, k_left)
    # Augmented RIGHT frame from Theta1's (link_r, site_r) row space, discarded-summed
    # onto V0. r_right is (link_r, site_r, new) -> reorder to (new, link_r, site_r).
    r_right, _ = decomp(theta1, axes=[2, 3], mode='QR', itag=v0.itags[0])
    v_aug, n_new_right = _augmented_right_isometry(v0, r_right.permute([2, 0, 1]))

    # Galerkin core = projection of the already-evolved block onto the augmented frames
    # (no separate S-step: the time evolution is in Theta1).
    s_left = contract(conj(u_aug), theta1, axes=([0, 1], [0, 1]))   # (mid_aug_u, link_r, site_r)
    s_new = contract(s_left, conj(v_aug), axes=([1, 2], [1, 2]))    # (mid_aug_u, mid_aug_v)

    return _truncate_and_assemble(
        u_aug, v_aug, s_new, snapshot.bond_itag,
        maxdim=maxdim, cutoff=cutoff,
        n_new_left=n_new_left, n_new_right=n_new_right,
    )


def _truncate_and_assemble(
    u_aug: Tensor,
    v_aug: Tensor,
    s_new: Tensor,
    bond_itag: str,
    *,
    maxdim: int,
    cutoff: float,
    n_new_left: int,
    n_new_right: int,
) -> LocalUpdate:
    """SVD-truncate the evolved core and re-absorb it into the augmented frames.

    The augmented-basis core ``s_new`` is decomposed ``s_new = U_s . S . Vh``,
    truncated to ``maxdim`` / ``cutoff``, and folded back: ``left = U_aug . U_s``
    (left-isometric) and ``right = (S Vh) . V_aug`` (carries the singular values).
    The truncation runs in the symmetry-blocked representation, so the kept rank
    respects the U(1) sectors.

    Parameters
    ----------
    u_aug, v_aug:
        Augmented left/right isometries from the K/L steps.
    s_new:
        Evolved augmented-basis core with axes ``(mid_aug_u, mid_aug_v)``.
    bond_itag:
        itag to assign to the truncated internal bond.
    maxdim:
        Maximum kept bond dimension.
    cutoff:
        Relative singular-value threshold.
    n_new_left, n_new_right:
        Rank-adaptivity diagnostics carried through to the result.

    Returns
    -------
    LocalUpdate
        The assembled cores and diagnostics.
    """
    trunc = {'nkeep': int(maxdim), 'thresh': max(float(cutoff), 0.0)}
    u_s, s_diag, vh = decomp(s_new, axes=0, mode='SVD', itag=(bond_itag, bond_itag), trunc=trunc)

    # left = U_aug . U_s  ->  (link_l, site_l, kept)
    left = contract(u_aug, u_s, axes=([2], [0]))
    # right = (S . Vh) . V_aug  ->  (kept, link_r, site_r)
    s_vh = contract(s_diag, vh, axes=([1], [0]))
    right = contract(s_vh, v_aug, axes=([1], [0]))

    # Re-order to the MPS core convention (link_left, link_right, physical).
    left = left.permute([0, 2, 1])   # (link_l, kept, site_l)
    # right is already (kept, link_r, site_r) = (link_left, link_right, physical).

    kept = left.indices[1].dim
    svals = _singular_values(s_diag)
    return LocalUpdate(
        left_core=left,
        right_core=right,
        n_new_left=n_new_left,
        n_new_right=n_new_right,
        kept=kept,
        svals=svals,
    )


def _singular_values(s_diag: Tensor) -> torch.Tensor:
    """Return the singular values held on the diagonal of ``s_diag``, descending."""
    values = []
    for block in s_diag.data.values():
        diag = torch.diagonal(block).abs().to(torch.float64)
        values.append(diag)
    if not values:
        return torch.zeros(0, dtype=torch.float64)
    return torch.sort(torch.cat(values), descending=True).values


# Re-exported for the global forward sweep (:mod:`~alice.algorithm.discarded_bug.sweep`).
__all__ = [
    'BondSnapshot',
    'LocalUpdate',
    'bond_snapshot',
    'block_local_update',
]
