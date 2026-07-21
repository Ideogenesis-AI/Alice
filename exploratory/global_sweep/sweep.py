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


"""One step of the rank-adaptive discarded-projector BUG integrator (Sulz Alg. 5–7).

This is the MPS realisation of the **rank-adaptive tree-tensor-network BUG** of
Ceruti–Lubich–Walach / Sulz (thesis, Algorithms 5–7), specialised to the linear
(MPS) tree. The two defining choices are:

* the basis growth is driven by the **discarded** (orthogonal-complement) projector,
  applied **explicitly** (``P_perp = I - U0 U0+``) and **per basis matrix**, never by
  forming the augmented overlap matrices ``M``/``N``; and
* the augmentation direction is read from the **full** Hamiltonian image
  ``phi = H · psi`` (computed once as an MPS), **not** from a local two-site block —
  this is what makes the augmented bases span ``range(psi) ⊕ range(H psi)`` (the exact
  rank-adaptive BUG basis) rather than a local approximation.

The step (:func:`global_step`)
------------------------------
1. **Image.** Form ``phi = H · psi`` as an MPS (:func:`mpo_times_mps`).
2. **K-sweep** (left→right, :func:`k_sweep`). Build the augmented **left** isometries
   ``W_i``. At each bond keep ``psi``'s left frame *exactly* (``U0 = qr(psi part)``),
   then admit the discarded part of ``phi``'s frame, ``(I - U0 U0+) phi``, SVD-truncating
   **only that complement** to the remaining budget ``maxdim - rank(U0)``. Keeping ``psi``
   exact is what makes truncation rank-*stable*: ``psi``'s own directions can never be
   dropped (a plain SVD of ``psi ⊞ phi`` can, and then whole charge sectors collapse).
3. **L-sweep** (right→left, :func:`l_sweep`). The mirror image: augmented **right**
   isometries ``Z_i``.
4. **Galerkin core (Alg. 7).** ``psi`` is projected onto the augmented frames to seed
   the connecting tensor ``S_start = ⟨W, Z | psi⟩`` (built from the sweep carries, with
   **no** ``M``/``N`` overlap matrices), and the single centre connecting tensor is
   integrated over the full step under the two-site effective Hamiltonian
   ``E_left . W_c . W_{c+1} . E_right``. This is the only time evolution in the step.
5. **Truncate & assemble** the new centre and return the orthogonality centre to site 0.

Forward-only / inverse-free
    A single Galerkin core evolution and a single truncation, with **no** backward
    ``-tau`` substep and no overlap-matrix inverse — BUG is inverse-free by design. At
    full bond dimension the step is *exact* (the two-site Galerkin core is lossless);
    truncation introduces a rank-adaptive error that converges monotonically as the bond
    dimension is raised. Richardson extrapolation lifts the time order when required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from nicole import Direction, Tensor, conj, contract, decomp, einsum, merge_axes, oplus

from alice.network import MPS, MPO

from ..dmrg.environ import (
    left_env_boundary,
    right_env_boundary,
    step_left_env,
    step_right_env,
)
from ..two_site_bug._kernel.local_solvers import local_expv
from ._krylov import to_complex


def mpo_times_mps(mpo: MPO, mps: MPS) -> MPS:
    """Apply the Hamiltonian MPO to ``mps`` and return ``phi = H · psi`` as an MPS.

    Each site contracts the MPS core ``A = (link_l, link_r, phys)`` with the MPO core
    ``W = (W_l, W_r, ket, bra)`` and merges the paired virtual legs into single bonds.
    The merged right bond is given ``Direction.IN`` and the merged left bond
    ``Direction.OUT`` so the result carries ``psi``'s bond convention; the interior bond
    itags and every physical itag are retagged to ``psi``'s, and the two trivial boundary
    bonds are aligned (itag **and** direction) to ``psi`` so ``phi`` and ``psi`` are
    contractible site-by-site in the sweeps.
    """
    L = mps.L
    cores: List[Tensor] = []
    for s in range(L):
        c = einsum('lrk,pqbk->lprqb', mps[s], mpo[s])           # (l, Wl, r, Wr, bra)
        c, _ = merge_axes(c, [2, 3], merged_tag='_phiR', direction=Direction.IN)
        c, _ = merge_axes(c, [1, 2], merged_tag='_phiL', direction=Direction.OUT)
        cores.append(c)
    for s in range(L - 1):
        tag = mps._bond_itag(s + 1)
        cores[s].retag(1, tag)
        cores[s + 1].retag(0, tag)
    for s in range(L):
        cores[s].retag(2, mps[s].itags[2])
    cores[0].retag(0, mps[0].itags[0])
    cores[L - 1].retag(1, mps[L - 1].itags[1])
    if cores[0].indices[0].direction != mps[0].indices[0].direction:
        cores[0].invert(0)
    if cores[L - 1].indices[1].direction != mps[L - 1].indices[1].direction:
        cores[L - 1].invert(1)
    return MPS(cores, center=None)


def _trunc(maxdim: int, cutoff: float) -> Dict:
    return {'nkeep': int(maxdim), 'thresh': max(float(cutoff), 0.0)}


# ---------------------------------------------------------------------------
# Two-site effective-Hamiltonian apply and centre truncation/assembly
# ---------------------------------------------------------------------------

@dataclass
class LocalUpdate:
    """Result of the Galerkin centre update of one discarded-BUG step.

    Attributes
    ----------
    left_core:
        New left core with axes ``(link_l, kept, site_l)`` (left-isometric).
    right_core:
        New right core with axes ``(kept, link_r, site_r)`` carrying the singular
        values (the orthogonality center).
    n_new_left, n_new_right:
        Number of directions the K/L sweeps added (carried through as diagnostics).
    kept:
        New centre bond dimension after the SVD truncation.
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

    The sweep keeps tensors in ``(link, ..., site)`` order, whereas
    :func:`~alice.algorithm.dmrg.scheme_2s.matvec_2s` expects and returns the DMRG
    bond order ``(link_l, link_r, site_l, site_r)``. This helper permutes in, applies
    ``matvec_2s``, and permutes back.

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
    """SVD-truncate the evolved centre and re-absorb it into the augmented frames.

    The augmented-basis core ``s_new`` is decomposed ``s_new = U_s . S . Vh``,
    truncated to ``maxdim`` / ``cutoff``, and folded back: ``left = U_aug . U_s``
    (left-isometric) and ``right = (S Vh) . V_aug`` (carries the singular values).
    The truncation runs in the symmetry-blocked representation, so the kept rank
    respects the U(1) sectors.

    Parameters
    ----------
    u_aug, v_aug:
        Augmented left/right isometries from the K/L sweeps.
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


def _discarded_weight(s_new: Tensor, bond_itag: str, kept: int) -> float:
    """Relative Frobenius weight discarded when the centre core is cut to ``kept``.

    Full (untruncated) SVD spectrum of the evolved centre ``s_new`` vs the kept
    leading ``kept`` values: ``sqrt(sum_{i>=kept} sigma_i^2 / sum_i sigma_i^2)`` — the
    standard MPS discarded-weight diagnostic for this step's truncation.
    """
    _, s_full, _ = decomp(s_new, axes=0, mode='SVD', itag=(bond_itag, bond_itag))
    sv = _singular_values(s_full)
    total = float((sv ** 2).sum())
    if total == 0.0 or kept >= sv.numel():
        return 0.0
    tail = float((sv[kept:] ** 2).sum())
    return (tail / total) ** 0.5


def k_sweep(
    psi: MPS, phi: MPS, c: int, maxdim: int, cutoff: float, aug_cutoff: float | None = None,
) -> Tuple[List[Tensor], Tensor, Tensor]:
    """Build the augmented **left** isometries ``W_0 … W_c`` (discarded-projector, per matrix).

    Sweeping left→right, the running carries ``aps``/``aph`` express ``psi``/``phi``'s
    current bond in the augmented left frame. At each site the augmented core is

        ``W_i = [ U0 | orthonormalize((I - U0 U0+) phi_part) ]``,

    where ``U0 = qr(psi_part)`` keeps ``psi``'s frame exactly and only the **discarded**
    part of ``phi`` is admitted, SVD-truncated to the remaining budget ``maxdim - rank(U0)``.
    No augmented overlap matrix is formed.

    ``aug_cutoff`` (when not ``None``) sets the singular-value threshold for *admitting*
    the discarded complement, decoupled from the final centre-truncation ``cutoff``: a
    looser ``aug_cutoff`` admits fewer new directions and so caps the augmented bond
    growth (the global analogue of the two-site BUG ``kl_cutoff``). When ``None`` the
    augmentation reuses ``cutoff`` (original behaviour).

    Returns the list ``[W_0 … W_c]`` (MPS-core order ``(link_l, link_r, phys)``) and the
    final carries ``aps = ⟨W | psi⟩`` and ``aph = ⟨W | phi⟩`` at bond ``c`` (shape
    ``(aug_c, psi_bond_c)`` / ``(aug_c, phi_bond_c)``).
    """
    aug_thresh = cutoff if aug_cutoff is None else aug_cutoff
    frames: List[Tensor] = []
    aps = aph = None
    for i in range(0, c + 1):
        bt = psi._bond_itag(i + 1)
        if aps is None:                                            # (link_l, phys, bond_i)
            psit = psi[i].permute([0, 2, 1])
            phit = phi[i].permute([0, 2, 1])
        else:                                                      # (aug_prev, phys, bond_i)
            psit = contract(aps, psi[i], axes=([1], [0])).permute([0, 2, 1])
            phit = contract(aph, phi[i], axes=([1], [0])).permute([0, 2, 1])
        u0, _ = decomp(psit, axes=[0, 1], mode='QR', itag=bt)      # keep psi frame exactly
        rpsi = u0.indices[2].dim
        proj = contract(conj(u0), phit, axes=([0, 1], [0, 1]))     # U0+ phi
        phi_perp = phit - contract(u0, proj, axes=([2], [0]))      # (I - U0 U0+) phi
        w = u0
        # Augment to 2r (budget = rpsi, Sulz Alg. 5): admit r extra discarded phi
        # directions so the propose bases can span new (incl. high-|charge|) sectors.
        # Capping off-central frames at maxdim-rpsi instead RE-STARVES the augmentation
        # and wrecks cooling (measured L=10 7.7e-2, L=20 1.26 vs 1.47e-3 here).
        budget = rpsi
        if budget > 0:
            q, _, _ = decomp(phi_perp, axes=[0, 1], mode='SVD', itag=(bt, bt),
                             trunc=_trunc(budget, aug_thresh))
            if q.indices[2].dim > 0:
                w, _ = decomp(oplus(u0, q, axes=2), axes=[0, 1], mode='QR', itag=bt)
        aps = contract(conj(w), psit, axes=([0, 1], [0, 1]))       # (aug_i, psi_bond_i)
        aph = contract(conj(w), phit, axes=([0, 1], [0, 1]))       # (aug_i, phi_bond_i)
        frames.append(w.permute([0, 2, 1]))                        # (link_l, aug_i, phys)
    return frames, aps, aph


def l_sweep(
    psi: MPS, phi: MPS, c: int, maxdim: int, cutoff: float, aug_cutoff: float | None = None,
) -> Tuple[Dict[int, Tensor], Tensor, Tensor]:
    """Build the augmented **right** isometries ``Z_{c+1} … Z_{L-1}`` (mirror of :func:`k_sweep`).

    Sweeping right→left with carries ``bps``/``bph`` (shape ``(psi_bond, aug)`` /
    ``(phi_bond, aug)``), each augmented right core keeps ``psi``'s right frame exactly and
    admits only the discarded part of ``phi``. ``aug_cutoff`` decouples the admission
    threshold from the final ``cutoff`` (see :func:`k_sweep`). Returns ``{i: Z_i}``
    (MPS-core order ``(link_l, link_r, phys)``) and the final carries at bond ``c+1``.
    """
    aug_thresh = cutoff if aug_cutoff is None else aug_cutoff
    L = psi.L
    frames: Dict[int, Tensor] = {}
    bps = bph = None
    for i in range(L - 1, c, -1):
        bt = psi._bond_itag(i)
        if bps is None:                                            # (bond_l, phys, link_r)
            psit = psi[i].permute([0, 2, 1])
            phit = phi[i].permute([0, 2, 1])
        else:                                                      # (bond_l, phys, aug_next)
            psit = contract(psi[i], bps, axes=([1], [0]))
            phit = contract(phi[i], bph, axes=([1], [0]))
        v0, _ = decomp(psit, axes=[1, 2], mode='QR', itag=bt)      # (phys, aug_next, rpsi)
        rpsi = v0.indices[2].dim
        proj = contract(phit, conj(v0), axes=([1, 2], [0, 1]))     # phi V0+
        phi_perp = phit - contract(proj, v0, axes=([1], [2]))      # phi (I - V0+ V0)
        v = v0
        # Mirror of k_sweep: augment to 2r (budget = rpsi). See note there.
        budget = rpsi
        if budget > 0:
            q, _, _ = decomp(phi_perp, axes=[1, 2], mode='SVD', itag=(bt, bt),
                             trunc=_trunc(budget, aug_thresh))     # (phys, aug_next, rphi)
            if q.indices[2].dim > 0:
                v, _ = decomp(oplus(v0, q, axes=2), axes=[0, 1], mode='QR', itag=bt)
        bps = contract(psit, conj(v), axes=([1, 2], [0, 1]))       # (psi_bond_l, aug_i)
        bph = contract(phit, conj(v), axes=([1, 2], [0, 1]))       # (phi_bond_l, aug_i)
        frames[i] = v.permute([2, 1, 0])                           # (aug_i, aug_next, phys)
    return frames, bps, bph


def global_step(
    mps: MPS,
    mpo: MPO,
    tau: complex,
    *,
    maxdim: int,
    cutoff: float,
    lanczos_tol: float,
    lanczos_maxiter: int,
    solver: str = 'krylov',
    solver_substeps: int = 1,
    aug_cutoff: float | None = None,
) -> int:
    """Advance ``mps`` by one rank-adaptive discarded-projector BUG step (Sulz Alg. 5–7).

    Forms ``phi = H · psi``, builds the augmented left/right isometries by the per-matrix
    discarded-projector sweeps (keeping ``psi`` exact and admitting only ``phi``'s
    complement), integrates the single Galerkin centre connecting tensor over the full
    step, truncates, and returns the orthogonality centre to site 0. At full bond
    dimension the step is exact; the truncation error converges monotonically as
    ``maxdim`` is raised. ``mps`` is modified in place.

    Parameters
    ----------
    mps:
        State to evolve in place. Canonical at site 0 on entry (the caller guarantees it);
        canonical at site 0 on return.
    mpo:
        Hamiltonian MPO of the same length.
    tau:
        Generator coefficient ``prefactor * dt`` (``-1j*dt`` real time, ``-dt`` imaginary).
    maxdim:
        Maximum bond dimension kept by the per-bond SVD truncation.
    cutoff:
        Relative singular-value threshold of the final centre-core SVD truncation.
    aug_cutoff:
        Optional separate threshold for admitting the discarded complement in the
        K/L sweeps; ``None`` reuses ``cutoff``. Decouples augmentation growth from
        the final truncation (the global analogue of two-site ``kl_cutoff``).
    lanczos_tol, lanczos_maxiter:
        Krylov termination tolerance and maximum dimension for the Galerkin core solve.

    Returns
    -------
    tuple[int, float, int, int]
        The maximum kept bond dimension after the step, the relative discarded
        weight of the centre-core SVD truncation (the standard rank-adaptation
        diagnostic), and the proposed augmented **K** (``mid_u``) and **L**
        (``mid_v``) central-window bond dimensions before the final SVD truncates
        the centre core back to ``kept`` — the global analogue of the two-site BUG
        ``aug_k_dims`` / ``aug_l_dims``.
    """
    L = mps.L
    c = L // 2 - 1
    mps.canonical(0, trunc=None)
    phi = mpo_times_mps(mpo, mps)

    W, aps_c, _ = k_sweep(mps, phi, c, maxdim, cutoff, aug_cutoff)
    Z, bps_c1, _ = l_sweep(mps, phi, c, maxdim, cutoff, aug_cutoff)

    # Two-site Galerkin window at the central bond: u0 = W_c, v0 = Z_{c+1}.
    u0 = W[c].permute([0, 2, 1])     # (link_l, site_l, mid_u)
    v0 = Z[c + 1]                    # (mid_v, link_r, site_r)
    # Seed S(t0) = <W, Z | psi> from the sweep carries (no M/N overlap matrices).
    s_start = contract(aps_c, bps_c1, axes=([1], [0]))   # (mid_u, mid_v)
    # Proposed augmented K (left/mid_u) and L (right/mid_v) central bonds before
    # the final SVD truncates the centre core back to `kept`.
    aug_k = int(u0.indices[2].dim)
    aug_l = int(v0.indices[0].dim)

    # MPO environments in the augmented basis (left from W, right from Z).
    e_left = to_complex(left_env_boundary(mps, mpo))
    for k in range(c):
        e_left = step_left_env(e_left, W[k], mpo[k])
    e_right = to_complex(right_env_boundary(mps, mpo))
    for s in range(L - 1, c + 1, -1):
        e_right = step_right_env(e_right, Z[s], mpo[s])

    def apply_s(s: Tensor) -> Tensor:
        theta = contract(contract(u0, s, axes=([2], [0])), v0, axes=([2], [0]))
        h_theta = _two_site_apply(theta, mpo[c], mpo[c + 1], e_left, e_right)
        s_l = contract(conj(u0), h_theta, axes=([0, 1], [0, 1]))
        return contract(s_l, conj(v0), axes=([1, 2], [1, 2]))

    # Central Galerkin core: the effective Hamiltonian is Hermitian, so 'krylov'
    # uses tensor Lanczos. In imaginary time the flow is a contraction, so the
    # substepped midpoint/rk4/trapezoid integrators are valid alternatives.
    s_new = local_expv(apply_s, tau, s_start, solver=solver, substeps=solver_substeps,
                       hermitian=True, krylov_maxiter=lanczos_maxiter, krylov_tol=lanczos_tol)
    upd = _truncate_and_assemble(u0, v0, s_new, mps._bond_itag(c + 1),
                                 maxdim=maxdim, cutoff=cutoff, n_new_left=0, n_new_right=0)
    disc_weight = _discarded_weight(s_new, mps._bond_itag(c + 1), upd.kept)

    cores = [W[k] for k in range(c)] + [upd.left_core, upd.right_core] \
        + [Z[k] for k in range(c + 2, L)]
    # Re-gauge from scratch: the per-matrix augmentation re-sorts bond charge sectors, so a
    # full canonical(0) (center cleared) is needed for a globally consistent gauge.
    #
    # maxdim is applied ONLY at the central S-step SVD (_truncate_and_assemble); this
    # re-gauge must NOT truncate (trunc=None). The K/L sweeps augment every bond to 2r as
    # scaffolding that carries the discarded-phi complement into the central Galerkin core.
    # Off-central those complement directions carry SMALL singular values (they are where
    # H.psi points, not where psi has weight yet) and have not been evolved by any local
    # Galerkin update, so a plain SVD truncation there would (i) drop exactly the admitted
    # complement, undoing the augmentation, and (ii) break the center<->frame consistency
    # the central core was built against -- measured to WRECK the imaginary-time cooling at
    # L=10 (E-E0 stuck ~1.5-2.2, non-monotonic) versus a clean monotonic convergence to
    # ~2e-3 with trunc=None. The off-central bonds therefore leave the step at (up to) 2r.
    mps._tensors = cores
    mps._center = None
    # TWO-WAY re-gauge (investigation 2026-07-08): a single right-canonical sweep
    # (canonical(0)) leaves the RIGHT frames -- already right-canonical from l_sweep and
    # never truncated -- at their full augmented rank, while only the LEFT frames get
    # compressed against the central bottleneck. That asymmetry let the right half blow up
    # to full Hilbert rank at L>=20. A lossless L->R then R->L pass yields the true minimal
    # Schmidt rank at every cut (symmetric), without discarding any weight.
    mps.canonical(L - 1, trunc=None)
    mps.canonical(0, trunc=None)
    return max(mps.bond_dims), disc_weight, aug_k, aug_l
