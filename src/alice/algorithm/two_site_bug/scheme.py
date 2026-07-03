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


"""Odd/even parity sweeps driving the faithful-KLS local bond update.

The chain Hamiltonian splits into two commuting groups — bonds with an even
left-site index (0, 2, 4, …) and bonds with an odd left-site index (1, 3, 5, …).
Gates within one group act on disjoint site pairs, so a parity sweep applies
them as an exact factor of the Trotter step; the splitting error lives only
between the two groups. This is the odd/even BUG sweep used for the domain-wall
XX chain.

Each bond update is the Ceruti–Kusch–Lubich K/L/S step from
:mod:`alice.algorithm.two_site_bug._kernel` (faithful Basis-Update & Galerkin).
This module is the thin Alice adapter: it brings the orthogonality center onto
the active bond, takes a canonical two-site snapshot of the Alice `MPS`, calls
the vendored kernel, and writes the updated cores back. The kernel works in the
`(link_l, site, link_r)` tensor layout; Alice stores `(left, right, phys)`, so
the snapshot and writeback transpose between the two.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from nicole import Tensor, permute

from alice.network import MPS

from ._kernel import (
    Ix,
    _discarded_kls_local_bond_candidate,
    _faithful_kls_local_bond_candidate,
    lq,
    qr,
    tcontract,
    to_dense,
)

# Local-bond candidate kernels selectable by ``two_site_bug.Options.variant``.
# ``'faithful'`` is the Ceruti–Kusch–Lubich K/L/S update (overlap matrices M̂/N̂);
# ``'discarded'`` is the project-before discarded-projector update that acts the
# augmented isometries directly (no overlap matrices) — see
# :mod:`._kernel.kls.discarded_candidate`.
_CANDIDATE_KERNELS: Dict[str, Callable] = {
    'faithful': _faithful_kls_local_bond_candidate,
    'discarded': _discarded_kls_local_bond_candidate,
}


def resolve_candidate(variant: str) -> Callable:
    """Return the local-bond candidate function for a ``variant`` name."""
    try:
        return _CANDIDATE_KERNELS[variant]
    except KeyError:
        known = ', '.join(sorted(_CANDIDATE_KERNELS))
        raise ValueError(f"unknown two-site BUG variant {variant!r}; recognised values are: {known}")


def _discarded_weight(s_new: Tensor, keep: int) -> float:
    """Relative Frobenius weight discarded when the S-step core is cut to `keep`.

    The kernel evolves the (small) two-site core `s_new` and then keeps `keep`
    singular values. This recomputes the full singular spectrum of `s_new` and
    returns `sqrt(Σ_{i≥keep} σ_i² / Σ_i σ_i²)` — the fraction of the bond's weight
    the truncation throws away, the standard MPS discarded-weight diagnostic.
    """
    matrix = to_dense(s_new, list(s_new.itags))
    svals = torch.linalg.svdvals(matrix.reshape(matrix.shape[0], -1))
    total = float((svals ** 2).sum())
    if total == 0.0 or keep >= svals.numel():
        return 0.0
    tail = float((svals[keep:] ** 2).sum())
    return (tail / total) ** 0.5


def _ix(tensor: Tensor, axis: int) -> Ix:
    """Wrap one leg of a Nicole tensor as a kernel `Ix` handle."""
    index = tensor.indices[axis]
    return Ix(tensor.itags[axis], int(index.dim), index.direction, index.sectors, index.group)


def _to_kernel_layout(site: Tensor) -> Tensor:
    """Transpose an Alice MPS tensor `(left, right, phys)` to `(left, phys, right)`."""
    return permute(site, [0, 2, 1])


def _to_mps_layout(core: Tensor) -> Tensor:
    """Transpose a kernel core `(left, phys, right)` back to Alice `(left, right, phys)`."""
    return permute(core, [0, 2, 1])


def bond_snapshot(mps: MPS, i: int) -> Dict[str, object]:
    """Take the canonical two-site snapshot the KLS kernel consumes at bond *(i, i+1)*.

    `mps` must already have its orthogonality center on site *i*. A QR of site
    *i* gives the left isometry `U0` and an LQ of site *i+1* gives the right
    isometry `V0`; their inner factors contract to the bond center `S0`. These
    are exact, truncation-free moves, so the subsequent KLS update — and only it
    — is responsible for the rank adaptation.

    Parameters
    ----------
    mps:
        State with `center == i`.
    i:
        Left site of the bond.

    Returns
    -------
    dict
        Snapshot mapping consumed by `LocalBondFrame.from_mapping`: the five leg
        handles (`link_l`, `site_l`, `link_mid`, `site_r`, `link_r`), the
        canonical factors (`U0_tens`, `V0_tens`, `S0_tens`), and the middle bond
        handles on each side (`canon_u0`, `canon_v0`).
    """
    left = _to_kernel_layout(mps[i])       # (link_l, site_l, link_mid)
    right = _to_kernel_layout(mps[i + 1])  # (link_mid, site_r, link_r)

    link_l = _ix(left, 0)
    site_l = _ix(left, 1)
    link_mid = _ix(left, 2)
    site_r = _ix(right, 1)
    link_r = _ix(right, 2)

    U0_tens, s_left, canon_u0 = qr(left, [link_l, site_l], tag=link_mid.itag)
    s_right, V0_tens, canon_v0 = lq(right, [site_r, link_r], tag=link_mid.itag)
    S0_tens = tcontract(s_left, s_right)

    return {
        'link_l': link_l,
        'site_l': site_l,
        'link_mid': link_mid,
        'site_r': site_r,
        'link_r': link_r,
        'U0_tens': U0_tens,
        'V0_tens': V0_tens,
        'S0_tens': S0_tens,
        'canon_u0': canon_u0,
        'canon_v0': canon_v0,
    }


def kls_bond(
    mps: MPS,
    i: int,
    gate: Tensor,
    tau: float,
    maxdim: int,
    augment: bool,
    aug_krylov_depth: int,
    trunc_thresh: float,
    lanczos_tol: float,
    lanczos_maxiter: int,
    candidate_fn: Callable = _faithful_kls_local_bond_candidate,
    solver: str = 'krylov',
    solver_substeps: int = 1,
    kl_cutoff: float | None = None,
    kl_cutoff_min_bond: int = 4,
) -> Tuple[int, int, float]:
    """Apply one faithful-KLS update to sites *(i, i+1)* of `mps`, in place.

    Moves the orthogonality center onto site *i* (truncation-free), snapshots the
    bond, runs the vendored K/L/S local update for time `tau` (the active
    evolution prefactor — `-1j` for real time, `-1` for imaginary — is applied by
    the kernel), and writes the two updated cores back. After the call
    `mps.center == i + 1`.

    Parameters
    ----------
    mps:
        State to update in place.
    i:
        Left site of the bond.
    gate:
        Bare two-site bond Hamiltonian in the kernel convention (see
        :func:`alice.algorithm.two_site_bug.bond.kernel_gate`).
    tau:
        Real time advanced by this local step.
    maxdim:
        Bond-dimension cap kept by the post-S-step SVD truncation.
    augment, aug_krylov_depth, lanczos_tol, lanczos_maxiter:
        KLS controls forwarded to the kernel.

    trunc_thresh:
        Singular-value threshold for the post-S-step SVD: the bond keeps only the
        directions whose weight exceeds it, so the rank grows only as far as the
        entanglement of the state requires (the rank-adaptive truncation).

    Returns
    -------
    int
        Proposed augmented **K** bond dimension (old rank + new K directions) at
        this bond, before the truncated split.
    int
        Proposed augmented **L** bond dimension (old rank + new L directions).
    float
        Relative weight discarded by this bond's S-step truncation.
    """
    mps.canonical(i, trunc=None)
    bond_data = bond_snapshot(mps, i)
    old_rank = int(bond_data['link_mid'].dim)

    # Adaptive-delay gate: only weight-trim the K/L augmentation once this bond has
    # grown past `kl_cutoff_min_bond`. Below it, fall back to full d·r completion so a
    # low-rank (product) state can grow its entanglement instead of collapsing.
    effective_kl = kl_cutoff
    if kl_cutoff is not None and old_rank < kl_cutoff_min_bond:
        effective_kl = None

    candidate = candidate_fn(
        bond_data,
        gate=gate,
        dt=tau,
        maxdim=maxdim,
        augment=augment,
        aug_krylov_depth=aug_krylov_depth,
        trunc_thresh=trunc_thresh,
        lanczos_tol=lanczos_tol,
        lanczos_maxiter=lanczos_maxiter,
        solver=solver,
        solver_substeps=solver_substeps,
        kl_cutoff=effective_kl,
    )

    mps[i] = _to_mps_layout(candidate['left_core'])
    mps[i + 1] = _to_mps_layout(candidate['right_core'])
    mps._center = i + 1

    aug_k = old_rank + int(candidate['n_new_k'])
    aug_l = old_rank + int(candidate['n_new_l'])
    discarded = _discarded_weight(candidate['S_new'], int(candidate['keep']))
    return aug_k, aug_l, discarded


def parity_bonds(length: int, parity: str) -> List[int]:
    """Return the left-site indices of all bonds in one commuting group.

    Parameters
    ----------
    length:
        Number of sites `L`.
    parity:
        `'even'` for bonds with an even left-site index (0, 2, 4, …) or `'odd'`
        for bonds with an odd left-site index (1, 3, 5, …).

    Returns
    -------
    list of int
        Left-site indices of the bonds in the requested group.

    Raises
    ------
    ValueError
        If `parity` is not `'even'` or `'odd'`.
    """
    if parity == 'even':
        return list(range(0, length - 1, 2))
    if parity == 'odd':
        return list(range(1, length - 1, 2))
    raise ValueError(f"parity must be 'even' or 'odd', got {parity!r}")


def parity_sweep(
    mps: MPS,
    gates: List[Optional[Tensor]],
    parity: str,
    tau: float,
    maxdim: int,
    augment: bool,
    aug_krylov_depth: int,
    trunc_thresh: float,
    lanczos_tol: float,
    lanczos_maxiter: int,
    candidate_fn: Callable = _faithful_kls_local_bond_candidate,
    solver: str = 'krylov',
    solver_substeps: int = 1,
    kl_cutoff: float | None = None,
    kl_cutoff_min_bond: int = 4,
) -> Tuple[int, int, float]:
    """Apply every bond gate of one commuting group to `mps`, in place.

    Bonds of the chosen parity act on disjoint site pairs, so the group is an
    exact factor of the Trotter step. Bonds whose gate is `None` (no Hamiltonian
    term) are skipped.

    Parameters
    ----------
    mps:
        State to update in place.
    gates:
        Per-bond kernel gates of length `L - 1`; entry *b* acts on bond *(b, b+1)*.
    parity:
        `'even'` or `'odd'` — selects the commuting bond group.
    tau:
        Real time advanced by each local KLS step in this group.
    maxdim, augment, aug_krylov_depth, trunc_thresh, lanczos_tol, lanczos_maxiter:
        KLS controls forwarded to each bond update.

    Returns
    -------
    int
        Largest proposed augmented bond dimension over the bonds of this group
        (0 if the group has no active bonds).
    float
        Largest relative discarded weight over the bonds of this group.
    """
    aug_k = aug_l = 0
    discarded = 0.0
    for i in parity_bonds(mps.L, parity):
        if gates[i] is not None:
            ak, al, disc = kls_bond(mps, i, gates[i], tau, maxdim, augment,
                                    aug_krylov_depth, trunc_thresh, lanczos_tol, lanczos_maxiter,
                                    candidate_fn, solver, solver_substeps, kl_cutoff,
                                    kl_cutoff_min_bond)
            aug_k = max(aug_k, ak)
            aug_l = max(aug_l, al)
            discarded = max(discarded, disc)
    return aug_k, aug_l, discarded
