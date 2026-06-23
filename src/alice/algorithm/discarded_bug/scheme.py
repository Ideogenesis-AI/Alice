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


"""Odd/even parity sweeps driving the discarded-projector BUG local bond update.

Identical odd/even Trotter sweep to :mod:`alice.algorithm.two_site_bug.scheme`,
reusing its canonical two-site snapshot and layout transposes; the only change is
the local bond kernel — here the discarded-projector K/L/S candidate (see
:func:`alice.algorithm.discarded_bug.candidate.discarded_bug_local_bond_candidate`)
instead of the faithful CKL candidate.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from nicole import Tensor

from alice.network import MPS

# Reuse the faithful sweep's snapshot, layout transposes, and diagnostics verbatim.
from ..two_site_bug.scheme import _discarded_weight, bond_snapshot, parity_bonds
from .candidate import discarded_bug_local_bond_candidate


def discarded_bond(
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
) -> Tuple[int, float]:
    """Apply one discarded-projector BUG update to sites *(i, i+1)* of `mps`, in place.

    Moves the orthogonality center onto site *i* (truncation-free), snapshots the
    bond, runs the discarded-projector K/L/S local update for time `tau`, and
    writes the two updated cores back. After the call `mps.center == i + 1`.

    Returns the proposed augmented bond dimension (old rank + new K/L directions)
    and the relative weight discarded by this bond's S-step truncation.
    """
    mps.canonical(i, trunc=None)
    bond_data = bond_snapshot(mps, i)
    old_rank = int(bond_data['link_mid'].dim)

    candidate = discarded_bug_local_bond_candidate(
        bond_data,
        gate=gate,
        dt=tau,
        maxdim=maxdim,
        augment=augment,
        aug_krylov_depth=aug_krylov_depth,
        trunc_thresh=trunc_thresh,
        lanczos_tol=lanczos_tol,
        lanczos_maxiter=lanczos_maxiter,
    )

    from ..two_site_bug.scheme import _to_mps_layout
    mps[i] = _to_mps_layout(candidate['left_core'])
    mps[i + 1] = _to_mps_layout(candidate['right_core'])
    mps._center = i + 1

    augmented = old_rank + max(int(candidate['n_new_k']), int(candidate['n_new_l']))
    discarded = _discarded_weight(candidate['S_new'], int(candidate['keep']))
    return augmented, discarded


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
) -> Tuple[int, float]:
    """Apply every bond gate of one commuting group to `mps`, in place.

    Bonds of the chosen parity act on disjoint site pairs, so the group is an
    exact factor of the Trotter step. Bonds whose gate is `None` are skipped.
    Returns the largest proposed augmented bond dimension and the largest relative
    discarded weight over the bonds of this group.
    """
    augmented = 0
    discarded = 0.0
    for i in parity_bonds(mps.L, parity):
        if gates[i] is not None:
            aug, disc = discarded_bond(mps, i, gates[i], tau, maxdim, augment,
                                       aug_krylov_depth, trunc_thresh, lanczos_tol, lanczos_maxiter)
            augmented = max(augmented, aug)
            discarded = max(discarded, disc)
    return augmented, discarded
