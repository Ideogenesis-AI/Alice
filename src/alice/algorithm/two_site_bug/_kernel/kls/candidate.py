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

"""Main entry points for KLS local bond candidates."""

from __future__ import annotations

import math
from typing import Any

import torch
from nicole import Tensor, decomp

from ..indices import Ix, fresh_itag
from ..nicole_helpers import dag, tcontract
from .augment import (
    _collect_tensor_krylov_directions,
    _stack_left_krylov_directions,
    _stack_right_krylov_directions,
    _tensor_expv,
)
from .symmetric_completion import (
    _symmetric_augmented_left_isometry_from_k,
    _symmetric_augmented_right_isometry_from_l,
)
from .frame import LocalBondFrame, _apply_gate_named, _clone_tensor_with_ixs, _singular_values_from_diag_tensor, _tensor_ix


def _symmetric_local_bond_candidate(
    frame: LocalBondFrame,
    gate: Tensor,
    dt: complex,
    maxdim: int = 200,
    s_dt: complex | None = None,
    augment: bool = True,
    aug_krylov_depth: int = 1,
    aug_tol: float = 1e-12,
    trunc_thresh: float | None = None,
    lanczos_tol: float = 1e-15,
    lanczos_maxiter: int = 30,
):
    """Run one fully symmetric K/L/S local update.

    Args:
        frame: Canonical two-site data on the active bond.
        gate: Local two-site Hamiltonian term.
        dt: Shared K/L timestep.
        maxdim: Maximum bond dimension kept after the post-S-step SVD.
        s_dt: Optional S-step timestep. When omitted, ``dt`` is reused.
        augment: Whether the local basis may grow before the S-step.
        aug_krylov_depth: Number of K/L Krylov directions stacked before basis extraction.
        aug_tol: Numerical threshold used when removing redundant directions.
        lanczos_tol: Lanczos termination tolerance for both tensor and dense ``expv`` solves.
        lanczos_maxiter: Maximum Lanczos iterations per local substep.

    Returns:
        A candidate dictionary containing the updated left/right cores together
        with augmentation diagnostics.
    """
    s_dt_eff = dt if s_dt is None else s_dt
    augment_left_here = augment and frame.old_rank < frame.left_capacity
    augment_right_here = augment and frame.old_rank < frame.right_capacity

    # K-step: evolve the left frame with the right frame frozen.
    def apply_k_tensor(x_tens: Tensor) -> Tensor:
        theta = tcontract(x_tens, frame.V0_tens)
        evolved = _apply_gate_named(gate, theta, frame.site_l.itag, frame.site_r.itag)
        return tcontract(evolved, dag(frame.V0_tens))

    K0_tens = tcontract(frame.U0_tens, frame.S0_tens)
    mid_k = _tensor_ix(K0_tens, 2)
    k_dirs = _collect_tensor_krylov_directions(
        K0_tens,
        apply_k_tensor,
        dt,
        aug_krylov_depth=aug_krylov_depth,
        lanczos_maxiter=lanczos_maxiter,
        lanczos_tol=lanczos_tol,
    )
    K1_tens, mid_k_ext = _stack_left_krylov_directions(k_dirs, frame.link_l, frame.site_l, mid_k)
    U_aug_tens, M_hat_tens, n_new_k = _symmetric_augmented_left_isometry_from_k(
        frame.U0_tens,
        K1_tens,
        frame.link_l,
        frame.site_l,
        frame.canon_u0,
        mid_k_ext,
        augment=augment_left_here,
        max_rank=math.inf,
        aug_tol=aug_tol,
    )

    # L-step: mirror the same logic with the left frame frozen.
    def apply_l_tensor(x_tens: Tensor) -> Tensor:
        theta = tcontract(frame.U0_tens, x_tens)
        evolved = _apply_gate_named(gate, theta, frame.site_l.itag, frame.site_r.itag)
        return tcontract(dag(frame.U0_tens), evolved)

    L0_tens = tcontract(frame.S0_tens, frame.V0_tens)
    mid_l = _tensor_ix(L0_tens, 0)
    l_dirs = _collect_tensor_krylov_directions(
        L0_tens,
        apply_l_tensor,
        dt,
        aug_krylov_depth=aug_krylov_depth,
        lanczos_maxiter=lanczos_maxiter,
        lanczos_tol=lanczos_tol,
    )
    L1_tens, mid_l_ext = _stack_right_krylov_directions(l_dirs, mid_l, frame.site_r, frame.link_r)
    V_aug_tens, N_hat_tens, n_new_l = _symmetric_augmented_right_isometry_from_l(
        frame.V0_tens,
        L1_tens,
        frame.canon_v0,
        mid_l_ext,
        frame.site_r,
        frame.link_r,
        augment=augment_right_here,
        max_rank=math.inf,
        aug_tol=aug_tol,
    )

    # S-step: evolve inside the augmented left/right bases.
    S_start_tens = tcontract(tcontract(M_hat_tens, frame.S0_tens), N_hat_tens)
    numops_s = [0]

    def apply_s_tensor(x_tens: Tensor) -> Tensor:
        numops_s[0] += 1
        theta = tcontract(tcontract(U_aug_tens, x_tens), V_aug_tens)
        evolved = _apply_gate_named(gate, theta, frame.site_l.itag, frame.site_r.itag)
        projected = tcontract(dag(U_aug_tens), evolved)
        return tcontract(projected, dag(V_aug_tens))

    S_new_tens = _tensor_expv(
        apply_s_tensor,
        s_dt_eff,
        S_start_tens,
        lanczos_maxiter=lanczos_maxiter,
        lanczos_tol=lanczos_tol,
    )

    # Final truncation writes the augmented S-step result back to a standard MPS pair.
    final_left_tag = fresh_itag(frame.link_mid.itag)
    final_right_tag = fresh_itag(frame.link_mid.itag)
    U_s, Sdiag, Vh = decomp(
        S_new_tens,
        0,
        mode="SVD",
        itag=(final_left_tag, final_right_tag),
        trunc={
            "nkeep": int(maxdim),
            # The SVD threshold controls the rank adaptation; it is decoupled from
            # `aug_tol` (which only filters near-dependent K/L directions) so the
            # caller can tune how aggressively the bond grows. Falls back to
            # `aug_tol` when not supplied (kernel default behaviour).
            "thresh": max(float(aug_tol if trunc_thresh is None else trunc_thresh), 1e-14),
        },
    )
    left_tmp = tcontract(U_aug_tens, U_s)
    right_tmp = tcontract(tcontract(Sdiag, Vh, axes=([1], [0])), V_aug_tens)
    left_tmp.retag({final_left_tag: frame.link_mid.itag})
    right_tmp.retag({final_left_tag: frame.link_mid.itag})

    new_bond = Ix(frame.link_mid.itag, int(left_tmp.indices[2].dim), left_tmp.indices[2].direction, left_tmp.indices[2].sectors, left_tmp.indices[2].group)
    right_bond = Ix(frame.link_mid.itag, int(right_tmp.indices[0].dim), right_tmp.indices[0].direction, right_tmp.indices[0].sectors, right_tmp.indices[0].group)
    left_core = _clone_tensor_with_ixs(left_tmp, [frame.link_l, frame.site_l, new_bond])
    right_core = _clone_tensor_with_ixs(right_tmp, [right_bond, frame.site_r, frame.link_r])

    return {
        "left_core": left_core,
        "right_core": right_core,
        "U_aug_tens": U_aug_tens,
        "V_aug_tens": V_aug_tens,
        "S_new": S_new_tens,
        "n_new_k": n_new_k,
        "n_new_l": n_new_l,
        "keep": int(left_core.indices[2].dim),
        "svals": _singular_values_from_diag_tensor(Sdiag),
        "numops_s": numops_s[0],
    }


def _faithful_kls_local_bond_candidate(
    bond_data: dict[str, Any],
    *,
    gate,
    dt: complex,
    maxdim: int = 200,
    s_dt: complex | None = None,
    augment: bool = True,
    aug_krylov_depth: int = 1,
    aug_tol: float = 1e-12,
    trunc_thresh: float | None = None,
    lanczos_tol: float = 1e-15,
    lanczos_maxiter: int = 30,
    **kwargs: Any,
):
    """Return the forward local BUG/KLS candidate on one bond.

    Args:
        bond_data: Canonical two-site snapshot dictionary.
        gate: Local two-site Hamiltonian term.
        dt: Shared K/L timestep.
        maxdim: Maximum bond dimension kept after the post-S-step SVD.
        s_dt: Optional S-step timestep. When omitted, ``dt`` is reused.
        augment: Whether the local basis may grow before the S-step.
        aug_krylov_depth: Number of K/L Krylov directions stacked before basis extraction.
        aug_tol: Numerical threshold used when removing redundant directions.
        lanczos_tol: Lanczos termination tolerance for both tensor and dense ``expv`` solves.
        lanczos_maxiter: Maximum Lanczos iterations per local substep.
        **kwargs: Legacy keyword arguments (substep_method, matrixfree_sstep are ignored).

    Returns:
        A dictionary containing the updated left/right cores, the augmented
        bases, the evolved S-step tensor, and diagnostic counts used by the
        tests and the bug sweep.
    """
    if aug_krylov_depth < 1:
        raise ValueError(f"aug_krylov_depth must be >= 1; got {aug_krylov_depth}")

    # Accept and ignore legacy bug compatibility keywords
    kwargs.pop("substep_method", None)
    kwargs.pop("matrixfree_sstep", None)
    # The faithful (unitary) update always uses the exact Krylov exponential; the
    # pluggable local solver applies only to the non-unitary discarded variant, so
    # accept and ignore the solver controls when the shared sweep forwards them.
    kwargs.pop("solver", None)
    kwargs.pop("solver_substeps", None)
    kwargs.pop("kl_cutoff", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown KLS option(s): {unknown}")

    frame = LocalBondFrame.from_mapping(bond_data)
    return _symmetric_local_bond_candidate(
        frame,
        gate,
        dt,
        maxdim=maxdim,
        s_dt=s_dt,
        augment=augment,
        aug_krylov_depth=aug_krylov_depth,
        aug_tol=aug_tol,
        trunc_thresh=trunc_thresh,
        lanczos_tol=lanczos_tol,
        lanczos_maxiter=lanczos_maxiter,
    )


def _faithful_reverse_kls_local_bond_candidate(bond_data: dict[str, Any], **kwargs):
    """Reverse-sweep alias of `_faithful_kls_local_bond_candidate`."""
    return _faithful_kls_local_bond_candidate(bond_data, **kwargs)
