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


"""Discarded-projector BUG local bond candidate (two-site BUG ``variant='discarded'``).

This is the *only* file that differs from the faithful Ceruti–Kusch–Lubich K/L/S
update in :mod:`alice.algorithm.two_site_bug._kernel.kls.candidate`. Everything
else — the Nicole tensor helpers, the Krylov ``expv`` substeps, the QR/SVD linear
algebra, the augmented-isometry construction, and the gate-application convention
— is reused unchanged from that kernel, so the odd/even Trotter sweep can swap
between the faithful and discarded local updates by selecting the candidate
function alone (see :data:`alice.algorithm.two_site_bug.scheme.parity_sweep`).

Discarded-projector BUG vs faithful BUG (state ``Θ0 = U0 · S0 · V0``)
---------------------------------------------------------------------
The faithful update grows the left frame by evolving ``K0 = U0·S0`` under the
right-projected generator ``H_K = V0† H V0`` and orthonormalising ``[U0 | K1]``
*through an overlap matrix* ``M̂`` that transports the core (``Ŝ0 = M̂ S0 N̂``).
The discarded variant changes exactly two things, and nothing else:

1. **Project-before.** The discarded (orthogonal-complement) projector is applied
   to the K/L *generator* before the exponential, not to the integrated factor.
   The K generator becomes ``G_K = P⊥_U0 · H_K`` with ``P⊥_U0 = I − U0 U0†`` and
   the L generator ``G_L = H_L · P⊥_V0`` with ``P⊥_V0 = I − V0† V0``. Because the
   projected generator is non-Hermitian, the K/L substep uses the general
   (``issymmetric=False``) Krylov path — a symmetry-preserving tensor Arnoldi
   exponential — rather than the Hermitian Lanczos.

2. **Act the augmented isometries, no overlap matrices.** The new directions are
   isolated by the discarded projector and stacked onto the old isometry to form
   ``Û = [U0 | Qk]`` / ``V̂ = [V0 ; Ql]`` — no ``M̂``/``N̂`` is formed. The S-step
   then projects the *current* two-site tensor directly onto the augmented bases,
   ``Ŝ0 = Û† Θ0 V̂†``, evolves it in the augmented basis (the Hermitian Galerkin
   generator), and truncates with an SVD.

The S-step generator, the augmented-basis Galerkin evolution, and the final SVD
truncation are identical to the faithful kernel. This is the Alice realisation of
the reference Julia ``discarded_bug_step!`` per-bond candidate.
"""

from __future__ import annotations

import math
from typing import Any

from nicole import Tensor, decomp

from ..indices import Ix, fresh_itag
from ..krylov import active_time_prefactor
from ..local_solvers import local_expv
from ..nicole_helpers import dag, tcontract
from .frame import (
    LocalBondFrame,
    _apply_gate_named,
    _clone_tensor_with_ixs,
    _singular_values_from_diag_tensor,
    _tensor_ix,
)
from .symmetric_completion import (
    _symmetric_augmented_left_isometry_from_k,
    _symmetric_augmented_right_isometry_from_l,
)


def _discarded_local_bond_candidate(
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
    solver: str = 'krylov',
    solver_substeps: int = 1,
    kl_cutoff: float | None = None,
):
    """Run one discarded-projector K/L/S local update (see module docstring).

    The K/L/S local exponentials are computed by the selected ``solver`` (see
    :mod:`alice.algorithm.two_site_bug._kernel.local_solvers`): ``'krylov'`` is the
    exact reference, ``'midpoint'``/``'rk4'`` are explicit RK with ``solver_substeps``
    internal steps, and ``'trapezoid'`` is the A-stable Crank–Nicolson rule. In
    imaginary time the evolution is non-unitary so any stable integrator is valid.
    """
    s_dt_eff = dt if s_dt is None else s_dt
    augment_left_here = augment and frame.old_rank < frame.left_capacity
    augment_right_here = augment and frame.old_rank < frame.right_capacity
    prefactor = active_time_prefactor()

    # ---- K-step: project-before, then integrate K0 = U0·S0 ----
    # H_K x = V0†-projected gate action; G_K x = P⊥_U0 (H_K x), P⊥_U0 = I − U0 U0†.
    # The projected generator is NON-Hermitian, so we use a symmetry-preserving
    # tensor Arnoldi exponential (never densifying to the standard basis, which
    # would break the U(1) block structure of the Nicole tensor).
    K0_tens = tcontract(frame.U0_tens, frame.S0_tens)        # (link_l, site_l, mid_k)
    mid_k = _tensor_ix(K0_tens, 2)

    def apply_gk(x_tens: Tensor) -> Tensor:
        theta = tcontract(x_tens, frame.V0_tens)
        evolved = _apply_gate_named(gate, theta, frame.site_l.itag, frame.site_r.itag)
        HK = tcontract(evolved, dag(frame.V0_tens))          # H_K x on (link_l, site_l, mid_k)
        # P⊥_U0 on (link_l, site_l): HK − U0 (U0† HK).
        return HK - tcontract(frame.U0_tens, tcontract(dag(frame.U0_tens), HK))

    K1_tens = local_expv(apply_gk, prefactor * dt, K0_tens,
                         solver=solver, substeps=solver_substeps, hermitian=False,
                         krylov_maxiter=lanczos_maxiter, krylov_tol=lanczos_tol)
    # Direct sum Û = [U0 | Qk], built per U(1) charge sector so the Nicole block
    # structure stays valid (a symmetry-blind dense QR would mix sectors and be
    # rejected). No overlap matrix M̂ is formed — the discarded variant projects
    # Θ0 onto the augmented bases directly in the S-step below.
    U_aug_tens, _M_hat, n_new_k = _symmetric_augmented_left_isometry_from_k(
        frame.U0_tens, K1_tens, frame.link_l, frame.site_l, frame.canon_u0, mid_k,
        augment=augment_left_here, max_rank=math.inf, aug_tol=aug_tol, kl_cutoff=kl_cutoff)

    # ---- L-step: project-before, then integrate L0 = S0·V0 ----
    L0_tens = tcontract(frame.S0_tens, frame.V0_tens)        # (mid_l, site_r, link_r)
    mid_l = _tensor_ix(L0_tens, 0)

    def apply_gl(x_tens: Tensor) -> Tensor:
        theta = tcontract(frame.U0_tens, x_tens)
        evolved = _apply_gate_named(gate, theta, frame.site_l.itag, frame.site_r.itag)
        HL = tcontract(dag(frame.U0_tens), evolved)          # H_L x on (mid_l, site_r, link_r)
        # P⊥_V0 on (site_r, link_r): HL − (HL V0†) V0.
        return HL - tcontract(tcontract(HL, dag(frame.V0_tens)), frame.V0_tens)

    L1_tens = local_expv(apply_gl, prefactor * dt, L0_tens,
                         solver=solver, substeps=solver_substeps, hermitian=False,
                         krylov_maxiter=lanczos_maxiter, krylov_tol=lanczos_tol)
    V_aug_tens, _N_hat, n_new_l = _symmetric_augmented_right_isometry_from_l(
        frame.V0_tens, L1_tens, frame.canon_v0, mid_l, frame.site_r, frame.link_r,
        augment=augment_right_here, max_rank=math.inf, aug_tol=aug_tol, kl_cutoff=kl_cutoff)

    # ---- S-step: project Θ0 directly onto the augmented bases (no M̂/N̂), evolve ----
    # Ŝ0 = Û† Θ0 V̂† as a tensor contraction. dag(U_aug) exposes the augmented left
    # mid-leg, dag(V_aug) the augmented right mid-leg, so Ŝ0 is automatically tagged
    # to contract back with U_aug_tens / V_aug_tens in apply_s_tensor below.
    theta0_tens = tcontract(tcontract(frame.U0_tens, frame.S0_tens), frame.V0_tens)
    S_start_tens = tcontract(tcontract(dag(U_aug_tens), theta0_tens), dag(V_aug_tens))

    def apply_s_tensor(x_tens: Tensor) -> Tensor:
        theta = tcontract(tcontract(U_aug_tens, x_tens), V_aug_tens)
        evolved = _apply_gate_named(gate, theta, frame.site_l.itag, frame.site_r.itag)
        projected = tcontract(dag(U_aug_tens), evolved)
        return tcontract(projected, dag(V_aug_tens))

    # S-step generator is Hermitian (the faithful Galerkin generator on the
    # augmented bases); imaginary time makes the flow a contraction either way.
    S_new_tens = local_expv(apply_s_tensor, prefactor * s_dt_eff, S_start_tens,
                            solver=solver, substeps=solver_substeps, hermitian=True,
                            krylov_maxiter=lanczos_maxiter, krylov_tol=lanczos_tol)

    # ---- truncate: SVD sets the new (rank-adaptive) bond dimension ----
    # Done in the symmetry-blocked Nicole representation (mirrors the faithful
    # kernel's S-step split), so the kept rank respects the U(1) sectors.
    final_left_tag = fresh_itag(frame.link_mid.itag)
    final_right_tag = fresh_itag(frame.link_mid.itag)
    U_s, Sdiag, Vh = decomp(
        S_new_tens, 0, mode="SVD",
        itag=(final_left_tag, final_right_tag),
        trunc={
            "nkeep": int(maxdim),
            "thresh": max(float(aug_tol if trunc_thresh is None else trunc_thresh), 1e-14),
        },
    )
    left_tmp = tcontract(U_aug_tens, U_s)
    right_tmp = tcontract(tcontract(Sdiag, Vh, axes=([1], [0])), V_aug_tens)
    left_tmp.retag({final_left_tag: frame.link_mid.itag})
    right_tmp.retag({final_left_tag: frame.link_mid.itag})

    new_bond = Ix(frame.link_mid.itag, int(left_tmp.indices[2].dim), left_tmp.indices[2].direction,
                  left_tmp.indices[2].sectors, left_tmp.indices[2].group)
    right_bond = Ix(frame.link_mid.itag, int(right_tmp.indices[0].dim), right_tmp.indices[0].direction,
                    right_tmp.indices[0].sectors, right_tmp.indices[0].group)
    left_core = _clone_tensor_with_ixs(left_tmp, [frame.link_l, frame.site_l, new_bond])
    right_core = _clone_tensor_with_ixs(right_tmp, [right_bond, frame.site_r, frame.link_r])
    svals = _singular_values_from_diag_tensor(Sdiag)

    return {
        "left_core": left_core,
        "right_core": right_core,
        "U_aug_tens": U_aug_tens,
        "V_aug_tens": V_aug_tens,
        "S_new": S_new_tens,
        "n_new_k": int(n_new_k),
        "n_new_l": int(n_new_l),
        "keep": int(left_core.indices[2].dim),
        "svals": svals,
    }


def _discarded_kls_local_bond_candidate(
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
    solver: str = 'krylov',
    solver_substeps: int = 1,
    kl_cutoff: float | None = None,
    **kwargs: Any,
):
    """Return the discarded-projector BUG candidate on one bond.

    Mirrors the call surface of
    :func:`alice.algorithm.two_site_bug._kernel.kls.candidate._faithful_kls_local_bond_candidate`
    so the odd/even sweep can swap kernels without any other change. ``solver`` and
    ``solver_substeps`` select the local (imaginary-time) integrator for the K/L/S
    substeps (see :mod:`alice.algorithm.two_site_bug._kernel.local_solvers`).
    """
    if aug_krylov_depth != 1:
        raise ValueError("discarded variant currently supports aug_krylov_depth == 1 only.")
    kwargs.pop("substep_method", None)
    kwargs.pop("matrixfree_sstep", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown discarded variant option(s): {unknown}")

    frame = LocalBondFrame.from_mapping(bond_data)
    return _discarded_local_bond_candidate(
        frame, gate, dt,
        maxdim=maxdim, s_dt=s_dt, augment=augment, aug_krylov_depth=aug_krylov_depth,
        aug_tol=aug_tol, trunc_thresh=trunc_thresh,
        lanczos_tol=lanczos_tol, lanczos_maxiter=lanczos_maxiter,
        solver=solver, solver_substeps=solver_substeps, kl_cutoff=kl_cutoff,
    )
