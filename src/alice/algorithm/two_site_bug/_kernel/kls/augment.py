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

"""Augmentation and basis building logic for KLS updates."""

from __future__ import annotations

import math
from typing import Any

import torch
from nicole import Sector, Tensor, decomp, einsum

from ..indices import Ix, fresh_itag, resolved_sectors
from ..krylov import active_time_prefactor, linear_substep, tensor_lanczos_expv
from ..linalg import (
    complete_column_basis,
    complete_row_basis,
    identity_overlap_matrix,
    qr_column_basis,
    qr_row_basis,
)
from ..nicole_helpers import dag, flatten_fortran, make_tensor, reshape_fortran, to_dense, tcontract
from .frame import (
    LocalBondFrame,
    _apply_gate_named,
    _clone_tensor_with_ixs,
    _dense_from_tensor_with_ixs,
    _left_row_indices_by_flux,
    _right_col_indices_by_flux,
    _sector_offsets,
    _tensor_ix,
)


def _tensor_expv(
    apply,
    dt: complex,
    tensor: Tensor,
    lanczos_maxiter: int = 30,
    lanczos_tol: float = 1e-15,
) -> Tensor:
    """Apply a Lanczos ``expv`` step to a Nicole tensor.

    Args:
        apply: Matrix-free tensor action representing the local Hamiltonian.
        dt: Local timestep for this substep.
        tensor: Input state tensor.
        lanczos_maxiter: Maximum Lanczos iterations per local substep.
        lanczos_tol: Lanczos termination tolerance.

    Returns:
        The evolved tensor after ``exp(prefactor * dt * H)``.
    """
    return tensor_lanczos_expv(
        apply,
        active_time_prefactor() * dt,
        tensor,
        maxiter=lanczos_maxiter,
        tol=lanczos_tol,
    )


def _collect_tensor_krylov_directions(
    seed: Tensor,
    apply,
    dt: complex,
    aug_krylov_depth: int = 1,
    lanczos_maxiter: int = 30,
    lanczos_tol: float = 1e-15,
) -> list[Tensor]:
    """Collect the K/L Krylov directions used for local basis growth.

    Args:
        seed: Input tensor for the K or L substep.
        apply: Matrix-free tensor action for the corresponding projected local
            Hamiltonian.
        dt: Local timestep used for the first Krylov direction.
        aug_krylov_depth: Number of K/L Krylov directions stacked before basis extraction.
        lanczos_maxiter: Maximum Lanczos iterations per local substep.
        lanczos_tol: Lanczos termination tolerance.

    Returns:
        A list containing the evolved first direction followed by repeated
        projected-Hamiltonian applications when ``aug_krylov_depth > 1``.
    """
    first_direction = _tensor_expv(apply, dt, seed, lanczos_maxiter=lanczos_maxiter, lanczos_tol=lanczos_tol)
    directions = [first_direction]
    next_direction = first_direction
    for _ in range(2, aug_krylov_depth + 1):
        next_direction = apply(next_direction)
        directions.append(next_direction)
    return directions


def _filter_left_aug_columns(U0_mat: torch.Tensor, K1_mat: torch.Tensor, aug_tol: float) -> torch.Tensor:
    """Keep only K-update columns that add directions beyond span(U0)."""
    if K1_mat.numel() == 0 or K1_mat.shape[1] == 0:
        return K1_mat[:, :0]
    proj = U0_mat @ (U0_mat.conj().transpose(0, 1) @ K1_mat)
    resid = K1_mat - proj
    keep = torch.linalg.norm(resid, dim=0) > aug_tol
    return resid[:, keep]


def _filter_right_aug_rows(V0_mat: torch.Tensor, L1_mat: torch.Tensor, aug_tol: float) -> torch.Tensor:
    """Keep only L-update rows that add directions beyond span(V0)."""
    if L1_mat.numel() == 0 or L1_mat.shape[0] == 0:
        return L1_mat[:0, :]
    proj = (L1_mat @ V0_mat.conj().transpose(0, 1)) @ V0_mat
    resid = L1_mat - proj
    keep = torch.linalg.norm(resid, dim=1) > aug_tol
    return resid[keep, :]


def _pick_left_update(
    U0_mat: torch.Tensor,
    K1_mat: torch.Tensor,
    augment: bool = True,
    max_rank: int | float = math.inf,
    aug_tol: float = 1e-12,
):
    """Choose an augmented left basis and its overlap with ``U0_mat``.

    Args:
        U0_mat: Current left isometry as a dense matrix.
        K1_mat: Candidate K-step directions as dense columns.
        augment: Whether new Krylov directions may enlarge the basis.
        max_rank: Hard cap on the returned basis rank.
        aug_tol: Threshold used to discard nearly dependent directions.

    Returns:
        ``(basis, overlap, n_new)`` for the chosen left basis.
    """

    if not augment or K1_mat.numel() == 0 or K1_mat.shape[1] == 0:
        overlap = identity_overlap_matrix(U0_mat.dtype, U0_mat.shape[1], device=U0_mat.device)
        return U0_mat, overlap, 0

    Kf = _filter_left_aug_columns(U0_mat, K1_mat, aug_tol)
    Qk, _ = qr_column_basis(Kf)
    cand = torch.cat([U0_mat, Qk], dim=1) if Qk.numel() else U0_mat
    Q, _ = qr_column_basis(cand)
    if max_rank is not math.inf:
        Q = Q[:, : min(Q.shape[1], int(max_rank))]
    overlap = Q.conj().transpose(0, 1) @ U0_mat
    n_new = max(0, Q.shape[1] - U0_mat.shape[1])
    return Q, overlap, n_new


def _pick_right_update(
    V0_mat: torch.Tensor,
    L1_mat: torch.Tensor,
    augment: bool = True,
    max_rank: int | float = math.inf,
    aug_tol: float = 1e-12,
):
    """Choose an augmented right basis and its overlap with ``V0_mat``.

    Args:
        V0_mat: Current right isometry as a dense matrix.
        L1_mat: Candidate L-step directions as dense rows.
        augment: Whether new Krylov directions may enlarge the basis.
        max_rank: Hard cap on the returned basis rank.
        aug_tol: Threshold used to discard nearly dependent directions.

    Returns:
        ``(basis, overlap, n_new)`` for the chosen right basis.
    """

    if not augment or L1_mat.numel() == 0 or L1_mat.shape[0] == 0:
        overlap = identity_overlap_matrix(V0_mat.dtype, V0_mat.shape[0], device=V0_mat.device)
        return V0_mat, overlap, 0

    Lf = _filter_right_aug_rows(V0_mat, L1_mat, aug_tol)
    Ql, _ = qr_row_basis(Lf)
    cand = torch.cat([V0_mat, Ql], dim=0) if Ql.numel() else V0_mat
    Q, _ = qr_row_basis(cand)
    if max_rank is not math.inf:
        Q = Q[: min(Q.shape[0], int(max_rank)), :]
    overlap = V0_mat @ Q.conj().transpose(0, 1)
    n_new = max(0, Q.shape[0] - V0_mat.shape[0])
    return Q, overlap, n_new


def _left_tensor_matrix(U_tens, link_l: Ix, site_l: Ix, mid: Ix):
    """Reshape a left tensor `(link_l, site_l, mid)` into matrix form."""
    block = _dense_from_tensor_with_ixs(U_tens, [link_l, site_l, mid]).to(torch.complex128)
    return reshape_fortran(block, (link_l.dim * site_l.dim, mid.dim))


def _right_tensor_matrix(V_tens, mid: Ix, site_r: Ix, link_r: Ix):
    """Reshape a right tensor `(mid, site_r, link_r)` into matrix form."""
    block = _dense_from_tensor_with_ixs(V_tens, [mid, site_r, link_r]).to(torch.complex128)
    return reshape_fortran(block, (mid.dim, site_r.dim * link_r.dim))


def _augmented_left_isometry_from_k(
    U0_tens,
    K1_tens,
    *args,
    link_l=None,
    site_l=None,
    old_mid=None,
    augment: bool = True,
    max_rank: int | float = math.inf,
    aug_tol: float = 1e-12,
    **kwargs: Any,
):
    """Build the augmented left isometry tensor from K-step directions.

    Args:
        U0_tens: Current left canonical factor.
        K1_tens: Stacked K-step directions.
        *args: Legacy positional tail ``(link_l, site_l, old_mid)``.
        link_l: Left bond index.
        site_l: Left physical site index.
        old_mid: Current middle bond index.
        augment: Whether new Krylov directions may enlarge the basis.
        max_rank: Hard cap on the returned basis rank.
        aug_tol: Threshold used to discard nearly dependent directions.
        **kwargs: Keyword overrides for index parameters.

    Returns:
        ``(U_aug_tens, overlap_tens, n_new)``.
    """

    if args:
        if len(args) != 3:
            raise TypeError("_augmented_left_isometry_from_k expects (link_l, site_l, old_mid) after the tensors.")
        if any(name in ("link_l", "site_l", "old_mid") for name in kwargs):
            raise TypeError("Provide left-augmentation indices either positionally or by keyword, not both.")
        link_l, site_l, old_mid = args
    if link_l is None or site_l is None or old_mid is None:
        try:
            link_l = kwargs.pop("link_l") if link_l is None else link_l
            site_l = kwargs.pop("site_l") if site_l is None else site_l
            old_mid = kwargs.pop("old_mid") if old_mid is None else old_mid
        except KeyError as exc:
            raise TypeError("Missing left-augmentation index input.") from exc
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown left-augmentation option(s): {unknown}")

    U0_mat = _left_tensor_matrix(U0_tens, link_l, site_l, old_mid)
    K1_mat = _left_tensor_matrix(K1_tens, link_l, site_l, old_mid)
    U1_mat, overlap_mat, n_new = _pick_left_update(U0_mat, K1_mat, augment=augment, max_rank=max_rank, aug_tol=aug_tol)
    new_mid = Ix(old_mid.itag, U1_mat.shape[1], old_mid.direction)
    U1_tens = make_tensor(
        reshape_fortran(U1_mat, (link_l.dim, site_l.dim, new_mid.dim)),
        [link_l, site_l, new_mid],
        dtype=torch.complex128,
    )
    overlap_tens = make_tensor(
        overlap_mat,
        [Ix(new_mid.itag, new_mid.dim, old_mid.direction), old_mid],
        dtype=torch.complex128,
    )
    return U1_tens, overlap_tens, n_new


def _augmented_right_isometry_from_l(
    V0_tens,
    L1_tens,
    *args,
    old_mid=None,
    site_r=None,
    link_r=None,
    augment: bool = True,
    max_rank: int | float = math.inf,
    aug_tol: float = 1e-12,
    **kwargs: Any,
):
    """Build the augmented right isometry tensor from L-step directions.

    Args:
        V0_tens: Current right canonical factor.
        L1_tens: Stacked L-step directions.
        *args: Legacy positional tail ``(old_mid, site_r, link_r)``.
        old_mid: Current middle bond index.
        site_r: Right physical site index.
        link_r: Right bond index.
        augment: Whether new Krylov directions may enlarge the basis.
        max_rank: Hard cap on the returned basis rank.
        aug_tol: Threshold used to discard nearly dependent directions.
        **kwargs: Keyword overrides for index parameters.

    Returns:
        ``(V_aug_tens, overlap_tens, n_new)``.
    """

    if args:
        if len(args) != 3:
            raise TypeError("_augmented_right_isometry_from_l expects (old_mid, site_r, link_r) after the tensors.")
        if any(name in ("old_mid", "site_r", "link_r") for name in kwargs):
            raise TypeError("Provide right-augmentation indices either positionally or by keyword, not both.")
        old_mid, site_r, link_r = args
    if old_mid is None or site_r is None or link_r is None:
        try:
            old_mid = kwargs.pop("old_mid") if old_mid is None else old_mid
            site_r = kwargs.pop("site_r") if site_r is None else site_r
            link_r = kwargs.pop("link_r") if link_r is None else link_r
        except KeyError as exc:
            raise TypeError("Missing right-augmentation index input.") from exc
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown right-augmentation option(s): {unknown}")

    V0_mat = _right_tensor_matrix(V0_tens, old_mid, site_r, link_r)
    L1_mat = _right_tensor_matrix(L1_tens, old_mid, site_r, link_r)
    V1_mat, overlap_mat, n_new = _pick_right_update(V0_mat, L1_mat, augment=augment, max_rank=max_rank, aug_tol=aug_tol)
    new_mid = Ix(old_mid.itag, V1_mat.shape[0], old_mid.direction)
    V1_tens = make_tensor(
        reshape_fortran(V1_mat, (new_mid.dim, site_r.dim, link_r.dim)),
        [new_mid, site_r, link_r],
        dtype=torch.complex128,
    )
    overlap_tens = make_tensor(
        overlap_mat,
        [old_mid, Ix(new_mid.itag, new_mid.dim, old_mid.direction)],
        dtype=torch.complex128,
    )
    return V1_tens, overlap_tens, n_new


def _transported_s_start_from_augmented_bases(U_basis, V_basis, theta0_tens, *args, **kwargs):
    """Project ``theta0_tens`` into augmented bases to form an initial S tensor.

    Args:
        U_basis: Left augmented basis matrix.
        V_basis: Right augmented basis matrix.
        theta0_tens: Two-site tensor to project.
        *args: Legacy positional tail ``(link_l, site_l, site_r, link_r)``.
        **kwargs: Keyword form of the same four indices.

    Returns:
        Rank-2 Nicole tensor containing the projected S data.
    """

    if args:
        if len(args) != 4:
            raise TypeError("_transported_s_start_from_augmented_bases expects four index arguments.")
        if any(name in kwargs for name in ("link_l", "site_l", "site_r", "link_r")):
            raise TypeError("Provide transport indices either positionally or by keyword, not both.")
        kwargs.update({"link_l": args[0], "site_l": args[1], "site_r": args[2], "link_r": args[3]})
    try:
        link_l = kwargs.pop("link_l")
        site_l = kwargs.pop("site_l")
        site_r = kwargs.pop("site_r")
        link_r = kwargs.pop("link_r")
    except KeyError as exc:
        raise TypeError("Missing transported-S basis index input.") from exc
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown transported-S option(s): {unknown}")

    theta = to_dense(theta0_tens, [link_l.itag, site_l.itag, site_r.itag, link_r.itag]).to(torch.complex128)
    theta_mat = reshape_fortran(theta, (link_l.dim * site_l.dim, site_r.dim * link_r.dim))
    S = U_basis.conj().transpose(0, 1) @ theta_mat @ V_basis.conj().transpose(0, 1)
    return make_tensor(
        S,
        [Ix("s_mid_l", U_basis.shape[1], link_l.direction), Ix("s_mid_r", V_basis.shape[0], link_r.direction)],
        dtype=torch.complex128,
    )


def _advance_s_tensor_in_bases(H_eff_mat, dt: complex, S_old_tens):
    """Evolve the S tensor with `linear_substep(..., method='expv')`."""
    block = to_dense(S_old_tens, list(S_old_tens.itags)).to(torch.complex128)
    s_old = flatten_fortran(block)
    s_new, numops = linear_substep(
        H_eff_mat,
        active_time_prefactor() * dt,
        s_old,
        method="expv",
        lanczos_tol=1e-14,
        lanczos_maxiter=max(4, len(s_old)),
    )
    shape = block.shape
    out = make_tensor(
        reshape_fortran(s_new, shape),
        [
            Ix(S_old_tens.itags[0], shape[0], S_old_tens.indices[0].direction),
            Ix(S_old_tens.itags[1], shape[1], S_old_tens.indices[1].direction),
        ],
        dtype=torch.complex128,
    )
    return out, numops


def _truncate_quantum_s_step(S_new_tens, maxdim: int):
    """Truncate `S_new_tens` by SVD and return split factors for write-back."""
    block = to_dense(S_new_tens, list(S_new_tens.itags)).to(torch.complex128)
    U, s, Vh = torch.linalg.svd(block, full_matrices=False)
    keep = min(int(s.numel()), int(maxdim))
    U_s = U[:, :keep]
    SV = torch.diag(s[:keep]) @ Vh[:keep, :]
    U_tens = make_tensor(
        U_s,
        [Ix(S_new_tens.itags[0], U_s.shape[0], S_new_tens.indices[0].direction), Ix("keep", keep, S_new_tens.indices[0].direction.reverse())],
        dtype=torch.complex128,
    )
    SV_tens = make_tensor(
        SV,
        [Ix("keep", keep, S_new_tens.indices[1].direction), Ix(S_new_tens.itags[1], SV.shape[1], S_new_tens.indices[1].direction)],
        dtype=torch.complex128,
    )
    return U_tens, SV_tens, keep, s


def _truncate_quantum_s_step_reverse(S_new_tens, maxdim: int):
    """Reverse-sweep alias of `_truncate_quantum_s_step`."""
    return _truncate_quantum_s_step(S_new_tens, maxdim)


def _stack_left_krylov_directions(directions, link_l: Ix, site_l: Ix, mid_k: Ix):
    if len(directions) == 1:
        return directions[0], mid_k

    mats = [_left_tensor_matrix(direction, link_l, site_l, mid_k) for direction in directions]
    sectors = tuple(Sector(int(sec.charge), int(sec.dim * len(directions))) for sec in resolved_sectors(mid_k))
    ext_mid = Ix(fresh_itag(mid_k.itag), sum(sec.dim for sec in sectors), mid_k.direction, sectors, mid_k.group)
    old_offsets = _sector_offsets(mid_k)
    new_offsets = _sector_offsets(ext_mid)
    stacked = torch.zeros((link_l.dim * site_l.dim, ext_mid.dim), dtype=torch.complex128, device=mats[0].device)
    for sec in resolved_sectors(mid_k):
        old_start, old_dim = old_offsets[sec.charge]
        new_start, _ = new_offsets[sec.charge]
        old_sl = slice(old_start, old_start + old_dim)
        for depth, mat in enumerate(mats):
            new_sl = slice(new_start + depth * old_dim, new_start + (depth + 1) * old_dim)
            stacked[:, new_sl] = mat[:, old_sl]
    tensor = make_tensor(
        reshape_fortran(stacked, (link_l.dim, site_l.dim, ext_mid.dim)),
        [link_l, site_l, ext_mid],
        dtype=torch.complex128,
    )
    return tensor, ext_mid


def _stack_right_krylov_directions(directions, mid_l: Ix, site_r: Ix, link_r: Ix):
    if len(directions) == 1:
        return directions[0], mid_l

    mats = [_right_tensor_matrix(direction, mid_l, site_r, link_r) for direction in directions]
    sectors = tuple(Sector(int(sec.charge), int(sec.dim * len(directions))) for sec in resolved_sectors(mid_l))
    ext_mid = Ix(fresh_itag(mid_l.itag), sum(sec.dim for sec in sectors), mid_l.direction, sectors, mid_l.group)
    old_offsets = _sector_offsets(mid_l)
    new_offsets = _sector_offsets(ext_mid)
    stacked = torch.zeros((ext_mid.dim, site_r.dim * link_r.dim), dtype=torch.complex128, device=mats[0].device)
    for sec in resolved_sectors(mid_l):
        old_start, old_dim = old_offsets[sec.charge]
        new_start, _ = new_offsets[sec.charge]
        old_sl = slice(old_start, old_start + old_dim)
        for depth, mat in enumerate(mats):
            new_sl = slice(new_start + depth * old_dim, new_start + (depth + 1) * old_dim)
            stacked[new_sl, :] = mat[old_sl, :]
    tensor = make_tensor(
        reshape_fortran(stacked, (ext_mid.dim, site_r.dim, link_r.dim)),
        [ext_mid, site_r, link_r],
        dtype=torch.complex128,
    )
    return tensor, ext_mid
