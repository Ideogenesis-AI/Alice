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

"""Symmetric U(1)-aware augmented isometry functions for BUG/KLS updates."""

from __future__ import annotations

import math
from typing import Any

import torch
from nicole import Sector

from ..indices import Ix, fresh_itag, resolved_sectors
from ..linalg import complete_column_basis, complete_row_basis
from ..nicole_helpers import make_tensor, reshape_fortran
from .augment import (
    _left_row_indices_by_flux,
    _left_tensor_matrix,
    _pick_left_update,
    _pick_right_update,
    _right_col_indices_by_flux,
    _right_tensor_matrix,
    _sector_offsets,
)


def _kl_truncated_left(U0_sub, K1_sub, kl_cutoff: float, max_rank):
    """Augmented left block ``[U0 | SVD-weight-truncated discarded complement of K1]``.

    Instead of completing ``U0`` to the full local capacity (``d*r``), keep ``U0``
    EXACTLY and admit only the discarded directions ``(I - U0 U0+) K1`` whose singular
    value clears ``kl_cutoff`` (relative to the top one) — the per-bond analogue of the
    global discarded_bug's SVD-truncated complement. This caps the augmented rank
    between ``r`` and ``d*r`` instead of always ``d*r``. Returns ``(Q, overlap, n_new)``
    with ``Q = [U0 | Q_new]`` orthonormal (``Q_new`` is orthogonal to ``U0`` by
    construction). ``overlap`` (the M-hat block) is computed for signature parity but is
    unused by the discarded variant, which seeds the S-step from ``Û† Θ0 V̂†`` directly.
    """
    r = U0_sub.shape[1]
    eye = U0_sub.conj().transpose(0, 1) @ U0_sub
    if K1_sub.numel() == 0 or K1_sub.shape[1] == 0:
        return U0_sub, eye, 0
    k_perp = K1_sub - U0_sub @ (U0_sub.conj().transpose(0, 1) @ K1_sub)
    u_k, s_k, _ = torch.linalg.svd(k_perp, full_matrices=False)
    if s_k.numel() == 0 or float(s_k[0]) == 0.0:
        return U0_sub, eye, 0
    keep = int((s_k > kl_cutoff * float(s_k[0])).sum().item())
    if max_rank is not math.inf:
        keep = min(keep, max(0, int(max_rank) - r))
    if keep <= 0:
        return U0_sub, eye, 0
    Q = torch.cat([U0_sub, u_k[:, :keep]], dim=1)
    overlap = Q.conj().transpose(0, 1) @ U0_sub
    return Q, overlap, keep


def _kl_truncated_right(V0_sub, L1_sub, kl_cutoff: float, max_rank):
    """Augmented right block ``[V0 ; SVD-weight-truncated discarded complement of L1]`` (row-wise mirror)."""
    r = V0_sub.shape[0]
    eye = V0_sub @ V0_sub.conj().transpose(0, 1)
    if L1_sub.numel() == 0 or L1_sub.shape[0] == 0:
        return V0_sub, eye, 0
    l_perp = L1_sub - (L1_sub @ V0_sub.conj().transpose(0, 1)) @ V0_sub
    _, s_l, vh_l = torch.linalg.svd(l_perp, full_matrices=False)
    if s_l.numel() == 0 or float(s_l[0]) == 0.0:
        return V0_sub, eye, 0
    keep = int((s_l > kl_cutoff * float(s_l[0])).sum().item())
    if max_rank is not math.inf:
        keep = min(keep, max(0, int(max_rank) - r))
    if keep <= 0:
        return V0_sub, eye, 0
    B = torch.cat([V0_sub, vh_l[:keep, :]], dim=0)
    overlap = V0_sub @ B.conj().transpose(0, 1)
    return B, overlap, keep


def _symmetric_augmented_left_isometry_from_k(
    U0_tens,
    K1_tens,
    *args,
    link_l=None,
    site_l=None,
    old_mid_u=None,
    k_mid=None,
    augment: bool = True,
    max_rank: int | float = math.inf,
    aug_tol: float = 1e-12,
    kl_cutoff: float | None = None,
    **kwargs: Any,
):
    if args:
        if len(args) != 4:
            raise TypeError(
                "_symmetric_augmented_left_isometry_from_k expects (link_l, site_l, old_mid_u, k_mid) after the tensors."
            )
        if any(name in ("link_l", "site_l", "old_mid_u", "k_mid") for name in kwargs):
            raise TypeError("Provide symmetric left-augmentation indices either positionally or by keyword, not both.")
        link_l, site_l, old_mid_u, k_mid = args
    if link_l is None or site_l is None or old_mid_u is None or k_mid is None:
        try:
            link_l = kwargs.pop("link_l") if link_l is None else link_l
            site_l = kwargs.pop("site_l") if site_l is None else site_l
            old_mid_u = kwargs.pop("old_mid_u") if old_mid_u is None else old_mid_u
            k_mid = kwargs.pop("k_mid") if k_mid is None else k_mid
        except KeyError as exc:
            raise TypeError("Missing symmetric left-augmentation index input.") from exc
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown symmetric left-augmentation option(s): {unknown}")

    dtype = torch.complex128
    device = next(iter(U0_tens.data.values())).device if U0_tens.data else torch.device("cpu")
    U0_mat = _left_tensor_matrix(U0_tens, link_l, site_l, old_mid_u)
    K1_mat = _left_tensor_matrix(K1_tens, link_l, site_l, k_mid)
    row_blocks = _left_row_indices_by_flux(link_l, site_l)
    u_offsets = _sector_offsets(old_mid_u)
    k_offsets = _sector_offsets(k_mid)
    d_old = int(old_mid_u.direction)
    d_k = int(k_mid.direction)

    fluxes: list[object] = []
    for sec in resolved_sectors(old_mid_u):
        flux = d_old * sec.charge
        if flux not in fluxes:
            fluxes.append(flux)
    for sec in resolved_sectors(k_mid):
        flux = d_k * sec.charge
        if flux not in fluxes:
            fluxes.append(flux)
    for flux in row_blocks:
        if flux not in fluxes:
            fluxes.append(flux)

    pieces: list[tuple[object, list[int], torch.Tensor, torch.Tensor]] = []
    total_dim = 0
    n_new_total = 0
    for flux in fluxes:
        rows = row_blocks.get(flux, [])
        if not rows:
            continue

        old_charge = flux // d_old
        k_charge = flux // d_k
        old_slice = u_offsets.get(old_charge)
        k_slice = k_offsets.get(k_charge)
        U0_sub = U0_mat[rows, old_slice[0] : old_slice[0] + old_slice[1]] if old_slice else torch.zeros((len(rows), 0), dtype=dtype, device=device)
        K1_sub = K1_mat[rows, k_slice[0] : k_slice[0] + k_slice[1]] if k_slice else torch.zeros((len(rows), 0), dtype=dtype, device=device)
        if kl_cutoff is not None and augment:
            # Efficient path: keep U0 exact, admit only the SVD-weight-significant
            # discarded directions of K1 (no full d*r completion).
            Q_block, overlap_block, n_new = _kl_truncated_left(U0_sub, K1_sub, kl_cutoff, max_rank)
        else:
            Q_block, overlap_block, n_new = _pick_left_update(U0_sub, K1_sub, augment=augment, max_rank=max_rank, aug_tol=aug_tol)
            if augment:
                Q_block = complete_column_basis(Q_block)
                overlap_block = Q_block.conj().transpose(0, 1) @ U0_sub
                n_new = Q_block.shape[1] - U0_sub.shape[1]
        if Q_block.shape[1] == 0:
            continue
        pieces.append((old_charge, rows, Q_block, overlap_block))
        total_dim += int(Q_block.shape[1])
        n_new_total += int(n_new)

    if total_dim == 0:
        raise ValueError("Left symmetric K-step augmentation produced zero rank.")

    sectors = tuple(Sector(int(charge), int(block.shape[1])) for charge, _, block, _ in pieces)
    new_mid = Ix(fresh_itag(old_mid_u.itag), total_dim, old_mid_u.direction, sectors, old_mid_u.group)
    U_aug_mat = torch.zeros((link_l.dim * site_l.dim, total_dim), dtype=dtype, device=device)
    M_hat_mat = torch.zeros((total_dim, old_mid_u.dim), dtype=dtype, device=device)

    cursor = 0
    for charge, rows, block, overlap_block in pieces:
        width = int(block.shape[1])
        sl = slice(cursor, cursor + width)
        U_aug_mat[rows, sl] = block
        old_slice = u_offsets.get(charge)
        if old_slice is not None and overlap_block.numel():
            old_sl = slice(old_slice[0], old_slice[0] + old_slice[1])
            M_hat_mat[sl, old_sl] = overlap_block
        cursor += width

    U_aug_tens = make_tensor(
        reshape_fortran(U_aug_mat, (link_l.dim, site_l.dim, new_mid.dim)),
        [link_l, site_l, new_mid],
        dtype=dtype,
    )
    M_hat_tens = make_tensor(M_hat_mat, [new_mid.reversed(), old_mid_u], dtype=dtype)
    return U_aug_tens, M_hat_tens, n_new_total


def _symmetric_augmented_right_isometry_from_l(
    V0_tens,
    L1_tens,
    *args,
    old_mid_v=None,
    l_mid=None,
    site_r=None,
    link_r=None,
    augment: bool = True,
    max_rank: int | float = math.inf,
    aug_tol: float = 1e-12,
    kl_cutoff: float | None = None,
    **kwargs: Any,
):
    if args:
        if len(args) != 4:
            raise TypeError(
                "_symmetric_augmented_right_isometry_from_l expects (old_mid_v, l_mid, site_r, link_r) after the tensors."
            )
        if any(name in ("old_mid_v", "l_mid", "site_r", "link_r") for name in kwargs):
            raise TypeError("Provide symmetric right-augmentation indices either positionally or by keyword, not both.")
        old_mid_v, l_mid, site_r, link_r = args
    if old_mid_v is None or l_mid is None or site_r is None or link_r is None:
        try:
            old_mid_v = kwargs.pop("old_mid_v") if old_mid_v is None else old_mid_v
            l_mid = kwargs.pop("l_mid") if l_mid is None else l_mid
            site_r = kwargs.pop("site_r") if site_r is None else site_r
            link_r = kwargs.pop("link_r") if link_r is None else link_r
        except KeyError as exc:
            raise TypeError("Missing symmetric right-augmentation index input.") from exc
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown symmetric right-augmentation option(s): {unknown}")

    dtype = torch.complex128
    device = next(iter(V0_tens.data.values())).device if V0_tens.data else torch.device("cpu")
    V0_mat = _right_tensor_matrix(V0_tens, old_mid_v, site_r, link_r)
    L1_mat = _right_tensor_matrix(L1_tens, l_mid, site_r, link_r)
    col_blocks = _right_col_indices_by_flux(site_r, link_r)
    v_offsets = _sector_offsets(old_mid_v)
    l_offsets = _sector_offsets(l_mid)
    d_old = int(old_mid_v.direction)
    d_l = int(l_mid.direction)

    fluxes: list[object] = []
    for sec in resolved_sectors(old_mid_v):
        flux = d_old * sec.charge
        if flux not in fluxes:
            fluxes.append(flux)
    for sec in resolved_sectors(l_mid):
        flux = d_l * sec.charge
        if flux not in fluxes:
            fluxes.append(flux)
    for flux in col_blocks:
        if flux not in fluxes:
            fluxes.append(flux)

    pieces: list[tuple[object, list[int], torch.Tensor, torch.Tensor]] = []
    total_dim = 0
    n_new_total = 0
    for flux in fluxes:
        cols = col_blocks.get(flux, [])
        if not cols:
            continue

        old_charge = flux // d_old
        l_charge = flux // d_l
        old_slice = v_offsets.get(old_charge)
        l_slice = l_offsets.get(l_charge)
        V0_sub = V0_mat[old_slice[0] : old_slice[0] + old_slice[1], cols] if old_slice else torch.zeros((0, len(cols)), dtype=dtype, device=device)
        L1_sub = L1_mat[l_slice[0] : l_slice[0] + l_slice[1], cols] if l_slice else torch.zeros((0, len(cols)), dtype=dtype, device=device)
        if kl_cutoff is not None and augment:
            B_block, overlap_block, n_new = _kl_truncated_right(V0_sub, L1_sub, kl_cutoff, max_rank)
        else:
            B_block, overlap_block, n_new = _pick_right_update(V0_sub, L1_sub, augment=augment, max_rank=max_rank, aug_tol=aug_tol)
            if augment:
                B_block = complete_row_basis(B_block)
                overlap_block = V0_sub @ B_block.conj().transpose(0, 1)
                n_new = B_block.shape[0] - V0_sub.shape[0]
        if B_block.shape[0] == 0:
            continue
        pieces.append((old_charge, cols, B_block, overlap_block))
        total_dim += int(B_block.shape[0])
        n_new_total += int(n_new)

    if total_dim == 0:
        raise ValueError("Right symmetric L-step augmentation produced zero rank.")

    sectors = tuple(Sector(int(charge), int(block.shape[0])) for charge, _, block, _ in pieces)
    new_mid = Ix(fresh_itag(old_mid_v.itag), total_dim, old_mid_v.direction, sectors, old_mid_v.group)
    V_aug_mat = torch.zeros((total_dim, site_r.dim * link_r.dim), dtype=dtype, device=device)
    N_hat_mat = torch.zeros((old_mid_v.dim, total_dim), dtype=dtype, device=device)

    cursor = 0
    for charge, cols, block, overlap_block in pieces:
        height = int(block.shape[0])
        sl = slice(cursor, cursor + height)
        V_aug_mat[sl, cols] = block
        old_slice = v_offsets.get(charge)
        if old_slice is not None and overlap_block.numel():
            old_sl = slice(old_slice[0], old_slice[0] + old_slice[1])
            N_hat_mat[old_sl, sl] = overlap_block
        cursor += height

    V_aug_tens = make_tensor(
        reshape_fortran(V_aug_mat, (new_mid.dim, site_r.dim, link_r.dim)),
        [new_mid, site_r, link_r],
        dtype=dtype,
    )
    N_hat_tens = make_tensor(N_hat_mat, [old_mid_v, new_mid.reversed()], dtype=dtype)
    return V_aug_tens, N_hat_tens, n_new_total
