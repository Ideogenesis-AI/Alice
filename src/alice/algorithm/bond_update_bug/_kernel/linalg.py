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

"""Linear-algebra helpers for Nicole tensors and their dense projections.

The higher-level BUG code talks in terms of QR, LQ, SVD, and orthonormal basis
completion. This module wraps Nicole's decompositions with small Python helpers
that preserve the richer :class:`bug_nicole.indices.Ix` metadata and expose a
friendlier, more explicit surface to the rest of the package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch
from nicole import Tensor
from nicole import decomp as _ndecomp

from .indices import Ix, fresh_itag
from .nicole_helpers import tcontract

__all__ = [
    "SVDOptions",
    "identity_overlap_matrix",
    "lq",
    "qr",
    "qr_column_basis",
    "qr_nonzero_diagonal_rank",
    "qr_row_basis",
    "random_unitary",
    "reconstruct_from_svd",
    "svd",
    "truncate",
]


@dataclass(frozen=True)
class SVDOptions:
    """Options controlling the tensor SVD wrapper.

    Args:
        maxdim: Maximum kept bond dimension. ``math.inf`` means no explicit cap.
        cutoff: Relative singular-value cutoff.
        tag: Base tag used for the newly created bond indices.
    """

    maxdim: int | float = math.inf
    cutoff: float = 0.0
    tag: str = "b"


_SVD_OPTION_FIELDS = {field.name for field in SVDOptions.__dataclass_fields__.values()}


def _coerce_svd_options(options: SVDOptions | None = None, **kwargs: Any) -> SVDOptions:
    """Normalize SVD options from an object or legacy keyword arguments.

    Args:
        options: Existing :class:`SVDOptions` instance.
        **kwargs: Field overrides such as ``maxdim=64``.

    Returns:
        A normalized :class:`SVDOptions` instance.
    """
    overrides = {name: kwargs.pop(name) for name in list(kwargs) if name in _SVD_OPTION_FIELDS}
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown SVD option(s): {unknown}")
    if options is None:
        return SVDOptions(**overrides)
    return replace(options, **overrides)


def _axis_positions(tensor: Tensor, ixs: Sequence[Ix]) -> list[int]:
    """Map a sequence of ``Ix`` handles to their axis positions in a tensor.

    Args:
        tensor: Nicole tensor whose itags should be searched.
        ixs: Sequence of :class:`Ix` handles.

    Returns:
        The matching axis positions in ``tensor``.
    """
    positions: list[int] = []
    for ix in ixs:
        try:
            positions.append(tensor.itags.index(ix.itag))
        except ValueError as exc:
            raise ValueError(f"itag {ix.itag!r} is missing from tensor tags {tensor.itags}.") from exc
    return positions


def _axes_arg(positions: Sequence[int]) -> int | list[int]:
    """Convert one or many positions into Nicole's decomp ``axes`` argument.

    Args:
        positions: Axis positions selected for one side of a decomposition.

    Returns:
        Either a single integer or a list of integers.
    """
    return positions[0] if len(positions) == 1 else list(positions)


def _remaining_positions(tensor: Tensor, selected: Sequence[int]) -> list[int]:
    """Return every axis position not present in ``selected``.

    Args:
        tensor: Nicole tensor whose axes are being partitioned.
        selected: Selected axis positions.

    Returns:
        The complementary axis positions.
    """
    selected_set = set(selected)
    return [axis for axis in range(len(tensor.indices)) if axis not in selected_set]


def _bond_ix(tensor: Tensor, axis: int) -> Ix:
    """Wrap one Nicole tensor leg as an :class:`Ix`.

    Args:
        tensor: Nicole tensor.
        axis: Axis to wrap.

    Returns:
        An :class:`Ix` view of that tensor leg.
    """
    index = tensor.indices[axis]
    return Ix(tensor.itags[axis], int(index.dim), index.direction, index.sectors, index.group)


def _clone_tensor_with_ixs(tensor: Tensor, ixs: Sequence[Ix]) -> Tensor:
    """Clone a Nicole tensor and replace its index metadata with ``ixs``.

    Args:
        tensor: Tensor whose data should be preserved.
        ixs: Replacement wrapped indices.

    Returns:
        A cloned Nicole tensor with the requested tags and indices.
    """
    if len(ixs) != len(tensor.indices):
        raise ValueError(f"Cannot reattach {len(ixs)} indices to rank-{len(tensor.indices)} tensor.")
    indices = tuple(ix.nicole() for ix in ixs)
    itags = tuple(ix.itag for ix in ixs)
    data = {tuple(key): block.clone() for key, block in tensor.data.items()}
    intw = None if tensor.intw is None else {tuple(key): bridge.clone() for key, bridge in tensor.intw.items()}
    return Tensor(indices=indices, itags=itags, data=data, intw=intw, dtype=tensor.dtype, label=tensor.label)


def qr(A: Tensor, Qixs: Sequence[Ix], tag: str = "b", positive: bool = False):
    """Split a tensor into ``(Q, R, new_bond)`` using Nicole's QR.

    Args:
        A: Tensor to split.
        Qixs: Legs that should remain on the ``Q`` side.
        tag: Base tag for the new bond.
        positive: Preserved for API compatibility. Nicole's native phase
            convention is used unchanged.

    Returns:
        ``(Q, R, bond)`` where ``bond`` is the new :class:`Ix` wrapper.
    """
    positions = _axis_positions(A, Qixs)
    bond_tag = fresh_itag(tag)
    Q, R = _ndecomp(A, _axes_arg(positions), mode="QR", itag=bond_tag)

    # The port never requests positive QR phases, but we keep the parameter so
    # callers can remain close to the Julia API.
    if positive:
        pass

    bond = _bond_ix(Q, len(Q.indices) - 1)
    remaining = [_bond_ix(A, axis) for axis in _remaining_positions(A, positions)]
    q_ixs = [*Qixs, bond]
    r_bond = Ix(bond.itag, bond.dim, R.indices[0].direction, bond.sectors, bond.group)
    r_ixs = [r_bond, *remaining]
    return _clone_tensor_with_ixs(Q, q_ixs), _clone_tensor_with_ixs(R, r_ixs), bond


def lq(A: Tensor, Qixs: Sequence[Ix], tag: str = "b"):
    """Split a tensor into a left factor and right isometry via Nicole ``LV``.

    Args:
        A: Tensor to split.
        Qixs: Legs that should remain on the right-isometric factor.
        tag: Base tag for the new bond.

    Returns:
        ``(L, Q, bond)`` where ``Q`` is right-isometric and ``bond`` is the new
        :class:`Ix` wrapper.
    """
    q_positions = _axis_positions(A, Qixs)
    left_positions = _remaining_positions(A, q_positions)
    bond_tag = fresh_itag(tag)
    L, Q = _ndecomp(A, _axes_arg(left_positions), mode="LV", itag=bond_tag)

    left_ixs = [_bond_ix(A, axis) for axis in left_positions]
    bond = _bond_ix(Q, 0)
    l_bond = Ix(bond.itag, bond.dim, L.indices[-1].direction, bond.sectors, bond.group)
    return _clone_tensor_with_ixs(L, [*left_ixs, l_bond]), _clone_tensor_with_ixs(Q, [bond, *Qixs]), bond


def truncate(s: torch.Tensor, maxdim: int | float, cutoff: float) -> int:
    """Compute how many singular values should be kept.

    Args:
        s: Singular values sorted in descending order.
        maxdim: Explicit cap on the kept rank.
        cutoff: Relative cutoff measured against ``abs(s[0])``.

    Returns:
        The kept rank after applying the cutoff and cap.
    """
    if s.numel() == 0:
        return 0
    if maxdim is None or maxdim == math.inf:
        maxdim_int = int(s.numel())
    else:
        maxdim_int = max(1, int(maxdim))

    thresh = float(cutoff) * float(torch.abs(s[0]))
    keep = int((torch.abs(s) > thresh).sum().item())
    if keep == 0:
        keep = 1
    return min(keep, maxdim_int, int(s.numel()))


def svd(
    A: Tensor,
    Uixs: Sequence[Ix],
    *,
    options: SVDOptions | None = None,
    **kwargs: Any,
):
    """Split a tensor into ``(U, S, V, bond_u, bond_v)`` using Nicole's SVD.

    Args:
        A: Tensor to split.
        Uixs: Legs that should remain on the left factor ``U``.
        options: Optional :class:`SVDOptions` instance.
        **kwargs: Legacy overrides such as ``maxdim=64`` or ``cutoff=1e-12``.

    Returns:
        ``(U, S, V, bond_u, bond_v)`` with the richer :class:`Ix` metadata
        restored on the tensor factors.
    """
    options = _coerce_svd_options(options, **kwargs)
    positions = _axis_positions(A, Uixs)
    left_tag = fresh_itag(options.tag)
    right_tag = fresh_itag(f"{options.tag}r")

    trunc_spec: dict[str, int | float] = {}
    if options.maxdim is not None and options.maxdim != math.inf:
        trunc_spec["nkeep"] = int(options.maxdim)
    if options.cutoff > 0:
        trunc_spec["thresh"] = float(options.cutoff)

    U, S, V = _ndecomp(
        A,
        _axes_arg(positions),
        mode="SVD",
        itag=(left_tag, right_tag),
        trunc=trunc_spec or None,
    )

    bond_u = _bond_ix(U, len(U.indices) - 1)
    bond_v = _bond_ix(V, 0)
    remaining = [_bond_ix(A, axis) for axis in _remaining_positions(A, positions)]
    return (
        _clone_tensor_with_ixs(U, [*Uixs, bond_u]),
        S,
        _clone_tensor_with_ixs(V, [bond_v, *remaining]),
        bond_u,
        bond_v,
    )


def random_unitary(m: int, n: int | None = None, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Sample a matrix with orthonormal columns.

    Args:
        m: Ambient row dimension.
        n: Number of orthonormal columns. Defaults to ``m``.
        dtype: Output dtype.

    Returns:
        An ``m x n`` matrix whose columns are orthonormal.
    """
    if n is None:
        n = m
    if n > m:
        raise ValueError(f"n must satisfy n<=m; got n={n}, m={m}")

    real = torch.randn((m, m), dtype=torch.float64)
    if dtype.is_complex:
        imag = torch.randn((m, m), dtype=torch.float64)
        mat = (real + 1j * imag).to(dtype=dtype)
    else:
        mat = real.to(dtype=dtype)

    q, r = torch.linalg.qr(mat, mode="reduced")

    # Normalize the QR phases so the result is invariant under the arbitrary QR
    # sign/phase convention returned by torch.
    diag = torch.diagonal(r)
    phases = torch.ones_like(diag)
    nonzero = diag != 0
    phases[nonzero] = diag[nonzero] / torch.abs(diag[nonzero])
    q = q * phases.conj().unsqueeze(0)
    return q[:, :n]


def qr_nonzero_diagonal_rank(rmat: torch.Tensor, tol: float | None = None) -> int:
    """Estimate the numerical rank of a QR ``R`` factor.

    Args:
        rmat: Upper-triangular QR factor.
        tol: Optional magnitude threshold.

    Returns:
        The number of diagonal entries above ``tol``.
    """
    diag = torch.abs(torch.diagonal(rmat))
    if diag.numel() == 0:
        return 0
    if tol is None:
        tol = max(rmat.shape) * torch.finfo(diag.dtype).eps * float(diag.max())
    return int(torch.count_nonzero(diag > tol).item())


def qr_column_basis(a: torch.Tensor, tol: float | None = None) -> tuple[torch.Tensor, int]:
    """Return an orthonormal basis for the column space of ``a``.

    Args:
        a: Dense matrix.
        tol: Optional QR rank tolerance.

    Returns:
        ``(basis, rank)``.
    """
    q, r = torch.linalg.qr(a, mode="reduced")
    rank = qr_nonzero_diagonal_rank(r, tol)
    return q[:, :rank], rank


def qr_row_basis(a: torch.Tensor, tol: float | None = None) -> tuple[torch.Tensor, int]:
    """Return a row-orthonormal basis for the row space of ``a``.

    Args:
        a: Dense matrix.
        tol: Optional QR rank tolerance.

    Returns:
        ``(basis, rank)`` where the basis rows span the row space of ``a``.
    """
    q, r = torch.linalg.qr(a.transpose(-2, -1), mode="reduced")
    rank = qr_nonzero_diagonal_rank(r, tol)
    return q[:, :rank].transpose(-2, -1), rank


def identity_overlap_matrix(dtype: torch.dtype, n: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return an ``n x n`` identity matrix for no-augmentation overlaps.

    Args:
        dtype: Matrix dtype.
        n: Matrix size.
        device: Optional torch device.

    Returns:
        An identity matrix.
    """
    return torch.eye(n, dtype=dtype, device=device)


def reconstruct_from_svd(U: Tensor, S: Tensor, V: Tensor) -> Tensor:
    """Reconstruct ``U * S * V`` in tensor form.

    Args:
        U: Left SVD tensor.
        S: Diagonal singular-value tensor.
        V: Right SVD tensor.

    Returns:
        The contracted reconstruction.
    """
    return tcontract(tcontract(U, S), V)
