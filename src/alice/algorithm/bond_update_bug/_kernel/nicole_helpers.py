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

"""Nicole tensor convenience helpers used across the BUG stack.

These helpers keep the tensor-manipulation code readable by centralizing a few
repeated chores: Fortran-order reshaping, dense/block conversion, explicit
identity construction, and a small collection of Nicole-flavored conjugation and
contraction utilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import torch
from nicole import Direction, Tensor
from nicole import conj as _nconj
from nicole import contract as _ncontract
from nicole import identity as _nidentity
from nicole import inv as _ninv
from nicole.blocks import BlockSchema

from .indices import Ix, resolved_sectors

__all__ = [
    "IdentityOptions",
    "conj",
    "dag",
    "delta",
    "diag_tensor",
    "flatten_fortran",
    "identity_tensor",
    "inv_diag",
    "make_tensor",
    "norm",
    "prime_bra",
    "reshape_fortran",
    "scalar",
    "star_itags",
    "tcontract",
    "to_dense",
]


@dataclass(frozen=True)
class IdentityOptions:
    """Options for raw-tag identity and relabeling tensors.

    Args:
        dim: Dense index dimension used when the source is given only as a tag.
        sectors: Optional explicit sector tuple for raw-tag construction.
        group: Optional Nicole symmetry group handle for raw-tag construction.
        direction: Nicole direction of the source leg when only raw tag metadata
            is provided.
    """

    dim: int | None = None
    sectors: Sequence[object] | None = None
    group: object | None = None
    direction: Direction = Direction.IN


_IDENTITY_OPTION_FIELDS = {field.name for field in IdentityOptions.__dataclass_fields__.values()}


def _coerce_identity_options(options: IdentityOptions | None = None, **kwargs: Any) -> IdentityOptions:
    """Normalize identity-construction options from an object or kwargs.

    Args:
        options: Existing options object to start from.
        **kwargs: Field overrides for :class:`IdentityOptions`.

    Returns:
        A normalized :class:`IdentityOptions` instance.
    """
    overrides = {name: kwargs.pop(name) for name in list(kwargs) if name in _IDENTITY_OPTION_FIELDS}
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown identity option(s): {unknown}")
    if options is None:
        return IdentityOptions(**overrides)
    return replace(options, **overrides)


def _materialize_indices(ixs: Sequence[Ix]) -> tuple:
    """Convert wrapped indices into Nicole indices.

    Args:
        ixs: Sequence of :class:`Ix` wrappers.

    Returns:
        A tuple of Nicole :class:`Index` objects.
    """
    return tuple(ix.nicole() for ix in ixs)


def _index_offsets(ix: Ix | object) -> tuple[dict[object, tuple[int, int]], int]:
    """Map each sector charge to its dense offset along one axis.

    Args:
        ix: Wrapped or native Nicole index.

    Returns:
        A pair ``(offsets, total_dim)``.
    """
    offsets: dict[object, tuple[int, int]] = {}
    cursor = 0
    for sector in resolved_sectors(ix):
        offsets[sector.charge] = (cursor, sector.dim)
        cursor += sector.dim
    return offsets, cursor


def reshape_fortran(tensor: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Return the torch equivalent of ``reshape(..., order='F')``.

    Args:
        tensor: Input torch tensor.
        shape: Target shape interpreted in Fortran/column-major order.

    Returns:
        A reshaped tensor with the requested shape.
    """
    target = tuple(int(dim) for dim in shape)
    if math.prod(target) != int(tensor.numel()):
        raise ValueError(f"Cannot reshape tensor with {tensor.numel()} entries into {target}.")
    if len(target) == 0:
        return tensor.reshape(())
    if tensor.ndim == 0:
        return tensor.reshape(target)

    # Reverse, reshape, then reverse back to emulate column-major memory order.
    rev_in = tuple(reversed(range(tensor.ndim)))
    rev_out = tuple(reversed(range(len(target))))
    reshaped = tensor.permute(rev_in).contiguous().reshape(tuple(reversed(target)))
    return reshaped.permute(rev_out).contiguous()


def flatten_fortran(tensor: torch.Tensor) -> torch.Tensor:
    """Return the torch equivalent of ``reshape(-1, order='F')``.

    Args:
        tensor: Input torch tensor.

    Returns:
        A one-dimensional tensor flattened in Fortran/column-major order.
    """
    if tensor.ndim <= 1:
        return tensor.reshape(-1)
    return tensor.permute(tuple(reversed(range(tensor.ndim)))).contiguous().reshape(-1)


def _dense_to_block_data(
    dense: torch.Tensor,
    ixs: Sequence[Ix],
    *,
    tol: float = 1e-12,
) -> dict[tuple[object, ...], torch.Tensor]:
    """Project a dense tensor into the admissible Nicole block dictionary.

    Args:
        dense: Dense torch tensor with one axis per index in ``ixs``.
        ixs: Wrapped indices describing the target block structure.
        tol: Numerical tolerance used when discarding zero blocks and checking
            for amplitudes outside the admissible symmetry support.

    Returns:
        A Nicole-style block dictionary keyed by sector charges.
    """
    indices = _materialize_indices(ixs)
    if dense.ndim != len(indices):
        raise ValueError(f"Dense tensor rank {dense.ndim} does not match index rank {len(indices)}.")

    expected_shape = tuple(int(ix.dim) for ix in indices)
    if tuple(dense.shape) != expected_shape:
        raise ValueError(f"Dense tensor shape {tuple(dense.shape)} does not match index dims {expected_shape}.")

    offsets = [_index_offsets(ix) for ix in ixs]
    reconstructed = torch.zeros(expected_shape, dtype=dense.dtype, device=dense.device)
    data: dict[tuple[object, ...], torch.Tensor] = {}

    # Nicole only stores symmetry-admissible blocks, so we reconstruct the
    # admissible support and confirm nothing significant lives outside it.
    for key in BlockSchema.iter_admissible_keys(indices):
        if not BlockSchema.charges_conserved(indices, key):
            continue
        slices = tuple(
            slice(offsets[axis][0][key[axis]][0], offsets[axis][0][key[axis]][0] + offsets[axis][0][key[axis]][1])
            for axis in range(len(key))
        )
        block = dense[slices].clone().contiguous()
        reconstructed[slices] = block
        if block.numel() == 0:
            continue
        if torch.max(torch.abs(block)).item() > tol:
            data[tuple(key)] = block

    residual = dense - reconstructed
    if residual.numel() and torch.max(torch.abs(residual)).item() > tol:
        raise ValueError("Dense tensor contains amplitudes outside the admissible symmetry blocks.")

    return data


def make_tensor(
    array_or_blocks: torch.Tensor | Mapping[tuple[object, ...], object] | object,
    ixs: Sequence[Ix],
    *,
    dtype: torch.dtype | None = None,
    tol: float = 1e-12,
) -> Tensor:
    """Build a Nicole tensor from dense data or explicit block data.

    Args:
        array_or_blocks: Dense tensor-like data or a Nicole block dictionary.
        ixs: Wrapped indices describing the target tensor legs.
        dtype: Optional dtype override.
        tol: Tolerance forwarded to dense-to-block conversion.

    Returns:
        A Nicole :class:`Tensor` with the requested indices and tags.
    """
    indices = _materialize_indices(ixs)
    itags = tuple(ix.itag for ix in ixs)

    if isinstance(array_or_blocks, Mapping):
        data = {tuple(key): torch.as_tensor(value) for key, value in array_or_blocks.items()}
        if dtype is not None:
            data = {key: value.to(dtype=dtype) for key, value in data.items()}
        inferred_dtype = next(iter(data.values())).dtype if data else (dtype or torch.complex128)
        return Tensor(indices=indices, itags=itags, data=data, dtype=inferred_dtype)

    dense = torch.as_tensor(array_or_blocks)
    if dtype is not None:
        dense = dense.to(dtype=dtype)

    if len(ixs) == 0:
        scalar_value = dense.reshape(())
        return Tensor(indices=(), itags=(), data={(): scalar_value}, dtype=scalar_value.dtype)

    data = _dense_to_block_data(dense, ixs, tol=tol)
    return Tensor(indices=indices, itags=itags, data=data, dtype=dense.dtype)


def to_dense(tensor: Tensor, itag_order: Sequence[str]) -> torch.Tensor:
    """Assemble a dense tensor in the requested itag order.

    Args:
        tensor: Nicole tensor to densify.
        itag_order: Desired order of the tensor tags in the dense output.

    Returns:
        A dense torch tensor with axes permuted to ``itag_order``.
    """
    if len(tensor.indices) == 0:
        return next(iter(tensor.data.values())).reshape(())
    if len(itag_order) != len(tensor.itags):
        raise ValueError(f"itag_order length {len(itag_order)} does not match tensor rank {len(tensor.itags)}.")

    offsets = [_index_offsets(index) for index in tensor.indices]
    shape = tuple(total_dim for _, total_dim in offsets)
    full = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    for key, block in tensor.data.items():
        slices = tuple(
            slice(offsets[axis][0][key[axis]][0], offsets[axis][0][key[axis]][0] + offsets[axis][0][key[axis]][1])
            for axis in range(len(key))
        )
        full[slices] = block

    positions: dict[str, list[int]] = {}
    for axis, tag in enumerate(tensor.itags):
        positions.setdefault(tag, []).append(axis)

    used: dict[str, int] = {}
    permutation: list[int] = []
    for tag in itag_order:
        taken = used.get(tag, 0)
        axes = positions.get(tag)
        if axes is None or taken >= len(axes):
            raise ValueError(f"itag '{tag}' is missing from tensor tags {tensor.itags}.")
        permutation.append(axes[taken])
        used[tag] = taken + 1

    return full.permute(permutation).contiguous()


def dag(tensor: Tensor) -> Tensor:
    """Return Nicole's conjugated tensor with flipped directions.

    Args:
        tensor: Input Nicole tensor.

    Returns:
        The Nicole ``dag``/conjugation result.
    """
    return _nconj(tensor)


def conj(tensor: Tensor) -> Tensor:
    """Alias for Nicole's conjugation helper.

    Args:
        tensor: Input Nicole tensor.

    Returns:
        The conjugated Nicole tensor.
    """
    return _nconj(tensor)


def star_itags(tensor: Tensor) -> Tensor:
    """Clone a tensor and append ``*`` to every Nicole tag.

    Args:
        tensor: Input Nicole tensor.

    Returns:
        A cloned tensor with starred tags.
    """
    retagged = tensor.clone()
    retagged.retag({tag: f"{tag}*" for tag in retagged.itags})
    return retagged


def prime_bra(tensor: Tensor) -> Tensor:
    """Return the Nicole analogue of ``dag(prime(x))``.

    Args:
        tensor: Input Nicole tensor.

    Returns:
        A conjugated tensor whose itags have been starred.
    """
    bra = _nconj(tensor)
    bra.retag({tag: f"{tag}*" for tag in bra.itags})
    return bra


def tcontract(
    left: Tensor,
    right: Tensor,
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]] | None = None,
) -> Tensor:
    """Contract two Nicole tensors with a gentle outer-product fallback.

    Args:
        left: Left tensor.
        right: Right tensor.
        axes: Optional explicit contraction axes.

    Returns:
        The Nicole contraction result.
    """
    if axes is not None:
        return _ncontract(left, right, axes=axes)
    try:
        return _ncontract(left, right)
    except ValueError as exc:
        if "No valid contraction pairs found" not in str(exc):
            raise
        return _ncontract(left, right, axes=([], []))


def scalar(tensor: Tensor):
    """Extract a Python scalar from a scalar-like Nicole tensor.

    Args:
        tensor: Scalar tensor or tensor whose open indices all have dimension 1.

    Returns:
        The scalar value stored in the tensor.
    """
    if tensor.is_scalar():
        return tensor.item()
    if all(int(index.dim) == 1 for index in tensor.indices):
        return to_dense(tensor, list(tensor.itags)).reshape(-1)[0].item()
    raise ValueError("scalar() requires a scalar tensor or all-dimension-1 open indices.")


def norm(tensor: Tensor):
    """Return Nicole's norm for one tensor.

    Args:
        tensor: Input Nicole tensor.

    Returns:
        The tensor norm in Nicole's backend dtype.
    """
    return tensor.norm()


def delta(
    source: Ix | str,
    out_itag: str,
    dim: int | None = None,
    *,
    options: IdentityOptions | None = None,
    **kwargs: Any,
) -> Tensor:
    """Construct a relabeling identity tensor.

    Args:
        source: Source index wrapper or raw itag string.
        out_itag: Output Nicole tag.
        dim: Optional raw-tag dimension. This is ignored when ``source`` is an
            :class:`Ix`.
        options: Optional :class:`IdentityOptions` instance.
        **kwargs: Legacy option overrides such as ``sectors=...`` or
            ``direction=Direction.IN``.

    Returns:
        A Nicole identity tensor that relabels one leg to ``out_itag``.
    """
    options = _coerce_identity_options(options, dim=dim, **kwargs)
    if isinstance(source, Ix):
        return _nidentity(source.nicole(), itags=(source.itag, out_itag))
    if options.dim is None:
        raise ValueError("dim is required when building a delta tensor from raw itag metadata.")
    ix = Ix(source, options.dim, options.direction, None if options.sectors is None else tuple(options.sectors), options.group)
    return _nidentity(ix.nicole(), itags=(source, out_itag))


def identity_tensor(
    left: Ix | str,
    right: str | None = None,
    dim: int | None = None,
    *,
    options: IdentityOptions | None = None,
    **kwargs: Any,
) -> Tensor:
    """Construct an identity tensor used to extend local operators.

    Args:
        left: Source index wrapper or raw itag string.
        right: Optional output tag. When ``left`` is an :class:`Ix`, the default
            is ``f"{left.itag}*"``.
        dim: Optional raw-tag dimension.
        options: Optional :class:`IdentityOptions` instance.
        **kwargs: Legacy option overrides forwarded to :func:`delta`.

    Returns:
        A rank-2 Nicole identity tensor.
    """
    options = _coerce_identity_options(options, dim=dim, **kwargs)
    if isinstance(left, Ix):
        if right is None:
            right = f"{left.itag}*"
        return delta(left, right, options=options)
    if right is None:
        raise ValueError("identity_tensor requires a right itag when using raw-tag construction.")
    return delta(left, right, options=options)


def diag_tensor(
    vec: Iterable[complex] | torch.Tensor,
    left: Ix | str,
    right: Ix | str,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Build a diagonal Nicole tensor from explicit left and right legs.

    Args:
        vec: Diagonal values.
        left: Left index wrapper or raw left tag.
        right: Right index wrapper or raw right tag.
        dtype: Output tensor dtype.

    Returns:
        A rank-2 Nicole tensor whose dense form is ``diag(vec)``.
    """
    values = torch.as_tensor(tuple(vec) if not isinstance(vec, torch.Tensor) else vec, dtype=dtype)
    mat = torch.diag(values)
    if isinstance(left, Ix) and isinstance(right, Ix):
        return make_tensor(mat, [left, right], dtype=dtype)
    if isinstance(left, str) and isinstance(right, str):
        size = int(values.numel())
        return make_tensor(mat, [Ix(left, size, Direction.OUT), Ix(right, size, Direction.IN)], dtype=dtype)
    raise TypeError("diag_tensor requires either two Ix handles or two itag strings.")


def inv_diag(diag_tensor_obj: Tensor) -> Tensor:
    """Invert a diagonal Nicole tensor using Nicole's native helper.

    Args:
        diag_tensor_obj: Diagonal Nicole tensor.

    Returns:
        The Nicole inverse tensor.
    """
    return _ninv(diag_tensor_obj)
