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


"""Complex-dtype promotion for Nicole tensors.

Real-time evolution needs complex amplitudes, while the rest of Alice (and
every Hamiltonian MPO built by `build_hamiltonian`) is real by default.
Nicole contracts only operands that share a single dtype, so a complex state
forces its Hamiltonian, its environment blocks, and every boundary tensor to
be complex as well; `to_complex` is the single place that conversion happens.

`Tensor.to` is not usable for this: it changes device only, deriving the
dtype from device capability rather than from a caller argument. Scalar
multiplication by `1 + 0j` promotes the block data but leaves the Bridge
intertwiners of a generic (non-Abelian) tensor real, which fails at the next
contraction. This module therefore rebuilds the tensor, casting the block
data and the intertwiner weights together.
"""

from __future__ import annotations

import torch
from nicole import Tensor


def to_complex(t: Tensor, dtype: torch.dtype = torch.complex128) -> Tensor:
    """Promote a tensor to a complex dtype, leaving a complex one untouched.

    Parameters
    ----------
    t:
        Tensor to promote. Returned unchanged when already complex, which
        makes the call idempotent and cheap to apply defensively.
    dtype:
        Target complex dtype, `torch.complex128` by default.

    Returns
    -------
    Tensor
        Tensor with complex block data and, for generic symmetry groups,
        complex Bridge intertwiner weights.

    Raises
    ------
    ValueError
        If `dtype` is not a complex dtype.
    """
    if not dtype.is_complex:
        raise ValueError(f"to_complex requires a complex dtype, got {dtype}")
    if t.dtype.is_complex:
        return t

    data = {k: v.to(dtype) for k, v in t.data.items()}
    # Intertwiner weights hold Clebsch-Gordan coefficients, which are real,
    # but Nicole contracts them against the block data directly and so needs
    # them at a matching dtype.
    intw = None
    if t.intw is not None:
        intw = {k: b.to(b.device, dtype=dtype) for k, b in t.intw.items()}

    return Tensor(
        indices=t.indices,
        itags=t.itags,
        data=data,
        intw=intw,
        dtype=dtype,
        label=t.label,
    )
