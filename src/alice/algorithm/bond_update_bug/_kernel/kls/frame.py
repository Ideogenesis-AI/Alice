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

"""LocalBondFrame and basic tensor/index utilities for KLS updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from nicole import Tensor, einsum

from ..indices import Ix, has_nontrivial_symmetry, resolved_sectors


@dataclass(frozen=True)
class LocalBondFrame:
    """Canonical two-site data needed by one local KLS update.

    Args:
        link_l: Left bond index entering the active two-site block.
        link_mid: Bond index between the active left and right sites.
        link_r: Right bond index exiting the active two-site block.
        site_l: Left physical site index.
        site_r: Right physical site index.
        U0_tens: Left canonical isometry.
        V0_tens: Right canonical isometry.
        S0_tens: Bond-center tensor between the canonical frames.
        canon_u0: Middle index carried by ``U0_tens`` and ``S0_tens``.
        canon_v0: Middle index carried by ``S0_tens`` and ``V0_tens``.
        theta0_tens: Optional assembled two-site tensor.
    """

    link_l: Ix
    link_mid: Ix
    link_r: Ix
    site_l: Ix
    site_r: Ix
    U0_tens: Tensor
    V0_tens: Tensor
    S0_tens: Tensor
    canon_u0: Ix
    canon_v0: Ix
    theta0_tens: Tensor | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LocalBondFrame":
        """Create a frame object from the historical snapshot dictionary.

        Args:
            data: Snapshot dictionary produced by the bug or environment code.

        Returns:
            A :class:`LocalBondFrame` instance with the expected fields.
        """
        return cls(
            link_l=data["link_l"],
            link_mid=data["link_mid"],
            link_r=data["link_r"],
            site_l=data["site_l"],
            site_r=data["site_r"],
            U0_tens=data["U0_tens"],
            V0_tens=data["V0_tens"],
            S0_tens=data["S0_tens"],
            canon_u0=data["canon_u0"],
            canon_v0=data["canon_v0"],
            theta0_tens=data.get("theta0_tens"),
        )

    @property
    def old_rank(self) -> int:
        """Return the current middle-bond rank."""
        return int(self.link_mid.dim)

    @property
    def left_capacity(self) -> int:
        """Return the maximum admissible left-frame rank ``dim(link_l)*dim(site_l)``."""
        return int(self.link_l.dim * self.site_l.dim)

    @property
    def right_capacity(self) -> int:
        """Return the maximum admissible right-frame rank ``dim(site_r)*dim(link_r)``."""
        return int(self.site_r.dim * self.link_r.dim)

    def has_symmetry(self) -> bool:
        """Return ``True`` when any leg on the active bond carries nontrivial symmetry."""
        return has_nontrivial_symmetry([self.link_l, self.link_mid, self.link_r, self.site_l, self.site_r])


def _clone_tensor_with_ixs(tensor: Tensor, ixs: list[Ix]) -> Tensor:
    indices = tuple(ix.nicole() for ix in ixs)
    itags = tuple(ix.itag for ix in ixs)
    data = {tuple(key): block.clone() for key, block in tensor.data.items()}
    intw = None if tensor.intw is None else {tuple(key): bridge.clone() for key, bridge in tensor.intw.items()}
    return Tensor(indices=indices, itags=itags, data=data, intw=intw, dtype=tensor.dtype, label=tensor.label)


def _tensor_ix(tensor: Tensor, axis: int) -> Ix:
    idx = tensor.indices[axis]
    return Ix(tensor.itags[axis], int(idx.dim), idx.direction, idx.sectors, idx.group)


def _sector_offsets(ix: Ix) -> dict[object, tuple[int, int]]:
    offsets: dict[object, tuple[int, int]] = {}
    cursor = 0
    for sector in resolved_sectors(ix):
        offsets[sector.charge] = (cursor, sector.dim)
        cursor += sector.dim
    return offsets


def _dense_from_tensor_with_ixs(tensor: Tensor, ixs: list[Ix]) -> torch.Tensor:
    shape = tuple(ix.dim for ix in ixs)
    device = next(iter(tensor.data.values())).device if tensor.data else torch.device("cpu")
    dense = torch.zeros(shape, dtype=tensor.dtype, device=device)
    offsets = [_sector_offsets(ix) for ix in ixs]
    for key, block in tensor.data.items():
        slices = tuple(slice(offsets[axis][key[axis]][0], offsets[axis][key[axis]][0] + offsets[axis][key[axis]][1]) for axis in range(len(key)))
        dense[slices] = block
    return dense


def _left_row_indices_by_flux(link_l: Ix, site_l: Ix) -> dict[object, list[int]]:
    rows: dict[object, list[int]] = {}
    link_offsets = _sector_offsets(link_l)
    site_offsets = _sector_offsets(site_l)
    dl = int(link_l.dim)
    d_link = int(link_l.direction)
    d_site = int(site_l.direction)

    for q_link, (link_start, link_dim) in link_offsets.items():
        for q_site, (site_start, site_dim) in site_offsets.items():
            flux = -(d_link * q_link + d_site * q_site)
            block_rows = rows.setdefault(flux, [])
            for site_local in range(site_dim):
                for link_local in range(link_dim):
                    block_rows.append((link_start + link_local) + dl * (site_start + site_local))
    return rows


def _right_col_indices_by_flux(site_r: Ix, link_r: Ix) -> dict[object, list[int]]:
    cols: dict[object, list[int]] = {}
    site_offsets = _sector_offsets(site_r)
    link_offsets = _sector_offsets(link_r)
    ds = int(site_r.dim)
    d_site = int(site_r.direction)
    d_link = int(link_r.direction)

    for q_site, (site_start, site_dim) in site_offsets.items():
        for q_link, (link_start, link_dim) in link_offsets.items():
            flux = -(d_site * q_site + d_link * q_link)
            block_cols = cols.setdefault(flux, [])
            for link_local in range(link_dim):
                for site_local in range(site_dim):
                    block_cols.append((site_start + site_local) + ds * (link_start + link_local))
    return cols


def _apply_gate_named(gate, theta, site_l_tag: str, site_r_tag: str):
    out = einsum("LRlr,aLRb->alrb", gate, theta)
    out.retag({f"{site_l_tag}*": site_l_tag, f"{site_r_tag}*": site_r_tag})
    return out


def _singular_values_from_diag_tensor(S) -> torch.Tensor:
    """Extract singular values from a diagonal Nicole tensor.

    Args:
        S: Diagonal tensor returned by Nicole's SVD.

    Returns:
        A flat torch tensor containing every block-diagonal entry.
    """
    vals = [torch.diagonal(block) for block in S.data.values()]
    if not vals:
        return torch.empty((0,), dtype=torch.complex128)
    return torch.cat(vals)
