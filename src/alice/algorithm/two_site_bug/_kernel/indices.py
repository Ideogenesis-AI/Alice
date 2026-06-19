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

"""Index builders and symmetry helpers used throughout ``bug_nicole``.

The Julia code this package was ported from leans heavily on lightweight index
wrappers and symmetry-aware site constructors. This module keeps that role, but
spells the ideas out in plain Python so the rest of the code can use readable
helpers instead of manipulating Nicole indices directly at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Iterable, Literal

from nicole import Direction, Index, Sector, U1Group

SymmetryName = Literal["trivial", "u1"]

_GROUP = U1Group()
_FRESH = count()
_SPIN_HALF_U1_SECTORS = (Sector(1, 1), Sector(-1, 1))

__all__ = [
    "Ix",
    "SymmetryName",
    "bond_index",
    "fresh_itag",
    "has_nontrivial_symmetry",
    "idx",
    "is_trivial_ix",
    "normalize_symmetry",
    "resolved_sectors",
    "siteinds",
    "spin_half_site_sectors",
]


def normalize_symmetry(symmetry: str) -> SymmetryName:
    """Normalize public symmetry spellings to the package-internal name.

    Args:
        symmetry: User-facing symmetry label such as ``"u1"``, ``"sz"``,
            ``"trivial"`` or ``"dense"``.

    Returns:
        ``"trivial"`` or ``"u1"``.
    """
    key = symmetry.strip().lower()
    if key in {"trivial", "none", "dense"}:
        return "trivial"
    if key in {"u1", "sz", "u(1)"}:
        return "u1"
    raise ValueError(f"Unknown symmetry specification: {symmetry!r}")


def spin_half_site_sectors(symmetry: str = "u1") -> tuple[Sector, ...]:
    """Return the canonical spin-1/2 site sectors for one symmetry choice.

    Args:
        symmetry: Symmetry label understood by :func:`normalize_symmetry`.

    Returns:
        The sector tuple used for one spin-1/2 physical site.
    """
    key = normalize_symmetry(symmetry)
    if key == "trivial":
        return (Sector(0, 2),)
    return _SPIN_HALF_U1_SECTORS


@dataclass(frozen=True)
class Ix:
    """Lightweight Nicole-index handle used throughout the package.

    Args:
        itag: Nicole tag.
        dim: Total index dimension.
        direction: Nicole direction carried by the leg.
        sectors: Optional explicit sector tuple. ``None`` means one trivial
            dense sector of size ``dim``.
        group: Nicole symmetry group object. The default is the package U(1)
            group handle.
    """

    itag: str
    dim: int
    direction: Direction
    sectors: tuple[Sector, ...] | None = None
    group: object = _GROUP

    def nicole(self) -> Index:
        """Materialize this wrapper as a Nicole :class:`Index`.

        Returns:
            A Nicole index with the same tag metadata and sectors.
        """
        return Index(self.direction, self.group, resolved_sectors(self))

    def resolved_sectors(self) -> tuple[Sector, ...]:
        """Return the explicit sector tuple for this index.

        Returns:
            The stored sectors, or a single trivial sector when the index is
            dense.
        """
        return resolved_sectors(self)

    def reversed(self) -> "Ix":
        """Return a copy whose Nicole direction is reversed.

        Returns:
            A new :class:`Ix` with the same metadata and opposite direction.
        """
        return Ix(
            self.itag,
            self.dim,
            self.direction.reverse(),
            self.sectors,
            self.group,
        )

    def retag(self, itag: str) -> "Ix":
        """Return a copy with a different Nicole tag.

        Args:
            itag: Replacement tag.

        Returns:
            A new :class:`Ix` with the requested tag.
        """
        return Ix(itag, self.dim, self.direction, self.sectors, self.group)

    def is_trivial(self) -> bool:
        """Return whether this index is one dense neutral sector.

        Returns:
            ``True`` when the index has only the neutral dense sector.
        """
        return is_trivial_ix(self)


def resolved_sectors(ix: Ix | Index) -> tuple[Sector, ...]:
    """Return the explicit sector tuple for an ``Ix`` or Nicole ``Index``.

    Args:
        ix: Wrapped or native Nicole index.

    Returns:
        An explicit tuple of Nicole sectors.
    """
    if isinstance(ix, Ix):
        return ix.sectors if ix.sectors is not None else (Sector(0, ix.dim),)
    return ix.sectors


def is_trivial_ix(ix: Ix | Index) -> bool:
    """Return whether an index carries only the neutral dense sector.

    Args:
        ix: Wrapped or native Nicole index.

    Returns:
        ``True`` when the sector structure is trivial.
    """
    sectors = resolved_sectors(ix)
    return len(sectors) == 1 and sectors[0].charge == 0


def has_nontrivial_symmetry(ixs: Iterable[Ix | Index]) -> bool:
    """Return whether any index in a collection carries charge structure.

    Args:
        ixs: Iterable of wrapped or native Nicole indices.

    Returns:
        ``True`` when at least one index is not dense-trivial.
    """
    return any(not is_trivial_ix(ix) for ix in ixs)


def idx(direction: Direction, dim: int, itag: str, **kwargs: object) -> Ix:
    """Construct an :class:`Ix` using the field order most call sites prefer.

    Args:
        direction: Nicole direction for the index.
        dim: Total index dimension.
        itag: Nicole tag string.
        **kwargs: Optional ``sectors=...`` and ``group=...`` overrides.

    Returns:
        A new :class:`Ix` wrapper.
    """

    sectors = kwargs.pop("sectors", None)
    group = kwargs.pop("group", _GROUP)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown idx option(s): {unknown}")
    return Ix(itag, dim, direction, sectors, group)


def bond_index(
    itag: str,
    direction: Direction,
    charge_dims: Iterable[tuple[int, int]],
    *,
    group: object = _GROUP,
) -> Ix:
    """Build a bond index from explicit ``(charge, multiplicity)`` data.

    Args:
        itag: Nicole tag string.
        direction: Nicole direction for the bond.
        charge_dims: Iterable of ``(charge, dim)`` pairs.
        group: Nicole symmetry group handle.

    Returns:
        A symmetry-aware :class:`Ix` wrapper for the bond.
    """
    sectors = tuple(Sector(int(charge), int(dim)) for charge, dim in charge_dims)
    return Ix(itag, sum(int(sector.dim) for sector in sectors), direction, sectors, group)


def fresh_itag(base: str) -> str:
    """Generate a unique Nicole tag with a monotone suffix.

    Args:
        base: Prefix that should remain recognizable in debug output.

    Returns:
        A fresh tag such as ``"b3#17"``.
    """
    return f"{base}#{next(_FRESH)}"


def siteinds(n: int, d: int = 2, *, symmetry: str = "trivial") -> list[Ix]:
    """Build the canonical physical site indices ``s1, s2, ..., sN``.

    Args:
        n: Number of sites.
        d: On-site Hilbert-space dimension.
        symmetry: Symmetry label such as ``"trivial"`` or ``"u1"``.

    Returns:
        A list of OUT-directed physical site indices.
    """
    key = normalize_symmetry(symmetry)
    if key == "u1":
        if d != 2:
            raise NotImplementedError("U(1) site indices currently support only spin-1/2 sites.")
        sectors = spin_half_site_sectors("u1")
    else:
        sectors = (Sector(0, d),)

    # Physical site legs always point outward in the MPS/MPO conventions used
    # throughout this port.
    return [Ix(f"s{k}", d, Direction.OUT, sectors) for k in range(1, n + 1)]
