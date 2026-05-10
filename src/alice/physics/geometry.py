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


"""Lattice geometry: the `Geometry` dataclass and public dispatchers.

This module owns the `Geometry` dataclass, `build_geometry` (struct factory),
`build_intrcmap` (interaction-list dispatcher), and the `_LATTICES` registry.

Naming convention used throughout:

- `geo_cfg` — raw `dict` from the TOML `[geometry]` section.
- `geo` — a `Geometry` instance.

To add a new lattice type, create a module under `alice.physics` that
implements:

- `build_traversal(geo_cfg: dict) → (ord_map, latt)` — traversal selection
  and validation for that lattice.
- An `intrcmap_*` builder with signature `(geo: Geometry) → list[Interaction2Site]`.

Add the `intrcmap_*` builder to `_LATTICES` and add a branch to the
conditional import in `build_geometry`.

Built-in lattice modules:

- `alice.physics.chain` — 1D chain (`build_traversal`, `intrcmap_1dchain`).
- `alice.physics.square` — square lattice (`build_traversal`, `intrcmap_square`).
- `alice.physics.kagome` — Kagome lattice (`build_traversal`, `intrcmap_kagome`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from alice.network.interaction import Interaction2Site
from alice.physics.chain import intrcmap_1dchain
from alice.physics.kagome import intrcmap_kagome
from alice.physics.square import intrcmap_square

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry dataclass
# ---------------------------------------------------------------------------

@dataclass
class Geometry:
    """Fully-resolved lattice geometry for one MPS simulation.

    Constructed by `build_geometry` from a raw `[geometry]` config dict.
    Passed to `intrcmap_*` builders and to traversal-aware utilities such
    as `to_1d` / `to_2d`.

    Attributes
    ----------
    cfg:
        Original geometry config dict (`[geometry]` section from TOML).
        Contains all geometry parameters including `lattice`, `traverse`,
        `lx`, `ly`, boundary conditions, and bond-inclusion flags.
    ord_map:
        Dict mapping coordinate tuples to 1D site indices (0-based).
        Key length depends on the lattice: `(row, col)` for chain/square,
        `(row, col, u)` for multi-sublattice lattices such as Kagome.
    latt:
        List where `latt[site]` gives the coordinate tuple for that site.
        Element type matches the key type of `ord_map`.
    """

    cfg:     dict
    ord_map: dict[tuple[int, ...], int]
    latt:    list[tuple[int, ...]]

    @property
    def lattice(self) -> str:
        """Lattice-type key, e.g. `'chain'` or `'square'`."""
        return self.cfg.get('lattice', 'square')

    @property
    def traverse(self) -> Optional[str]:
        """Traversal-order key. `None` when not set."""
        return self.cfg.get('traverse')

    @property
    def lx(self) -> int:
        """Number of columns (sites along x)."""
        return self.cfg['lx']

    @property
    def ly(self) -> int:
        """Number of rows (sites along y); `1` for a 1D chain."""
        return self.cfg.get('ly', 1)

    @property
    def L(self) -> int:
        """Total number of MPS/MPO sites (length of the tensor network).

        Equals `len(latt)`, which is the authoritative site count for all
        lattice types. For single-sublattice lattices (chain, square) this
        coincides with `lx * ly`. For multi-sublattice lattices (e.g.
        Kagome with 3 sites per unit cell) it equals `lx * ly * n_sub`.
        """
        return len(self.latt)

    def to_1d(self, coord: tuple[int, ...]) -> int:
        """Convert a lattice coordinate tuple to a 1D site index."""
        return self.ord_map[coord]

    def to_2d(self, site: int) -> tuple[int, ...]:
        """Convert a 1D site index to a lattice coordinate tuple."""
        return self.latt[site]


# ---------------------------------------------------------------------------
# Dispatch registries
# ---------------------------------------------------------------------------

# Map lattice key → (geo: Geometry) → list[Interaction2Site].
# To add a new lattice type: import the builder and add it here.
_LATTICES: Dict[str, object] = {
    'chain':  intrcmap_1dchain,
    'square': intrcmap_square,
    'kagome': intrcmap_kagome,
}


# ---------------------------------------------------------------------------
# Public dispatchers
# ---------------------------------------------------------------------------

def build_geometry(geo_cfg: dict) -> Geometry:
    """Construct a `Geometry` from a `[geometry]` config dict.

    Validates the `lattice` key, then delegates traversal selection and
    validation to the lattice-specific `build_traversal` dispatcher, then
    returns a fully-populated `Geometry` dataclass.

    Parameters
    ----------
    geo_cfg:
        Geometry sub-dict from the TOML `[geometry]` section. Must contain
        `lx` and optionally `ly` (default `1`), `lattice` (default
        `'square'`), and `traverse` (default `'serpentine'`, used by 2D
        lattices).

    Returns
    -------
    Geometry
        Fully-resolved geometry struct ready for passing to `build_intrcmap`
        or any `intrcmap_*` builder.

    Raises
    ------
    ValueError
        If `lattice` names an unrecognised option, or if `traverse` names
        an option not supported by the selected lattice module.
    """
    if geo_cfg.get('lattice', 'chain') not in _LATTICES:
        raise ValueError(
            f"Unknown lattice type '{geo_cfg.get('lattice', 'chain')}'. "
            f"Available: {list(_LATTICES.keys())}"
        )

    match geo_cfg.get('lattice', 'chain'):
        case 'chain':
            from alice.physics.chain import build_traversal   # noqa: PLC0415
        case 'square':
            from alice.physics.square import build_traversal  # noqa: PLC0415
        case 'kagome':
            from alice.physics.kagome import build_traversal  # noqa: PLC0415

    ord_map, latt = build_traversal(geo_cfg)

    return Geometry(geo_cfg, ord_map, latt)


def build_intrcmap(geo: Geometry) -> List[Interaction2Site]:
    """Generate an interaction map from a `Geometry` instance.

    Looks up the lattice builder registered for `geo.lattice` and delegates
    to it.

    Parameters
    ----------
    geo:
        Fully-resolved geometry struct, typically produced by `build_geometry`.

    Returns
    -------
    list[Interaction2Site]
        Interaction objects sorted by `leading_site`, with `cpl == 0.0`
        and tensor fields set to `None`.
    """
    # Lookup the lattice builder registered for `geo.lattice`
    lattice_fn = _LATTICES[geo.lattice]
    # Delegate to the lattice builder
    return lattice_fn(geo)
