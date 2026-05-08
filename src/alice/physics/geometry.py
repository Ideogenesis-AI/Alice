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
`build_intrcmap` (interaction-list dispatcher), and the two dispatch
registries `_TRAVERSALS` and `_LATTICES`.

Naming convention used throughout:

- `geo_cfg` — raw `dict` from the TOML `[geometry]` section.
- `geo` — a `Geometry` instance.

To add a new lattice type, create a module under `alice.physics`, implement
its builder (signature `(geo: Geometry) → list[Interaction2Site]`), import it
here, and add it to `_LATTICES`.  To add a new traversal mode, implement the
generator (signature `(lx, ly) → (ord_map, latt)`) in the appropriate module,
import it here, and add it to `_TRAVERSALS`.

1D chain geometry (`intrcmap_1dchain`) lives in `alice.physics.chain`.
Square-lattice geometry (traversal orders and `intrcmap_square`) lives in
`alice.physics.square`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from alice.network.interaction import Interaction2Site
from alice.physics.chain import intrcmap_1dchain
from alice.physics.square import (
    generate_snake_order,
    generate_zigzag_order,
    intrcmap_square,
)

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
    lattice:
        Lattice-type key, e.g. `'chain'` or `'square'`.
    traverse:
        Traversal-order key, e.g. `'snake'` or `'zigzag'`.
    cfg:
        Original geometry config dict (`[geometry]` section from TOML).
    ord_map:
        2D list where `ord_map[row][col]` gives the 1D site index (0-based).
    latt:
        List where `latt[site]` gives the `(row, col)` lattice coordinate.
    """

    lattice:  str
    traverse: str
    cfg:      dict
    ord_map:  List[List[int]]
    latt:     List[Tuple[int, int]]

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
        """Total number of sites (`lx * ly`)."""
        return self.lx * self.ly

    def to_1d(self, row: int, col: int) -> int:
        """Convert a 2D lattice coordinate to a 1D site index."""
        return self.ord_map[row][col]

    def to_2d(self, site: int) -> Tuple[int, int]:
        """Convert a 1D site index to a 2D lattice coordinate."""
        return self.latt[site]


# ---------------------------------------------------------------------------
# Dispatch registries
# ---------------------------------------------------------------------------

# Map traverse key → (lx, ly) → (ord_map, latt).
# To add a new traversal mode: import the generator and add it here.
_TRAVERSALS: Dict[str, object] = {
    'snake':  generate_snake_order,
    'zigzag': generate_zigzag_order,
}

# Map lattice key → (geo: Geometry) → list[Interaction2Site].
# To add a new lattice type: import the builder and add it here.
_LATTICES: Dict[str, object] = {
    'chain':  intrcmap_1dchain,
    'square': intrcmap_square,
}


# ---------------------------------------------------------------------------
# Public dispatchers
# ---------------------------------------------------------------------------

def build_geometry(geo_cfg: dict) -> Geometry:
    """Construct a `Geometry` from a `[geometry]` config dict.

    Validates the `lattice` and `traverse` keys, calls the traversal-order
    generator to produce `ord_map` and `latt`, then returns a fully-populated
    `Geometry` dataclass.

    Parameters
    ----------
    geo_cfg:
        Geometry sub-dict from the TOML `[geometry]` section.  Must contain
        `lx` and optionally `ly` (default `1`), `lattice` (default
        `'square'`), and `traverse` (default `'snake'`).

    Returns
    -------
    Geometry
        Fully-resolved geometry struct ready for passing to `build_intrcmap`
        or any `intrcmap_*` builder.

    Raises
    ------
    ValueError
        If `lattice` or `traverse` names an unrecognised option.
    """
    lattice  = geo_cfg.get('lattice', 'square')
    traverse = geo_cfg.get('traverse', 'snake')

    if traverse not in _TRAVERSALS:
        raise ValueError(
            f"Unknown traversal order '{traverse}'. "
            f"Available: {list(_TRAVERSALS)}"
        )
    if lattice not in _LATTICES:
        raise ValueError(
            f"Unknown lattice type '{lattice}'. "
            f"Available: {list(_LATTICES)}"
        )

    lx = geo_cfg['lx']
    ly = geo_cfg.get('ly', 1)
    order_fn = _TRAVERSALS[traverse]
    ord_map, latt = order_fn(lx, ly)

    return Geometry(lattice, traverse, geo_cfg, ord_map, latt)


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
    lattice_fn = _LATTICES[geo.lattice]
    return lattice_fn(geo)
