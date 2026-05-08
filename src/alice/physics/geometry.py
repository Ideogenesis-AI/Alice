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


"""Lattice geometry builders for MPS interaction maps.

This module provides the public dispatcher `build_geometry` and owns the two
dispatch registries `_TRAVERSALS` and `_LATTICES`.

To add a new lattice type, create a module under `alice.physics`, implement
its builder (signature `(geo, order_fn) → list[Interaction2Site]`), import
it here, and add it to `_LATTICES`.  To add a new traversal mode, implement
the generator (signature `(lx, ly) → (ord_map, latt)`) in the appropriate
module, import it here, and add it to `_TRAVERSALS`.

1D chain geometry (traversal order and `intrcmap_1dchain`) lives in
`alice.physics.chain`.  Square-lattice geometry (traversal orders and
`intrcmap_square`) lives in `alice.physics.square`.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from alice.network.interaction import Interaction2Site
from alice.physics.chain import intrcmap_1dchain
from alice.physics.square import (
    generate_snake_order,
    generate_zigzag_order,
    intrcmap_square,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dispatch registries
# ---------------------------------------------------------------------------

# Map traverse key → (lx, ly) → (ord_map, latt).
# To add a new traversal mode: import the generator and add it here.
_TRAVERSALS: Dict[str, object] = {
    'snake':  generate_snake_order,
    'zigzag': generate_zigzag_order,
}

# Map lattice key → (geo, order_fn) → list[Interaction2Site].
# To add a new lattice type: import the builder and add it here.
_LATTICES: Dict[str, object] = {
    'chain':  intrcmap_1dchain,
    'square': intrcmap_square,
}


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def build_geometry(geo: dict) -> List[Interaction2Site]:
    """Dispatch geometry construction from a `[geometry]` config dict.

    Reads the `lattice` and `traverse` keys to select the lattice builder
    and traversal-order generator, then delegates to the builder.

    Parameters
    ----------
    geo:
        Geometry sub-dict from the TOML `[geometry]` section.  Must contain
        `lattice` (e.g. `'square'`) and optionally `traverse` (default
        `'snake'`).

    Returns
    -------
    list[Interaction2Site]
        Interaction objects sorted by `leading_site`, with `cpl == 0.0`
        and tensor fields set to `None`.

    Raises
    ------
    ValueError
        If `lattice` or `traverse` names an unrecognised option.
    """
    lattice  = geo.get('lattice', 'square')
    traverse = geo.get('traverse', 'snake')

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

    order_fn   = _TRAVERSALS[traverse]
    lattice_fn = _LATTICES[lattice]
    return lattice_fn(geo, order_fn)
