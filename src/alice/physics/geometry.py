# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Alice library.
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

This module provides functions to generate interaction maps for 1D MPS
traversing 2D lattices.  Each builder returns a list of `Interaction2Site`
objects with `leading_site`, `terminal_site`, and `label` filled in.
Coupling constants (`cpl`) are left at their default (`0.0`) and are
assigned by the model builder in the second stage of the pipeline.

The public entry point for TOML-driven construction is `build_geometry`,
which dispatches on the `lattice` and `traverse` keys.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from alice.network.interaction import Interaction2Site

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_lattice_diagram(lx: int, ly: int, ord_map: List[List[int]]) -> None:
    """Log a visual diagram of the snake-like lattice traversal."""
    logger.info("=" * 60)
    logger.info("Traverse over 2D Lattice via Snake-like Chain".center(60))
    logger.info("=" * 60)
    logger.info("")

    # Each site: 2 chars; each connector: 5 chars.
    # Total width = lx * 2 + (lx - 1) * 5 = 7 * lx - 5.
    diagram_width = 7 * lx - 5
    left_padding  = max(0, (60 - diagram_width) // 2)
    padding       = " " * left_padding

    for row in range(ly):
        line = ""
        for col in range(lx):
            site = ord_map[row][col]
            if col < lx - 1:
                if (row == 0 and col % 2 == 1) or (row == ly - 1 and col % 2 == 0):
                    connector = "-----"
                else:
                    connector = ". . ."
            else:
                connector = ""
            line += f"{site:02d}{connector}"
        logger.info(padding + line)

        if row < ly - 1:
            line = "".join("|      " for _ in range(lx))
            logger.info(padding + line)

    logger.info("")


def _log_pairs(pairs: List[str], indent: int = 3, max_per_line: int = 6) -> None:
    """Log interaction pairs with automatic line wrapping."""
    indent_str = " " * indent
    for i in range(0, len(pairs), max_per_line):
        chunk = pairs[i:i + max_per_line]
        logger.info(indent_str + ", ".join(chunk))


# ---------------------------------------------------------------------------
# Traversal orders
# ---------------------------------------------------------------------------

def generate_snake_order(
    lx: int,
    ly: int,
) -> tuple[List[List[int]], List[tuple[int, int]]]:
    """Generate snake-like traversal order for a 2D square lattice.

    Creates a mapping between site indices and lattice coordinates for
    a snake-like path through the lattice::

        00. . .07-----08. . .15
        |      |      |      |
        01. . .06. . .09. . .14
        |      |      |      |
        02. . .05. . .10. . .13
        |      |      |      |
        03-----04. . .11-----12

    Parameters
    ----------
    lx:
        Number of columns.
    ly:
        Number of rows.

    Returns
    -------
    tuple
        `(ord_map, latt)` where `ord_map[row][col]` gives the site index
        (0-based) and `latt[site_idx]` gives the `(row, col)` tuple.
    """
    L = lx * ly

    ord_map = [[0] * lx for _ in range(ly)]
    for idx in range(L):
        row = idx % ly
        col = idx // ly
        ord_map[row][col] = idx

    # Reverse odd columns to create the snake pattern.
    for col in range(lx):
        if col % 2 == 1:
            for row in range(ly // 2):
                ord_map[row][col], ord_map[ly - 1 - row][col] = (
                    ord_map[ly - 1 - row][col], ord_map[row][col]
                )

    latt: List[tuple[int, int]] = [(0, 0)] * L
    for row in range(ly):
        for col in range(lx):
            latt[ord_map[row][col]] = (row, col)

    return ord_map, latt


# Registry of available traversal-order generators.
_TRAVERSALS: Dict[str, object] = {
    'snake': generate_snake_order,
}


# ---------------------------------------------------------------------------
# Lattice geometry builders
# ---------------------------------------------------------------------------

def intrcmap_1dchain(geo: dict, order_fn=None) -> List[Interaction2Site]:
    """Generate an interaction map for a 1D chain.

    Produces nearest-neighbor (NN) bonds along the chain and, when
    `bcx='PBC'`, a single periodic bond connecting the two ends.
    Coupling constants are not set; `cpl` is `0.0` on all returned objects.

    Parameters
    ----------
    geo:
        Geometry sub-dict from the TOML `[geometry]` section.  Expected
        keys:

        - `lx` — number of sites.
        - `bcx` — boundary condition (`'OBC'` or `'PBC'`).
        - `n2x` — include NN bonds (default `True`).
    order_fn:
        Accepted but ignored.  Present so the function can be stored in
        `_LATTICES` alongside 2D builders that receive a traversal function
        from `build_geometry`.

    Returns
    -------
    list[Interaction2Site]
        Interaction objects sorted by `leading_site`.  Tensor fields are
        `None`; `cpl` is `0.0`.
    """
    L   = geo['lx']
    bcx = geo.get('bcx', 'OBC').upper()
    n2x = bool(geo.get('n2x', True))

    ord_map, _ = generate_snake_order(L, 1)

    interactions: List[Interaction2Site] = []

    _log_lattice_diagram(L, 1, ord_map)
    logger.info("=" * 60)
    logger.info("Interactions Info".center(60))
    logger.info("=" * 60)
    logger.info("")

    if n2x:
        logger.info(" NN interaction (N2X):")
        pairs = []
        for si in range(L - 1):
            interactions.append(Interaction2Site(
                label=['NN', 'N2X'],
                leading_site=si,
                terminal_site=si + 1,
            ))
            pairs.append(f"({si:02d},{si+1:02d})")
        _log_pairs(pairs)

        if bcx == 'PBC':
            logger.info("")
            logger.info(" PBC interaction at X edge:")
            interactions.append(Interaction2Site(
                label=['NN', 'PBC', 'N2X'],
                leading_site=0,
                terminal_site=L - 1,
            ))
            _log_pairs([f"({0:02d},{L-1:02d})"])

    interactions.sort(key=lambda x: x.leading_site)

    logger.info("")
    logger.info(f"Total interactions: {len(interactions)}")

    return interactions


def intrcmap_square(geo: dict, order_fn=generate_snake_order) -> List[Interaction2Site]:
    """Generate an interaction map for a 2D square lattice.

    Produces nearest-neighbor (NN) and optionally next-nearest-neighbor
    (NNN) interactions for a 2D square lattice.  Coupling constants are not
    set here; the returned interactions have `cpl == 0.0` (the default).
    Labels encode bond topology so that the model builder can assign the
    correct coupling per bond type.

    Parameters
    ----------
    geo:
        Geometry sub-dict from the TOML `[geometry]` section.  Expected keys:

        - `lx` — number of columns.
        - `ly` — number of rows.
        - `bcx` — boundary condition along x (`'OBC'` or `'PBC'`).
        - `bcy` — boundary condition along y (`'OBC'` or `'PBC'`).
        - `n2x` — include NN bonds along x (default `True`).
        - `n2y` — include NN bonds along y (default `True`).
        - `n3d` — include NNN diagonal bonds (default `False`).
        - `n3o` — include NNN off-diagonal bonds (default `False`).
    order_fn:
        Traversal-order generator `(lx, ly) → (ord_map, latt)`.  Defaults
        to `generate_snake_order`; `build_geometry` supplies a different
        function when a non-default traversal is requested.

    Returns
    -------
    list[Interaction2Site]
        Interaction objects sorted by `leading_site`.  Tensor fields are
        `None`; `cpl` is `0.0`.
    """
    lx  = geo['lx']
    ly  = geo['ly']
    L   = lx * ly
    bcx = geo.get('bcx', 'OBC').upper()
    bcy = geo.get('bcy', 'OBC').upper()

    # Bond-inclusion flags.
    n2x = bool(geo.get('n2x', True))
    n2y = bool(geo.get('n2y', True))
    n3d = bool(geo.get('n3d', False))
    n3o = bool(geo.get('n3o', False))

    # === 1D CHAIN (ly == 1): delegate to the dedicated builder ===
    if ly == 1:
        return intrcmap_1dchain(geo)

    ord_map, _ = order_fn(lx, ly)

    interactions: List[Interaction2Site] = []

    _log_lattice_diagram(lx, ly, ord_map)
    logger.info("=" * 60)
    logger.info("Interactions Info".center(60))
    logger.info("=" * 60)
    logger.info("")

    if n2x:
        logger.info(" NN interaction along X axis:")
        pairs = []
        for row in range(ly):
            for col in range(lx - 1):
                a, b = ord_map[row][col], ord_map[row][col + 1]
                start, terminal = min(a, b), max(a, b)
                interactions.append(Interaction2Site(
                    label=['NN', 'N2X'],
                    leading_site=start,
                    terminal_site=terminal,
                ))
                pairs.append(f"({start:02d},{terminal:02d})")
        _log_pairs(pairs)

    if n2y:
        logger.info("")
        logger.info(" NN interaction along Y axis:")
        pairs = []
        for col in range(lx):
            for row in range(ly - 1):
                a, b = ord_map[row][col], ord_map[row + 1][col]
                start, terminal = min(a, b), max(a, b)
                interactions.append(Interaction2Site(
                    label=['NN', 'N2Y'],
                    leading_site=start,
                    terminal_site=terminal,
                ))
                pairs.append(f"({start:02d},{terminal:02d})")
        _log_pairs(pairs)

    # === PBC along X ===
    if bcx == 'PBC':
        logger.info("")
        logger.info(" PBC interaction at X edge:")
        pairs = []
        for row in range(ly):
            a, b = ord_map[row][0], ord_map[row][lx - 1]
            start, terminal = min(a, b), max(a, b)
            interactions.append(Interaction2Site(
                label=['NN', 'PBC', 'N2X'],
                leading_site=start,
                terminal_site=terminal,
            ))
            pairs.append(f"({start:02d},{terminal:02d})")
        _log_pairs(pairs)

    # === PBC along Y ===
    if bcy == 'PBC':
        logger.info("")
        logger.info(" PBC interaction at Y edge:")
        pairs = []
        for col in range(lx):
            a, b = ord_map[0][col], ord_map[ly - 1][col]
            start, terminal = min(a, b), max(a, b)
            interactions.append(Interaction2Site(
                label=['NN', 'PBC', 'N2Y'],
                leading_site=start,
                terminal_site=terminal,
            ))
            pairs.append(f"({start:02d},{terminal:02d})")
        _log_pairs(pairs)

    # === NNN off-diagonal (N3O): (row, col) ↔ (row-1, col+1) ===
    if n3o:
        logger.info("")
        logger.info(" NNN off-diagonal interaction (N3O):")
        pairs = []
        for col in range(lx - 1):
            for row in range(1, ly):
                a, b = ord_map[row][col], ord_map[row - 1][col + 1]
                start, terminal = min(a, b), max(a, b)
                interactions.append(Interaction2Site(
                    label=['NNN', 'N3O'],
                    leading_site=start,
                    terminal_site=terminal,
                ))
                pairs.append(f"({start:02d},{terminal:02d})")
        _log_pairs(pairs)

    # === NNN diagonal (N3D): (row, col) ↔ (row+1, col+1) ===
    if n3d:
        logger.info("")
        logger.info(" NNN diagonal interaction (N3D):")
        pairs = []
        for col in range(lx - 1):
            for row in range(ly - 1):
                a, b = ord_map[row][col], ord_map[row + 1][col + 1]
                start, terminal = min(a, b), max(a, b)
                interactions.append(Interaction2Site(
                    label=['NNN', 'N3D'],
                    leading_site=start,
                    terminal_site=terminal,
                ))
                pairs.append(f"({start:02d},{terminal:02d})")
        _log_pairs(pairs)

    # === NNN PBC along X ===
    if bcx == 'PBC':
        if n3o:
            logger.info("")
            logger.info(" NNN off-diagonal PBC interaction at X edge (N3O):")
            pairs = []
            for row in range(ly - 1):
                a, b = ord_map[row][0], ord_map[row + 1][lx - 1]
                start, terminal = min(a, b), max(a, b)
                interactions.append(Interaction2Site(
                    label=['NNN', 'PBC', 'N3O'],
                    leading_site=start,
                    terminal_site=terminal,
                ))
                pairs.append(f"({start:02d},{terminal:02d})")
            _log_pairs(pairs)

        if n3d:
            logger.info("")
            logger.info(" NNN diagonal PBC interaction at X edge (N3D):")
            pairs = []
            for row in range(1, ly):
                a, b = ord_map[row][0], ord_map[row - 1][lx - 1]
                start, terminal = min(a, b), max(a, b)
                interactions.append(Interaction2Site(
                    label=['NNN', 'PBC', 'N3D'],
                    leading_site=start,
                    terminal_site=terminal,
                ))
                pairs.append(f"({start:02d},{terminal:02d})")
            _log_pairs(pairs)

    # === NNN PBC along Y ===
    if bcy == 'PBC':
        if n3o:
            logger.info("")
            logger.info(" NNN off-diagonal PBC interaction at Y edge (N3O):")
            pairs = []
            for col in range(lx - 1):
                a, b = ord_map[0][col], ord_map[ly - 1][col + 1]
                start, terminal = min(a, b), max(a, b)
                interactions.append(Interaction2Site(
                    label=['NNN', 'PBC', 'N3O'],
                    leading_site=start,
                    terminal_site=terminal,
                ))
                pairs.append(f"({start:02d},{terminal:02d})")
            _log_pairs(pairs)

        if n3d:
            logger.info("")
            logger.info(" NNN diagonal PBC interaction at Y edge (N3D):")
            pairs = []
            for col in range(1, lx):
                a, b = ord_map[ly - 1][col - 1], ord_map[0][col]
                start, terminal = min(a, b), max(a, b)
                interactions.append(Interaction2Site(
                    label=['NNN', 'PBC', 'N3D'],
                    leading_site=start,
                    terminal_site=terminal,
                ))
                pairs.append(f"({start:02d},{terminal:02d})")
            _log_pairs(pairs)

    interactions.sort(key=lambda x: x.leading_site)

    logger.info("")
    logger.info(f"Total interactions: {len(interactions)}")

    return interactions


# Registry of available lattice builders.
_LATTICES: Dict[str, object] = {
    'chain': intrcmap_1dchain,
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
