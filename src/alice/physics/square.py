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


"""Square-lattice geometry: traversal orders and interaction map.

This module provides traversal-order generators (`generate_sequential_order`,
`generate_serpentine_order`) and the `intrcmap_square` geometry builder. The
builder returns a list of `Interaction2Site` objects with `leading_site`,
`terminal_site`, and `label` filled in. Coupling constants (`cpl`) are left
at their default (`0.0`) and are assigned by the model builder in the second
stage of the pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List

from alice.network.interaction import Interaction2Site

if TYPE_CHECKING:
    from alice.physics.geometry import Geometry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_DIAG_THRESHOLD = 8   # lx > this → truncate the diagram
_DIAG_HEAD      = 4   # columns shown at the left in truncated mode
_DIAG_TAIL      = 2   # columns shown at the right in truncated mode


def _log_2d_diagram(
    lx: int,
    ly: int,
    ord_map: dict[tuple[int, int], int],
    title: str,
    connector: Callable[[int, int], str],
) -> None:
    """Shared renderer for 2D traversal diagrams.

    `connector(row, col)` returns the horizontal connector string between
    column `col` and `col+1` at the given row (`"-----"` for on-path,
    `". . ."` for off-path).
    """
    logger.info("─" * 60)
    logger.info(title.center(60))
    logger.info("─" * 60)
    logger.info("")

    if lx <= _DIAG_THRESHOLD:
        # Full render: all columns shown.
        # Each site: 2 chars; each connector: 5 chars. Width = 7 * lx - 5.
        diagram_width = 7 * lx - 5
        padding       = " " * max(0, (60 - diagram_width) // 2)

        for row in range(ly):
            line = "".join(
                f"{ord_map[(row, col)]:02d}" + (connector(row, col) if col < lx - 1 else "")
                for col in range(lx)
            )
            logger.info(padding + line)

            if row < ly - 1:
                logger.info(padding + "".join("|      " for _ in range(lx)))

    else:
        # Truncated render: first _DIAG_HEAD columns + last _DIAG_TAIL columns.
        # Gap token "  ⋯ ⋯  " (8 chars) replaces the hidden interior.
        head_cols  = list(range(_DIAG_HEAD))
        tail_cols  = list(range(lx - _DIAG_TAIL, lx))
        gap = "  ⋯ ⋯  "

        # Width of the visible portion:
        #   head: _DIAG_HEAD * 2 + (_DIAG_HEAD - 1) * 5 = 7 * _DIAG_HEAD - 5
        #   gap:  len(gap)
        #   tail: _DIAG_TAIL * 2 + (_DIAG_TAIL - 1) * 5 = 7 * _DIAG_TAIL - 5
        diagram_width = (7 * _DIAG_HEAD - 5) + len(gap) + (7 * _DIAG_TAIL - 5)
        padding       = " " * max(0, (60 - diagram_width) // 2)

        for row in range(ly):
            head_str = "".join(
                f"{ord_map[(row, col)]:02d}" + (connector(row, col) if col < head_cols[-1] else "")
                for col in head_cols
            )
            tail_str = "".join(
                f"{ord_map[(row, col)]:02d}" + (connector(row, col) if col < tail_cols[-1] else "")
                for col in tail_cols
            )
            logger.info(padding + head_str + gap + tail_str)

            if row < ly - 1:
                # Build the inter-row vline with the same total width as a data
                # row so that each "|" sits directly under its column's site.
                vline = [" "] * diagram_width
                for i in range(_DIAG_HEAD):
                    vline[i * 7] = "|"
                tail_offset = 7 * _DIAG_HEAD - 5 + len(gap)
                for i in range(_DIAG_TAIL):
                    vline[tail_offset + i * 7] = "|"
                logger.info(padding + "".join(vline))

    logger.info("")


def _log_sequential_diagram(lx: int, ly: int, ord_map: List[List[int]]) -> None:
    """Log a visual diagram of the sequential-order lattice traversal."""
    # All horizontal connectors are off-path: the inter-column MPS jump
    # goes from the bottom of col c to the top of col c+1, spanning rows.
    _log_2d_diagram(lx, ly, ord_map, "Traverse over 2D Lattice via Sequential Chain",
                    lambda row, col: ". . .")


def _log_serpentine_diagram(lx: int, ly: int, ord_map: List[List[int]]) -> None:
    """Log a visual diagram of the serpentine lattice traversal."""
    def _connector(row: int, col: int) -> str:
        """Return the horizontal connector between col and col+1 at the given row."""
        if (row == 0 and col % 2 == 1) or (row == ly - 1 and col % 2 == 0):
            return "-----"
        return ". . ."
    _log_2d_diagram(lx, ly, ord_map, "Traverse over 2D Lattice via Serpentine Chain", _connector)


def _log_pairs(pairs: List[str], indent: int = 3, max_per_line: int = 6) -> None:
    """Log interaction pairs with automatic line wrapping."""
    indent_str = " " * indent
    for i in range(0, len(pairs), max_per_line):
        chunk = pairs[i:i + max_per_line]
        logger.info(indent_str + ", ".join(chunk))


# ---------------------------------------------------------------------------
# Traversal builders
# ---------------------------------------------------------------------------

def build_traversal_sequential(
    lx: int,
    ly: int,
) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    """Build sequential traversal order for a 2D square lattice.

    Creates a column-major mapping where every column goes top→bottom —
    no column reversals, unlike `build_traversal_serpentine` which reverses odd
    columns:

        00. . .04. . .08. . .12
        |      |      |      |
        01. . .05. . .09. . .13
        |      |      |      |
        02. . .06. . .10. . .14
        |      |      |      |
        03. . .07. . .11. . .15

    The MPS jump between columns goes from the bottom of col `c`
    (site `(c+1)*ly - 1`) to the top of col `c+1` (site `(c+1)*ly`).
    These sites are adjacent in the MPS but span different lattice rows,
    so no `"-----"` appears in the diagram.

    For `ly=1` the result is identical to `build_traversal_serpentine`.

    Logs a visual diagram of the traversal at INFO level.

    Parameters
    ----------
    lx:
        Number of columns.
    ly:
        Number of rows.

    Returns
    -------
    tuple
        `(ord_map, latt)` where `ord_map[(row, col)]` gives the site index
        (0-based) and `latt[site_idx]` gives the `(row, col)` tuple.
    """
    L = lx * ly

    ord_map: dict[tuple[int, int], int] = {}
    # Fill column-by-column, top-to-bottom for every column.
    for idx in range(L):
        row = idx % ly
        col = idx // ly
        ord_map[(row, col)] = idx
    # No column reversal — unlike serpentine which reverses odd columns.

    latt: list[tuple[int, int]] = [None] * L  # type: ignore[list-item]
    for (row, col), site in ord_map.items():
        latt[site] = (row, col)

    _log_sequential_diagram(lx, ly, ord_map)
    return ord_map, latt


def build_traversal_serpentine(
    lx: int,
    ly: int,
) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    """Build serpentine traversal order for a 2D square lattice.

    Creates a mapping between site indices and lattice coordinates for
    a serpentine path through the lattice:

        00. . .07-----08. . .15
        |      |      |      |
        01. . .06. . .09. . .14
        |      |      |      |
        02. . .05. . .10. . .13
        |      |      |      |
        03-----04. . .11-----12

    Logs a visual diagram of the traversal at INFO level.

    Parameters
    ----------
    lx:
        Number of columns.
    ly:
        Number of rows.

    Returns
    -------
    tuple
        `(ord_map, latt)` where `ord_map[(row, col)]` gives the site index
        (0-based) and `latt[site_idx]` gives the `(row, col)` tuple.
    """
    L = lx * ly

    ord_map: dict[tuple[int, int], int] = {}
    for idx in range(L):
        row = idx % ly
        col = idx // ly
        ord_map[(row, col)] = idx

    # Reverse odd columns to create the serpentine pattern.
    for col in range(lx):
        if col % 2 == 1:
            for row in range(ly // 2):
                r1, r2 = row, ly - 1 - row
                ord_map[(r1, col)], ord_map[(r2, col)] = (
                    ord_map[(r2, col)], ord_map[(r1, col)]
                )

    latt: list[tuple[int, int]] = [None] * L  # type: ignore[list-item]
    for (row, col), site in ord_map.items():
        latt[site] = (row, col)

    _log_serpentine_diagram(lx, ly, ord_map)
    return ord_map, latt


# Map traverse key → (lx, ly) → (ord_map, latt).
_TRAVERSALS = {
    'sequential': build_traversal_sequential,
    'serpentine': build_traversal_serpentine,
}


# ---------------------------------------------------------------------------
# Traversal dispatcher
# ---------------------------------------------------------------------------

def build_traversal(
    geo_cfg: dict,
) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    """Select and run the traversal-order generator for a square lattice.

    Parameters
    ----------
    geo_cfg:
        Geometry config dict. Must contain `lx` and optionally `ly`
        (default `1`) and `traverse` (default `'sequential'`).

    Returns
    -------
    tuple
        `(ord_map, latt)` as returned by the selected traversal generator.

    Raises
    ------
    ValueError
        If `traverse` names an unrecognised option.
    """
    # Determine the traversal order, default: sequential order
    traverse = geo_cfg.get('traverse', 'sequential')
    if traverse not in _TRAVERSALS:
        raise ValueError(
            f"Unknown traversal order '{traverse}'. "
            f"Available: {list(_TRAVERSALS)}"
        )
    lx = geo_cfg['lx']
    ly = geo_cfg.get('ly', 1)

    return _TRAVERSALS[traverse](lx, ly)


# ---------------------------------------------------------------------------
# Lattice geometry builder
# ---------------------------------------------------------------------------

def intrcmap_square(geo: Geometry) -> List[Interaction2Site]:
    """Generate an interaction map for a 2D square lattice.

    Produces nearest-neighbor (NN) and optionally next-nearest-neighbor
    (NNN) interactions for a 2D square lattice. Coupling constants are not
    set here; the returned interactions have `cpl == 0.0` (the default).
    Labels encode bond topology so that the model builder can assign the
    correct coupling per bond type.

    Parameters
    ----------
    geo:
        Fully-resolved geometry struct for the square lattice. Relevant
        config keys (read from `geo.cfg`):

        - `bcx` — boundary condition along x (`'OBC'` or `'PBC'`).
        - `bcy` — boundary condition along y (`'OBC'` or `'PBC'`).
        - `n2x` — include NN bonds along x (default `True`).
        - `n2y` — include NN bonds along y (default `True`).
        - `n3d` — include NNN diagonal bonds (default `False`).
        - `n3o` — include NNN off-diagonal bonds (default `False`).

    Returns
    -------
    List[Interaction2Site]
        Interaction objects sorted by `leading_site`. Tensor fields are
        `None`; `cpl` is `0.0`.
    """
    lx  = geo.lx
    ly  = geo.ly
    bcx = geo.cfg.get('bcx', 'OBC').upper()
    bcy = geo.cfg.get('bcy', 'OBC').upper()

    # Bond-inclusion flags.
    n2x = bool(geo.cfg.get('n2x', True))
    n2y = bool(geo.cfg.get('n2y', True))
    n3d = bool(geo.cfg.get('n3d', False))
    n3o = bool(geo.cfg.get('n3o', False))

    # === 1D CHAIN (ly == 1): run naturally; N2Y loop over range(0) emits nothing ===

    ord_map = geo.ord_map

    interactions: List[Interaction2Site] = []

    logger.info("─" * 60)
    logger.info("Interactions Info".center(60))
    logger.info("─" * 60)
    logger.info("")

    if n2x:
        logger.info(" NN interaction along X axis:")
        pairs = []
        for row in range(ly):
            for col in range(lx - 1):
                a, b = ord_map[(row, col)], ord_map[(row, col + 1)]
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
                a, b = ord_map[(row, col)], ord_map[(row + 1, col)]
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
            a, b = ord_map[(row, 0)], ord_map[(row, lx - 1)]
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
            a, b = ord_map[(0, col)], ord_map[(ly - 1, col)]
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
                a, b = ord_map[(row, col)], ord_map[(row - 1, col + 1)]
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
                a, b = ord_map[(row, col)], ord_map[(row + 1, col + 1)]
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
                a, b = ord_map[(row, 0)], ord_map[(row + 1, lx - 1)]
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
                a, b = ord_map[(row, 0)], ord_map[(row - 1, lx - 1)]
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
                a, b = ord_map[(0, col)], ord_map[(ly - 1, col + 1)]
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
                a, b = ord_map[(ly - 1, col - 1)], ord_map[(0, col)]
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
    logger.info(f"Two-site interactions: {len(interactions)}")
    logger.info("")

    return interactions
