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


"""Kagome-lattice geometry: traversal orders and interaction map.

This module provides the snake-order traversal generator for the Kagome
lattice (`build_traversal_snake`) and the `intrcmap_kagome` geometry builder.
The builder returns a list of `Interaction2Site` objects with `leading_site`,
`terminal_site`, and `label` filled in.  Coupling constants (`cpl`) are left
at their default (`0.0`) and are assigned by the model builder in the second
stage of the pipeline.

The Kagome lattice has a 3-site unit cell (sublattices A=0, B=1, C=2).
With `lx × ly` unit cells the total MPS length is `L = lx * ly * 3`.
`ord_map` is keyed by `(row, col, u)` and `latt[site]` returns `(row, col, u)`.

For this lattice `geo.L` correctly returns `lx * ly * 3` (the length of
`geo.latt`), as `Geometry.L` is defined as `len(latt)` for all lattice types.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Dict, List

from alice.network.interaction import Interaction2Site

if TYPE_CHECKING:
    from alice.physics.geometry import Geometry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_DIAG_THRESHOLD = 4   # lx > this → truncate the diagram
_DIAG_HEAD      = 2   # columns shown at the left in truncated mode
_DIAG_TAIL      = 1   # columns shown at the right in truncated mode


_TRAVERSE_TITLES: Dict[str, str] = {
    'snake':  'Snake-like Chain',
    'zigzag': 'Zigzag Chain',
}


def _log_kagome_diagram(
    lx: int,
    ly: int,
    ord_map: dict[tuple[int, int, int], int],
    traverse: str = 'snake',
) -> None:
    """Log a staggered-row ASCII diagram of the Kagome traversal.

    The diagram follows the convention used by `_log_2d_diagram` in
    `square.py`: centered within a 60-character log width. Each unit cell
    occupies two text lines — the A–B base and the C apex — separated from
    the next row by an inter-row connector line showing bonds 5 and 6.

    Within each unit cell, A and B are on the base line (B is 5 characters
    to the right of A); C is on the line below, 2 characters to the right
    of A (apex of the upward triangle).

    Parameters
    ----------
    lx:
        Number of unit-cell columns.
    ly:
        Number of unit-cell rows.
    ord_map:
        Mapping `(row, col, u)` → MPS site index.
    traverse:
        Traversal key used to select the diagram title (e.g. `'snake'`,
        `'zigzag'`).
    """
    title = f"Traverse over Kagome Lattice via {_TRAVERSE_TITLES.get(traverse, traverse)}"
    logger.info("─" * 60)
    logger.info(title.center(60))
    logger.info("─" * 60)
    logger.info("")

    truncate = lx > _DIAG_THRESHOLD

    # Centering padding:
    #   Full mode:      width = 12*(lx-1) + 6*(ly-1) + 8
    #   Truncated mode: width = head (HEAD cols) + gap (5) + tail (1 col)
    #                         = 12*(HEAD-1) + 6*(ly-1) + 8 + 5 + 8
    if not truncate:
        diagram_width = 12 * (lx - 1) + 6 * (ly - 1) + 8
    else:
        diagram_width = 12 * (_DIAG_HEAD - 1) + 6 * (ly - 1) + 21
    padding = " " * max(0, (60 - diagram_width) // 2)
    gap     = "  ⋯  "

    # Scratch buffer large enough for any row in full mode.
    buf_size = 12 * (lx - 1) + 6 * (ly - 1) + 12

    def _make_buf() -> list[str]:
        return [' '] * buf_size

    def _put(buf: list[str], x: int, text: str) -> None:
        for i, ch in enumerate(text):
            if 0 <= x + i < len(buf):
                buf[x + i] = ch

    def _render(buf: list[str]) -> str:
        return ''.join(buf).rstrip()

    def _emit(buf: list[str], stagger: int) -> None:
        if not truncate:
            logger.info(padding + _render(buf))
        else:
            # head: cols 0 .. HEAD-1, up to and including last B label
            head_end   = 12 * (_DIAG_HEAD - 1) + stagger + 8
            # tail: last column at its natural x position
            tail_start = 12 * (lx - 1) + stagger
            head_str   = _render(buf[:head_end])
            tail_str   = _render(buf[tail_start:])
            logger.info(padding + head_str + gap + tail_str)

    for row in range(ly):
        stagger = 6 * row

        # A–B base line
        ab = _make_buf()
        for col in range(lx):
            x = 12 * col + stagger
            _put(ab, x,     f"{ord_map[(row, col, 0)]:02d}")
            _put(ab, x + 2, "────")
            _put(ab, x + 6, f"{ord_map[(row, col, 1)]:02d}")
            if col < lx - 1:
                _put(ab, x + 8, "····")
        _emit(ab, stagger)

        # N2U connector: \ at x+2 (A→C), / at x+5 (B→C)
        n2u = _make_buf()
        for col in range(lx):
            x = 12 * col + stagger
            _put(n2u, x + 2, "\\")
            _put(n2u, x + 5, "/")
        _emit(n2u, stagger)

        # C apex line
        c = _make_buf()
        for col in range(lx):
            x = 12 * col + stagger
            _put(c, x + 3, f"{ord_map[(row, col, 2)]:02d}")
        _emit(c, stagger)

        # Inter-row connector (bond 5 \ and bond 6 /)
        if row < ly - 1:
            ir = _make_buf()
            for col in range(lx):
                x = 12 * col + stagger
                _put(ir, x + 5, "\\")       # bond 5: C(col,row) → A(col,row+1)
                if col >= 1:
                    _put(ir, x + 2, "/")    # bond 6: C(col,row) ↔ B(col-1,row+1)
            _emit(ir, stagger)

    logger.info("")


def _log_pairs(pairs: list[str], indent: int = 3, max_per_line: int = 6) -> None:
    """Log interaction pairs with automatic line wrapping."""
    indent_str = " " * indent
    for i in range(0, len(pairs), max_per_line):
        chunk = pairs[i:i + max_per_line]
        logger.info(indent_str + ", ".join(chunk))


# ---------------------------------------------------------------------------
# Traversal builder
# ---------------------------------------------------------------------------

def build_traversal_zigzag(
    lx: int,
    ly: int,
) -> tuple[dict[tuple[int, int, int], int], list[tuple[int, int, int]]]:
    r"""Build zigzag traversal order for a Kagome lattice.

    Column-major zigzag: all columns fill rows top→bottom, with no reversal.
    Within each (col, row) unit cell the three sublattice sites are ordered
    A (u=0), B (u=1), C (u=2).

    For `lx=3, ly=3` the MPS indices are:

                    col=0        col=1        col=2
        row=0    [ 0,  1,  2]  [ 9, 10, 11]  [18, 19, 20]
        row=1    [ 3,  4,  5]  [12, 13, 14]  [21, 22, 23]
        row=2    [ 6,  7,  8]  [15, 16, 17]  [24, 25, 26]

    The logged diagram for the same case:

            00────01····09────10····18────19
              \  /        \  /        \  /
               02          11          20
                 \        /  \        /  \
                  03────04····12────13····21────22
                    \  /        \  /        \  /
                     05          14          23
                       \        /  \        /  \
                        06────07····15────16····24────25
                          \  /        \  /        \  /
                           08          17          26

    Logs a visual diagram of the traversal at INFO level.

    Parameters
    ----------
    lx:
        Number of unit-cell columns.
    ly:
        Number of unit-cell rows.

    Returns
    -------
    tuple
        `(ord_map, latt)` where `ord_map[(row, col, u)]` gives the MPS
        site index (0-based) and `latt[site]` gives `(row, col, u)`.
    """
    L = lx * ly * 3

    ord_map: dict[tuple[int, int, int], int] = {}
    site = 0
    for col in range(lx):
        for row in range(ly):
            for u in range(3):
                ord_map[(row, col, u)] = site
                site += 1

    latt: list[tuple[int, int, int]] = [None] * L  # type: ignore[list-item]
    for coord, s in ord_map.items():
        latt[s] = coord

    _log_kagome_diagram(lx, ly, ord_map, 'zigzag')
    return ord_map, latt


def build_traversal_snake(
    lx: int,
    ly: int,
) -> tuple[dict[tuple[int, int, int], int], list[tuple[int, int, int]]]:
    r"""Build snake-like traversal order for a Kagome lattice.

    Column-major snake: even columns fill rows top→bottom, odd columns
    fill rows bottom→top. Within each (col, row) unit cell the three
    sublattice sites are ordered A (u=0), B (u=1), C (u=2).

    For `lx=3, ly=3` the MPS indices are:

                    col=0        col=1        col=2
        row=0    [ 0,  1,  2]  [15, 16, 17]  [18, 19, 20]
        row=1    [ 3,  4,  5]  [12, 13, 14]  [21, 22, 23]
        row=2    [ 6,  7,  8]  [ 9, 10, 11]  [24, 25, 26]

    The logged diagram for the same case:

            00────01····15────16····18────19
              \  /        \  /        \  /
               02          17          20
                 \        /  \        /  \
                  03────04····12────13····21────22
                    \  /        \  /        \  /
                     05          14          23
                       \        /  \        /  \
                        06────07····09────10····24────25
                          \  /        \  /        \  /
                           08          11          26

    Logs a visual diagram of the traversal at INFO level.

    Parameters
    ----------
    lx:
        Number of unit-cell columns.
    ly:
        Number of unit-cell rows.

    Returns
    -------
    tuple
        `(ord_map, latt)` where `ord_map[(row, col, u)]` gives the MPS
        site index (0-based) and `latt[site]` gives `(row, col, u)`.
    """
    L = lx * ly * 3

    ord_map: dict[tuple[int, int, int], int] = {}
    site = 0
    for col in range(lx):
        rows = range(ly) if col % 2 == 0 else range(ly - 1, -1, -1)
        for row in rows:
            for u in range(3):
                ord_map[(row, col, u)] = site
                site += 1

    latt: list[tuple[int, int, int]] = [None] * L  # type: ignore[list-item]
    for coord, s in ord_map.items():
        latt[s] = coord

    _log_kagome_diagram(lx, ly, ord_map, 'snake')
    return ord_map, latt


_TRAVERSALS: Dict[str, Callable] = {
    'snake':  build_traversal_snake,
    'zigzag': build_traversal_zigzag,
}


# ---------------------------------------------------------------------------
# Traversal dispatcher
# ---------------------------------------------------------------------------

def build_traversal(
    geo_cfg: dict,
) -> tuple[dict[tuple[int, int, int], int], list[tuple[int, int, int]]]:
    """Select and run the traversal-order generator for a Kagome lattice.

    Parameters
    ----------
    geo_cfg:
        Geometry config dict. Must contain `lx` and optionally `ly`
        (default `1`) and `traverse` (default `'snake'`; also accepts
        `'zigzag'`).

    Returns
    -------
    tuple
        `(ord_map, latt)` as returned by the selected traversal generator.

    Raises
    ------
    ValueError
        If `traverse` names an unrecognised option.
    """
    traverse = geo_cfg.get('traverse', 'snake')
    if traverse not in _TRAVERSALS:
        raise ValueError(
            f"Unknown traversal order '{traverse}' for Kagome lattice. "
            f"Available: {list(_TRAVERSALS)}"
        )
    lx = geo_cfg['lx']
    ly = geo_cfg.get('ly', 1)
    return _TRAVERSALS[traverse](lx, ly)


# ---------------------------------------------------------------------------
# Kagome geometry builder
# ---------------------------------------------------------------------------

def intrcmap_kagome(geo: Geometry) -> List[Interaction2Site]:
    """Generate an interaction map for a Kagome lattice.

    Produces nearest-neighbor (NN) interactions within each upward triangle
    (N2U) and between unit cells via three downward-triangle bond families
    (N2D).  Coupling constants are not set here; all returned interactions
    have `cpl == 0.0`.

    Labels encode bond topology:

    - `['NN', 'N2U']` — upward-triangle bonds (A–B, A–C, B–C within a cell).
    - `['NN', 'N2D']` — downward-triangle bonds (bonds 4, 5, 6 between cells).
    - PBC bonds additionally carry `'PBC'` in the label list.

    Parameters
    ----------
    geo:
        Fully-resolved geometry struct for the Kagome lattice. Relevant
        config keys (read from `geo.cfg`):

        - `bcx` — boundary condition along x (`'OBC'` or `'PBC'`).
        - `bcy` — boundary condition along y (`'OBC'` or `'PBC'`).
        - `n2u` — include N2U (upward-triangle) bonds (default `True`).
        - `n2d` — include N2D (downward-triangle) bonds (default `True`).

    Returns
    -------
    List[Interaction2Site]
        Interaction objects sorted by `leading_site`.  Tensor fields are
        `None`; `cpl` is `0.0`.
    """
    lx  = geo.lx
    ly  = geo.ly
    bcx = geo.cfg.get('bcx', 'OBC').upper()
    bcy = geo.cfg.get('bcy', 'OBC').upper()
    n2u = bool(geo.cfg.get('n2u', True))
    n2d = bool(geo.cfg.get('n2d', True))

    ord_map = geo.ord_map
    interactions: List[Interaction2Site] = []

    logger.info("─" * 60)
    logger.info("Interactions Info".center(60))
    logger.info("─" * 60)
    logger.info("")

    def _add(a: int, b: int, label: list[str]) -> tuple[int, int]:
        lo, hi = min(a, b), max(a, b)
        interactions.append(Interaction2Site(
            label=label,
            leading_site=lo,
            terminal_site=hi,
        ))
        return lo, hi

    # N2U bonds: A–B, A–C, B–C within each unit cell (upward triangle)
    if n2u:
        logger.info(" N2U interactions (upward triangles, A–B, A–C, B–C):")
        pairs = []
        for row in range(ly):
            for col in range(lx):
                a = ord_map[(row, col, 0)]  # A
                b = ord_map[(row, col, 1)]  # B
                c = ord_map[(row, col, 2)]  # C
                for x, y in ((a, b), (a, c), (b, c)):
                    lo, hi = _add(x, y, ['NN', 'N2U'])
                    pairs.append(f"({lo:02d},{hi:02d})")
        _log_pairs(pairs)

    # N2D bonds: three bond families crossing between unit cells
    if n2d:
        # Bond 4 (OBC): B(col,row) ↔ A(col+1,row), horizontal
        logger.info("")
        logger.info(" N2D bond 4 (horizontal, B↔A_right):")
        pairs = []
        for row in range(ly):
            for col in range(lx - 1):
                lo, hi = _add(
                    ord_map[(row, col, 1)],
                    ord_map[(row, col + 1, 0)],
                    ['NN', 'N2D'],
                )
                pairs.append(f"({lo:02d},{hi:02d})")
        _log_pairs(pairs)

        # Bond 5 (OBC): C(col,row) ↔ A(col,row+1), up-right
        logger.info("")
        logger.info(" N2D bond 5 (up-right, C↔A_below):")
        pairs = []
        for row in range(ly - 1):
            for col in range(lx):
                lo, hi = _add(
                    ord_map[(row, col, 2)],
                    ord_map[(row + 1, col, 0)],
                    ['NN', 'N2D'],
                )
                pairs.append(f"({lo:02d},{hi:02d})")
        _log_pairs(pairs)

        # Bond 6 (OBC): C(col,row) ↔ B(col-1,row+1), up-left
        logger.info("")
        logger.info(" N2D bond 6 (up-left, C↔B_below-left):")
        pairs = []
        for row in range(ly - 1):
            for col in range(1, lx):
                lo, hi = _add(
                    ord_map[(row, col, 2)],
                    ord_map[(row + 1, col - 1, 1)],
                    ['NN', 'N2D'],
                )
                pairs.append(f"({lo:02d},{hi:02d})")
        _log_pairs(pairs)

        # PBC along X
        if bcx == 'PBC':
            logger.info("")
            logger.info(" N2D PBC (bcx) — bond 4 wrap, B(lx-1,row)↔A(0,row):")
            pairs = []
            for row in range(ly):
                lo, hi = _add(
                    ord_map[(row, lx - 1, 1)],
                    ord_map[(row, 0, 0)],
                    ['NN', 'N2D', 'PBC'],
                )
                pairs.append(f"({lo:02d},{hi:02d})")
            _log_pairs(pairs)

            logger.info("")
            logger.info(" N2D PBC (bcx) — bond 6 wrap, C(0,row)↔B(lx-1,row+1):")
            pairs = []
            for row in range(ly - 1):
                lo, hi = _add(
                    ord_map[(row, 0, 2)],
                    ord_map[(row + 1, lx - 1, 1)],
                    ['NN', 'N2D', 'PBC'],
                )
                pairs.append(f"({lo:02d},{hi:02d})")
            _log_pairs(pairs)

        # PBC along Y
        if bcy == 'PBC':
            logger.info("")
            logger.info(" N2D PBC (bcy) — bond 5 wrap, C(col,ly-1)↔A(col,0):")
            pairs = []
            for col in range(lx):
                lo, hi = _add(
                    ord_map[(ly - 1, col, 2)],
                    ord_map[(0, col, 0)],
                    ['NN', 'N2D', 'PBC'],
                )
                pairs.append(f"({lo:02d},{hi:02d})")
            _log_pairs(pairs)

            logger.info("")
            logger.info(" N2D PBC (bcy) — bond 6 wrap, C(col,ly-1)↔B(col-1,0):")
            pairs = []
            for col in range(1, lx):
                lo, hi = _add(
                    ord_map[(ly - 1, col, 2)],
                    ord_map[(0, col - 1, 1)],
                    ['NN', 'N2D', 'PBC'],
                )
                pairs.append(f"({lo:02d},{hi:02d})")
            _log_pairs(pairs)

        # Corner bond: bcx + bcy combined wrap — C(0,ly-1) ↔ B(lx-1,0)
        if bcx == 'PBC' and bcy == 'PBC':
            logger.info("")
            logger.info(" N2D PBC (corner) — C(0,ly-1)↔B(lx-1,0):")
            lo, hi = _add(
                ord_map[(ly - 1, 0, 2)],
                ord_map[(0, lx - 1, 1)],
                ['NN', 'N2D', 'PBC'],
            )
            _log_pairs([f"({lo:02d},{hi:02d})"])

    interactions.sort(key=lambda x: x.leading_site)

    logger.info("")
    logger.info(f"Two-site interactions: {len(interactions)}")
    logger.info("")

    return interactions
