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


"""1D chain geometry: traversal order and interaction map.

This module provides the `intrcmap_1dchain` geometry builder. The builder
returns a list of `Interaction2Site` objects with `leading_site`,
`terminal_site`, and `label` filled in. Coupling constants (`cpl`) are left
at their default (`0.0`) and are assigned by the model builder in the second
stage of the pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

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


def _log_1dchain_diagram(lx: int, ord_map: dict[tuple[int, int], int]) -> None:
    """Log a visual diagram of the 1D chain lattice."""
    logger.info("─" * 60)
    logger.info(f"1D Chain Lattice ({lx} Sites)".center(60))
    logger.info("─" * 60)
    logger.info("")

    if lx <= _DIAG_THRESHOLD:
        # Full render: all sites connected by "-----".
        line = "-----".join(f"{ord_map[(0, col)]:02d}" for col in range(lx))
    else:
        # Truncated render: first _DIAG_HEAD + last _DIAG_TAIL, with ⋯ ⋯ gap.
        head = "-----".join(f"{ord_map[(0, col)]:02d}" for col in range(_DIAG_HEAD))
        tail = "-----".join(f"{ord_map[(0, col)]:02d}" for col in range(lx - _DIAG_TAIL, lx))
        line = f"{head}  ⋯ ⋯  {tail}"

    logger.info(line.center(60))
    logger.info("")


def _log_pairs(pairs: List[str], indent: int = 3, max_per_line: int = 6) -> None:
    """Log interaction pairs with automatic line wrapping."""
    indent_str = " " * indent
    for i in range(0, len(pairs), max_per_line):
        chunk = pairs[i:i + max_per_line]
        logger.info(indent_str + ", ".join(chunk))


# ---------------------------------------------------------------------------
# Traversal builder
# ---------------------------------------------------------------------------

def build_traversal(
    geo_cfg: dict,
) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    """Build the trivial sequential traversal order for a 1D chain.

    A 1D chain has only one meaningful traversal (sequential), so the
    `traverse` config key is accepted but ignored. Logs a visual diagram
    of the traversal at INFO level.

    Parameters
    ----------
    geo_cfg:
        Geometry config dict. Must contain `lx`.

    Returns
    -------
    tuple
        `(ord_map, latt)` where `ord_map[(0, col)]` gives the site index
        and `latt[site]` gives `(0, col)`.
    """
    # Generate the trivial sequential traversal order for a 1D chain.
    lx = geo_cfg['lx']
    ord_map = {(0, col): col for col in range(lx)}
    latt    = [(0, col) for col in range(lx)]

    _log_1dchain_diagram(lx, ord_map)
    return ord_map, latt


# ---------------------------------------------------------------------------
# 1D chain geometry builder
# ---------------------------------------------------------------------------

def intrcmap_1dchain(geo: Geometry) -> List[Interaction2Site]:
    """Generate an interaction map for a 1D chain.

    Produces nearest-neighbor (NN) and, optionally, next-nearest-neighbor
    (NNN) bonds along the chain. Coupling constants are not set; `cpl` is
    `0.0` on all returned objects.

    Three boundary conditions are supported:

    - `'OBC'` — bonds stay inside `[0, L)`, so the chain has open ends.
    - `'PBC'` — bonds additionally wrap from the right end back to the left,
      normalized so that `leading_site < terminal_site`.
    - `'IBC'` — the `L` sites form one unit cell of an infinite chain. Every
      site leads a bond, and a `terminal_site` at or beyond `L` denotes a
      site in the *next* unit cell; `build_ihamiltonian` folds those back
      modulo `L` when it lays out the operator-string channels. Unlike PBC
      these are not wrap-around bonds within a finite ring: site `L-1` is
      bonded to site `0` of the following cell, not to site `0` of its own.

    Parameters
    ----------
    geo:
        Fully-resolved geometry struct for the 1D chain. Relevant config
        keys (read from `geo.cfg`):

        - `bcx` — boundary condition (`'OBC'`, `'PBC'`, or `'IBC'`).
        - `n2x` — include NN bonds (default `True`).
        - `n3x` — include NNN bonds (default `False`).

    Returns
    -------
    List[Interaction2Site]
        Interaction objects sorted by `leading_site`. Tensor fields are
        `None`; `cpl` is `0.0`.

    Raises
    ------
    ValueError
        If `bcx` is not one of `'OBC'`, `'PBC'`, or `'IBC'`.
    """
    L   = geo.lx
    bcx = geo.cfg.get('bcx', 'OBC').upper()
    n2x = bool(geo.cfg.get('n2x', True))
    n3x = bool(geo.cfg.get('n3x', False))

    if bcx not in ('OBC', 'PBC', 'IBC'):
        raise ValueError(f"bcx must be 'OBC', 'PBC', or 'IBC', got '{bcx}'")

    interactions: List[Interaction2Site] = []

    logger.info("─" * 60)
    logger.info("Interactions Info".center(60))
    logger.info("─" * 60)
    logger.info("")

    def _emit(dist: int, labels: List[str], heading: str) -> None:
        """Append every bond of range `dist`, honoring the boundary condition."""
        logger.info(heading)
        pairs = []
        # IBC lets the terminal index run past the cell; the extra bonds are
        # exactly the ones that couple this cell to the next one.
        last = L if bcx == 'IBC' else L - dist
        for si in range(last):
            interactions.append(Interaction2Site(
                label=list(labels),
                leading_site=si,
                terminal_site=si + dist,
            ))
            pairs.append(f"({si:02d},{si+dist:02d})")
        _log_pairs(pairs)

        if bcx == 'PBC':
            logger.info("")
            logger.info(f" PBC interaction at X edge (range {dist}):")
            pairs = []
            for si in range(L - dist, L):
                # Normalize so that leading_site < terminal_site, which is
                # what Interaction2Site and build_hamiltonian require.
                lo, hi = sorted(((si + dist) % L, si))
                interactions.append(Interaction2Site(
                    label=[*labels, 'PBC'],
                    leading_site=lo,
                    terminal_site=hi,
                ))
                pairs.append(f"({lo:02d},{hi:02d})")
            _log_pairs(pairs)

    if n2x:
        _emit(1, ['NN', 'N2X'], " NN interaction (N2X):")

    if n3x:
        if n2x:
            logger.info("")
        _emit(2, ['NNN', 'N3X'], " NNN interaction (N3X):")

    interactions.sort(key=lambda x: x.leading_site)

    logger.info("")
    logger.info(f"Two-site interactions: {len(interactions)}")
    logger.info("")

    return interactions
