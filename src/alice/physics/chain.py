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

    Produces nearest-neighbor (NN) bonds along the chain and, when
    `bcx='PBC'`, a single periodic bond connecting the two ends.
    Coupling constants are not set; `cpl` is `0.0` on all returned objects.

    Parameters
    ----------
    geo:
        Fully-resolved geometry struct for the 1D chain. Relevant config
        keys (read from `geo.cfg`):

        - `bcx` — boundary condition (`'OBC'` or `'PBC'`).
        - `n2x` — include NN bonds (default `True`).

    Returns
    -------
    List[Interaction2Site]
        Interaction objects sorted by `leading_site`. Tensor fields are
        `None`; `cpl` is `0.0`.
    """
    L   = geo.lx
    bcx = geo.cfg.get('bcx', 'OBC').upper()
    n2x = bool(geo.cfg.get('n2x', True))

    interactions: List[Interaction2Site] = []

    logger.info("─" * 60)
    logger.info("Interactions Info".center(60))
    logger.info("─" * 60)
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
    logger.info(f"Two-site interactions: {len(interactions)}")
    logger.info("")

    return interactions
