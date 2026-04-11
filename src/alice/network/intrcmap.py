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


"""Interaction map generation for MPS on 2D lattices.

This module provides functions to generate interaction maps for 1D MPS
traversing 2D lattices using various orderings (snake-like, etc.).
"""

import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


def _log_lattice_diagram(lx: int, ly: int, ord_map: list[list[int]]) -> None:
    """Log a visual diagram of the snake-like lattice traversal."""
    # Header
    logger.info("=" * 60)
    logger.info("Traverse over 2D Lattice via Snake-like Chain".center(60))
    logger.info("=" * 60)
    logger.info("")
    
    # Calculate diagram width and centering offset
    # Each site: 2 chars, each connector: 5 chars
    # Total width = lx * 2 + (lx - 1) * 5 = 7 * lx - 5
    diagram_width = 7 * lx - 5
    left_padding = max(0, (60 - diagram_width) // 2)
    padding = " " * left_padding
    
    for row in range(ly):
        line = ""
        for col in range(lx):
            site = ord_map[row][col]
            # Print horizontal connections
            if col < lx - 1:
                if (row == 0 and col % 2 == 1) or (row == ly - 1 and col % 2 == 0):
                    connector = "-----"
                else:
                    connector = ". . ."
            else:
                connector = ""
            line += f"{site:02d}{connector}"
        logger.info(padding + line)
        
        # Print vertical connections
        if row < ly - 1:
            line = ""
            for col in range(lx):
                line += "|      "
            logger.info(padding + line)
    
    logger.info("")


def _log_pairs(pairs: list[str], indent: int = 3, max_per_line: int = 6) -> None:
    """Log interaction pairs with automatic line breaks.
    
    Args:
        pairs: List of pair strings like "(00,01)"
        indent: Number of spaces for indentation
        max_per_line: Maximum number of pairs per line
    """
    indent_str = " " * indent
    for i in range(0, len(pairs), max_per_line):
        chunk = pairs[i:i + max_per_line]
        logger.info(indent_str + ", ".join(chunk))


class Interaction(TypedDict):
    """Interaction term between two sites.
    
    Attributes:
        start_site: Starting site index (0-based)
        terminal_site: Ending site index (0-based)
        cpl: Coupling strength
        label: List of labels describing the interaction type
    """
    start_site: int
    terminal_site: int
    cpl: float
    label: list[str]


def generate_snake_order(lx: int, ly: int) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Generate snake-like traversal order for a 2D square lattice.
    
    Creates a mapping between site indices and lattice coordinates for
    a snake-like path through the lattice:
    
        00. . .07-----08. . .15
        |      |      |      |
        01. . .06. . .09. . .14
        |      |      |      |
        02. . .05. . .10. . .13
        |      |      |      |
        03-----04. . .11-----12
    
    Args:
        lx: Number of columns
        ly: Number of rows
    
    Returns:
        ord_map: 2D list where ord_map[row][col] gives the site index (0-based)
        latt: List where latt[site_idx] gives (row, col) tuple (0-based indexing)
    """
    L = lx * ly
    
    # Initialize with sequential numbering (0-based)
    ord_map = [[0] * lx for _ in range(ly)]
    for idx in range(L):
        row = idx % ly
        col = idx // ly
        ord_map[row][col] = idx
    
    # Reverse odd columns to create snake pattern
    for col in range(lx):
        if col % 2 == 1:
            for row in range(ly // 2):
                ord_map[row][col], ord_map[ly - 1 - row][col] = \
                    ord_map[ly - 1 - row][col], ord_map[row][col]
    
    # Create lattice coordinate lookup (0-based site indexing)
    latt = [(0, 0)] * L
    for row in range(ly):
        for col in range(lx):
            site_idx = ord_map[row][col]
            latt[site_idx] = (row, col)
    
    return ord_map, latt


def intrcmap_square(config: dict) -> list[Interaction]:
    """Generate interaction map for square lattice with snake-like MPS traversal.
    
    Generates nearest-neighbor (NN) and next-nearest-neighbor (NNN) interactions
    for a 2D square lattice traversed by a 1D MPS in snake-like order.
    
    Expected config structure (from TOML):
        lx: int - Number of columns
        ly: int - Number of rows
        bcx: str - Boundary condition in x ('OBC' or 'PBC')
        bcy: str - Boundary condition in y ('OBC' or 'PBC')
        label: str - Model label (e.g., 'SpinSqLatt', 'HubbardSqLatt')
        cpl: float or list[float] - NN coupling(s)
        cplp: list[float, float] - NNN couplings [diagonal, off-diagonal]
    
    Args:
        config: Configuration dictionary from TOML file
    
    Returns:
        List of interaction dictionaries with keys:
            - start_site: Starting site (0-based)
            - terminal_site: Terminal site (0-based)
            - cpl: Coupling strength
            - label: List of interaction type labels
    """
    # Extract configuration
    lx = config['lx']
    ly = config['ly']
    L = lx * ly
    bcx = config.get('bcx', 'OBC').upper()
    bcy = config.get('bcy', 'OBC').upper()
    label = config.get('label', '')
    
    # Get coupling parameters based on model type
    if 'Spin' in label:
        cpl = config.get('cpl', config.get('j1', 1.0))
        cplp = config.get('cplp', config.get('C2', [0.0, 0.0]))
    else:  # Hubbard or spinless fermion
        cpl = config.get('cpl', config.get('t1', 1.0))
        cplp = config.get('cplp', config.get('t2', [0.0, 0.0]))
    
    # Ensure cplp is a list
    if not isinstance(cplp, list):
        cplp = [cplp, cplp]
    if len(cplp) == 1:
        cplp = [cplp[0], cplp[0]]
    
    # Generate snake-like order
    ord_map, latt = generate_snake_order(lx, ly)
    
    interactions: list[Interaction] = []
    
    # Log lattice visualization
    _log_lattice_diagram(lx, ly, ord_map)
    
    logger.info("=" * 60)
    logger.info("Interactions Info".center(60))
    logger.info("=" * 60)
    logger.info("")
    
    # === 1D CHAIN (ly == 1) ===
    if ly == 1:
        logger.info(" NN interaction:")
        pairs = []
        for si in range(L - 1):
            interaction: Interaction = {
                'start_site': si,
                'terminal_site': si + 1,
                'cpl': cpl,
                'label': ['NN', 'N2Y']
            }
            interactions.append(interaction)
            pairs.append(f"({si:02d},{si+1:02d})")
        _log_pairs(pairs)
    
    # === 2D LATTICE (ly > 1) ===
    if ly > 1:
        # NN interactions along X axis (horizontal)
        logger.info(" NN interaction along X axis:")
        pairs = []
        for si in range(L):
            col = si // ly
            if col == lx - 1:  # Last column
                break
            
            terminal = 2 * (col + 1) * ly - 1 - si
            interaction: Interaction = {
                'start_site': si,
                'terminal_site': terminal,
                'cpl': cpl,
                'label': ['NN', 'N2X']
            }
            interactions.append(interaction)
            pairs.append(f"({si:02d},{terminal:02d})")
        _log_pairs(pairs)
        
        # NN interactions along Y axis (vertical)
        logger.info("")
        logger.info(" NN interaction along Y axis:")
        pairs = []
        for si in range(L - 1):
            if si % ly != ly - 1:  # Not at bottom of column
                interaction: Interaction = {
                    'start_site': si,
                    'terminal_site': si + 1,
                    'cpl': cpl,
                    'label': ['NN', 'N2Y']
                }
                interactions.append(interaction)
                pairs.append(f"({si:02d},{si+1:02d})")
        _log_pairs(pairs)
    
    # === PBC along X direction ===
    if bcx == 'PBC':
        logger.info("")
        logger.info(" PBC interaction at X edge:")
        pairs = []
        for si in range(ly):
            if lx % 2 == 0:  # Even number of columns
                terminal = L - 1 - si
            else:  # Odd number of columns
                terminal = L - ly + si
            
            interaction: Interaction = {
                'start_site': si,
                'terminal_site': terminal,
                'cpl': cpl,
                'label': ['NN', 'PBC', 'N2X']
            }
            interactions.append(interaction)
            pairs.append(f"({si:02d},{terminal:02d})")
        _log_pairs(pairs)
    
    # === PBC along Y direction ===
    if bcy == 'PBC':
        logger.info("")
        logger.info(" PBC interaction at Y edge:")
        pairs = []
        for si in range(lx):
            start = si * ly
            terminal = si * ly + ly - 1
            
            interaction: Interaction = {
                'start_site': start,
                'terminal_site': terminal,
                'cpl': cpl,
                'label': ['NN', 'PBC', 'N2Y']
            }
            interactions.append(interaction)
            pairs.append(f"({start:02d},{terminal:02d})")
        _log_pairs(pairs)
    
    # === NNN C2 interactions ===
    # Type O (off-diagonal): connects (row, col) with (row-1, col+1)
    if cplp[1] != 0 and ly > 1:
        logger.info("")
        logger.info(" Adding C2 type(O) interaction:")
        pairs = []
        for col in range(lx - 1):
            for row in range(1, ly):
                start = ord_map[row][col]
                terminal = ord_map[row - 1][col + 1]
                
                interaction: Interaction = {
                    'start_site': start,
                    'terminal_site': terminal,
                    'cpl': cplp[1],
                    'label': ['NNN', 'N3O']
                }
                interactions.append(interaction)
                pairs.append(f"({start:02d},{terminal:02d})")
        _log_pairs(pairs)
    
    # Type D (diagonal): connects (row, col) with (row+1, col+1)
    if cplp[0] != 0 and ly > 1:
        logger.info("")
        logger.info(" Adding C2 type(D) interaction:")
        pairs = []
        for col in range(lx - 1):
            for row in range(ly - 1):
                start = ord_map[row][col]
                terminal = ord_map[row + 1][col + 1]
                
                interaction: Interaction = {
                    'start_site': start,
                    'terminal_site': terminal,
                    'cpl': cplp[0],
                    'label': ['NNN', 'N3D']
                }
                interactions.append(interaction)
                pairs.append(f"({start:02d},{terminal:02d})")
        _log_pairs(pairs)
    
    # === C2 PBC along X ===
    if bcx == 'PBC':
        if cplp[1] != 0 and ly > 1:
            logger.info("")
            logger.info(" Adding C2(O) PBC interaction at X edge:")
            pairs = []
            for row in range(ly - 1):
                start = ord_map[row][0]
                terminal = ord_map[row + 1][lx - 1]
                
                interaction: Interaction = {
                    'start_site': start,
                    'terminal_site': terminal,
                    'cpl': cplp[1],
                    'label': ['NNN', 'PBC', 'N3O']
                }
                interactions.append(interaction)
                pairs.append(f"({start:02d},{terminal:02d})")
            _log_pairs(pairs)
        
        if cplp[0] != 0 and ly > 1:
            logger.info("")
            logger.info(" Adding C2(D) PBC interaction at X edge:")
            pairs = []
            for row in range(1, ly):
                start = ord_map[row][0]
                terminal = ord_map[row - 1][lx - 1]
                
                interaction: Interaction = {
                    'start_site': start,
                    'terminal_site': terminal,
                    'cpl': cplp[0],
                    'label': ['NNN', 'PBC', 'N3D']
                }
                interactions.append(interaction)
                pairs.append(f"({start:02d},{terminal:02d})")
            _log_pairs(pairs)
    
    # === C2 PBC along Y ===
    if bcy == 'PBC':
        if cplp[1] != 0 and lx > 1:
            logger.info("")
            logger.info(" Adding C2(O) PBC interaction at Y edge:")
            pairs = []
            for col in range(lx - 1):
                start = ord_map[0][col]
                terminal = ord_map[ly - 1][col + 1]
                
                interaction: Interaction = {
                    'start_site': start,
                    'terminal_site': terminal,
                    'cpl': cplp[1],
                    'label': ['NNN', 'PBC', 'N3O']
                }
                interactions.append(interaction)
                pairs.append(f"({start:02d},{terminal:02d})")
            _log_pairs(pairs)
        
        if cplp[0] != 0 and lx > 1:
            logger.info("")
            logger.info(" Adding C2(D) PBC interaction at Y edge:")
            pairs = []
            for col in range(1, lx):
                start = ord_map[ly - 1][col - 1]
                terminal = ord_map[0][col]
                
                interaction: Interaction = {
                    'start_site': start,
                    'terminal_site': terminal,
                    'cpl': cplp[0],
                    'label': ['NNN', 'PBC', 'N3D']
                }
                interactions.append(interaction)
                pairs.append(f"({start:02d},{terminal:02d})")
            _log_pairs(pairs)
    
    # Sort interactions by start_site
    interactions.sort(key=lambda x: x['start_site'])
    
    logger.info("")
    logger.info(f"Total interactions: {len(interactions)}")
    
    return interactions
