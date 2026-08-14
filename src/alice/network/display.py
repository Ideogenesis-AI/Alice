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


"""Text diagram rendering for MPS and MPO chains."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from alice.network.network import Network

# ── characters ────────────────────────────────────────────────────────────────
_MPS_NODE   = '○'   # U+25CB
_MPS_CENTER = '⊙'   # U+2299
_MPO_NODE   = '□'   # U+25A1
_MPO_CENTER = '⊡'   # U+22A1
_BOND       = '───' # U+2500 ×3
_GAP        = '─···─'  # 5 chars: bond, three middle dots, bond
_PHYS_DOWN  = '╵'   # U+2575  — downward-pointing stub (MPS, MPO phys_in)
_PHYS_UP    = '╷'   # U+2577  — upward-pointing stub  (MPO phys_out)
_UNDERLINE  = '‾'   # U+203E

_BOND_BLANK = '   '   # 3 spaces — aligns with _BOND
_GAP_BLANK  = '     ' # 5 spaces — aligns with _GAP

_LINE_WIDTH = 60

_FULL_NAMES = {
    'MPS': 'Matrix Product State (MPS)',
    'MPO': 'Matrix Product Operator (MPO)',
}


def _select_sites(L: int, center: Optional[int], max_sites: int) -> List[int]:
    """Choose which site indices to display.

    Always includes site 0, site `L−1`, and `center` (when set). Remaining
    slots up to `max_sites` are filled by alternately expanding from the left
    and right edges inward.

    Parameters
    ----------
    L:
        Total number of sites.
    center:
        Orthogonality center index, or `None`.
    max_sites:
        Maximum number of sites to show.

    Returns
    -------
    List[int]
        Sorted list of site indices to render.
    """
    if L <= max_sites:
        return list(range(L))

    shown: set = {0, L - 1}
    if center is not None:
        shown.add(center)

    budget = max_sites - len(shown)
    left, right = 1, L - 2
    use_left = True
    while budget > 0 and left <= right:
        candidate = left if use_left else right
        if candidate not in shown:
            shown.add(candidate)
            budget -= 1
        if use_left:
            left += 1
        else:
            right -= 1
        use_left = not use_left

    return sorted(shown)


def network_summary(net: Network, max_sites: int = 9) -> str:
    """Render a text diagram of an MPS or MPO chain.

    The output contains three independently centered blocks within a
    60-character line width:

    1. Title block — full class name + `‾` underline.
    2. Diagram block — chain row(s) and physical-index row(s).
    3. Info block — `length`/`max bond` and `center`/`norm` lines.

    Parameters
    ----------
    net:
        An `MPS` or `MPO` instance.
    max_sites:
        Maximum number of sites to show in the diagram. The orthogonality
        center and both edge sites are always included.

    Returns
    -------
    str
        Multi-line string representation.
    """
    from alice.network.network import MPS  # lazy import avoids circular dependency

    is_mps = isinstance(net, MPS)
    L      = net.L
    center = net.center

    # ── title block ───────────────────────────────────────────────────────────
    cls_name = type(net).__name__
    title    = _FULL_NAMES.get(cls_name, cls_name)
    title_line     = title.center(_LINE_WIDTH)
    underline_line = (_UNDERLINE * len(title)).center(_LINE_WIDTH)

    # ── diagram block ─────────────────────────────────────────────────────────
    shown = _select_sites(L, center, max_sites)

    chain_parts: List[str] = []
    phys_parts:  List[str] = []   # used for MPS (below) and MPO (below)
    phys_up_parts: List[str] = [] # MPO only (above)

    for k, site in enumerate(shown):
        if k > 0:
            prev = shown[k - 1]
            if site == prev + 1:
                seg, blank = _BOND, _BOND_BLANK
            else:
                seg, blank = _GAP, _GAP_BLANK
            chain_parts.append(seg)
            phys_parts.append(blank)
            if not is_mps:
                phys_up_parts.append(blank)

        node = (_MPS_CENTER if site == center else _MPS_NODE) if is_mps \
               else (_MPO_CENTER if site == center else _MPO_NODE)
        chain_parts.append(node)
        phys_parts.append(_PHYS_DOWN)
        if not is_mps:
            phys_up_parts.append(_PHYS_UP)

    chain_row = ''.join(chain_parts)
    phys_row  = ''.join(phys_parts)

    if is_mps:
        diagram_lines = [
            chain_row.center(_LINE_WIDTH),
            phys_row.center(_LINE_WIDTH),
        ]
    else:
        phys_up_row = ''.join(phys_up_parts)
        diagram_lines = [
            phys_up_row.center(_LINE_WIDTH),
            chain_row.center(_LINE_WIDTH),
            phys_row.center(_LINE_WIDTH),
        ]

    # ── info block ────────────────────────────────────────────────────────────
    max_bond = max(net.bond_dims) if L > 1 else 1
    norm     = net.norm()

    info1 = f"length: {str(L):<8}max bond: {max_bond}"
    info2 = f"center: {str(center):<8}norm: {norm:.2e}"

    info_pad = (_LINE_WIDTH - max(len(info1), len(info2))) // 2
    pad      = ' ' * info_pad
    info_lines = [pad + info1, pad + info2]

    # ── assemble ──────────────────────────────────────────────────────────────
    lines = ['', title_line, underline_line] + diagram_lines + info_lines + ['']
    return '\n'.join(lines)
