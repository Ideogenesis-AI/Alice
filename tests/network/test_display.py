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


"""Tests for alice.network.display — network_summary and __repr__/__str__."""

from __future__ import annotations

import pytest

from alice.network import MPS, MPO
from alice.network.display import _select_sites, network_summary


# ── helpers ───────────────────────────────────────────────────────────────────

def _lines(net, **kw):
    return network_summary(net, **kw).splitlines()


# ── TestSelectSites ───────────────────────────────────────────────────────────

class TestSelectSites:
    """Unit tests for the site-selection helper."""

    def test_short_chain_returns_all(self):
        assert _select_sites(5, None, 9) == [0, 1, 2, 3, 4]

    def test_exact_max_returns_all(self):
        assert _select_sites(9, None, 9) == list(range(9))

    def test_always_includes_edges(self):
        for L in (10, 20, 50):
            sites = _select_sites(L, None, 5)
            assert 0 in sites
            assert L - 1 in sites

    def test_always_includes_center(self):
        sites = _select_sites(20, 10, 7)
        assert 10 in sites

    def test_respects_max_sites(self):
        for max_s in (3, 5, 7, 9):
            sites = _select_sites(50, 25, max_s)
            assert len(sites) == max_s

    def test_no_center_fills_from_edges(self):
        sites = _select_sites(20, None, 5)
        assert sites[0] == 0
        assert sites[-1] == 19
        # inner sites come from both edges
        assert sites[1] < sites[-2]

    def test_center_near_edge(self):
        sites = _select_sites(20, 1, 7)
        assert 0 in sites
        assert 1 in sites
        assert 19 in sites
        assert len(sites) == 7

    def test_sorted(self):
        sites = _select_sites(30, 15, 7)
        assert sites == sorted(sites)


# ── TestNetworkSummary ────────────────────────────────────────────────────────

class TestNetworkSummary:
    """Structural checks on the rendered string."""

    # ── line count ────────────────────────────────────────────────────────────

    def test_mps_line_count(self, mps_tensors):
        mps = MPS(mps_tensors)
        # blank + title + underline + chain + phys_down + info×2 = 7
        assert len(_lines(mps)) == 7

    def test_mpo_line_count(self, mpo_tensors):
        mpo = MPO(mpo_tensors)
        # blank + title + underline + phys_up + chain + phys_down + info×2 = 8
        assert len(_lines(mpo)) == 8

    # ── line width ────────────────────────────────────────────────────────────

    def test_all_mps_lines_fit_60_chars(self, mps_tensors):
        mps = MPS(mps_tensors)
        for line in _lines(mps):
            assert len(line) <= 60, f"line too long ({len(line)}): {line!r}"

    def test_all_mpo_lines_fit_60_chars(self, mpo_tensors):
        mpo = MPO(mpo_tensors)
        for line in _lines(mpo):
            assert len(line) <= 60, f"line too long ({len(line)}): {line!r}"

    # ── node characters ───────────────────────────────────────────────────────

    def test_mps_node_char(self, mps_tensors):
        mps = MPS(mps_tensors)
        chain_row = _lines(mps)[3]  # blank, title, underline, chain
        assert '○' in chain_row

    def test_mpo_node_char(self, mpo_tensors):
        mpo = MPO(mpo_tensors)
        chain_row = _lines(mpo)[4]  # blank, title, underline, phys_up, chain
        assert '□' in chain_row

    # ── center markers ────────────────────────────────────────────────────────

    def test_mps_center_marker(self, mps_tensors):
        mps = MPS(mps_tensors, center=4)
        chain_row = _lines(mps)[3]
        assert '⊙' in chain_row

    def test_mpo_center_marker(self, mpo_tensors):
        mpo = MPO(mpo_tensors, center=4)
        chain_row = _lines(mpo)[4]
        assert '⊡' in chain_row

    def test_no_center_marker_when_unset(self, mps_tensors):
        mps = MPS(mps_tensors)
        chain_row = _lines(mps)[2]
        assert '⊙' not in chain_row

    # ── physical index characters ─────────────────────────────────────────────

    def test_mps_phys_down_row(self, mps_tensors):
        mps = MPS(mps_tensors)
        phys_row = _lines(mps)[4]
        assert '╵' in phys_row
        assert '╷' not in phys_row

    def test_mpo_has_both_phys_rows(self, mpo_tensors):
        mpo = MPO(mpo_tensors)
        lines = _lines(mpo)
        assert '╷' in lines[3]   # phys_up row
        assert '╵' in lines[5]   # phys_down row

    # ── ellipsis / truncation ─────────────────────────────────────────────────

    def test_ellipsis_when_truncated(self, mps_tensors):
        mps = MPS(mps_tensors)
        # L=10, max_sites=5 → two gaps
        assert '···' in network_summary(mps, max_sites=5)

    def test_no_ellipsis_when_fits(self, mps_tensors):
        mps = MPS(mps_tensors)
        # L=10, max_sites=10 → all sites shown
        assert '···' not in network_summary(mps, max_sites=10)

    # ── title block ───────────────────────────────────────────────────────────

    def test_title_mps(self, mps_tensors):
        mps = MPS(mps_tensors)
        assert 'Matrix Product State (MPS)' in _lines(mps)[1]

    def test_title_mpo(self, mpo_tensors):
        mpo = MPO(mpo_tensors)
        assert 'Matrix Product Operator (MPO)' in _lines(mpo)[1]

    def test_underline_char(self, mps_tensors):
        mps = MPS(mps_tensors)
        underline = _lines(mps)[2].strip()
        assert underline == '‾' * len(underline)
        assert len(underline) == len('Matrix Product State (MPS)')

    # ── info block ────────────────────────────────────────────────────────────

    def test_info_contains_keywords(self, mps_tensors):
        out = network_summary(MPS(mps_tensors))
        assert 'length:' in out
        assert 'max bond:' in out
        assert 'center:' in out
        assert 'norm:' in out

    def test_info_length_value(self, mps_tensors):
        mps = MPS(mps_tensors)
        lines = _lines(mps)
        length_line = lines[5]
        assert str(mps.L) in length_line

    def test_info_center_none(self, mps_tensors):
        mps = MPS(mps_tensors)
        assert mps.center is None
        assert 'None' in network_summary(mps)

    def test_info_center_value(self, mps_tensors):
        mps = MPS(mps_tensors, center=3)
        lines = _lines(mps)
        center_line = lines[6]
        assert '3' in center_line

    def test_info_lines_same_indent(self, mps_tensors):
        """Both info lines start at the same column (block centering)."""
        mps = MPS(mps_tensors)
        lines = _lines(mps)
        indent_length = len(lines[5]) - len(lines[5].lstrip())
        indent_center = len(lines[6]) - len(lines[6].lstrip())
        assert indent_length == indent_center


# ── TestStrRepr ───────────────────────────────────────────────────────────────

class TestStrRepr:
    """Both str() and repr() delegate to network_summary."""

    def test_mps_str_equals_summary(self, mps_tensors):
        mps = MPS(mps_tensors)
        assert str(mps) == network_summary(mps)

    def test_mpo_str_equals_summary(self, mpo_tensors):
        mpo = MPO(mpo_tensors)
        assert str(mpo) == network_summary(mpo)

    def test_mps_repr_equals_summary(self, mps_tensors):
        mps = MPS(mps_tensors)
        assert repr(mps) == network_summary(mps)

    def test_mpo_repr_equals_summary(self, mpo_tensors):
        mpo = MPO(mpo_tensors)
        assert repr(mpo) == network_summary(mpo)
