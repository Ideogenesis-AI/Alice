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


"""Tests for alice.algorithm.xtrg.sweep."""

from __future__ import annotations

import pytest

from alice.network.thermal import NormalMPO, thermal_mpo
from alice.algorithm.xtrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
)
from alice.algorithm.xtrg.sweep import _Truncation, backward_sweep, forward_sweep
from alice.algorithm.xtrg.xtrg import Options, _fit_mpo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_compression(mpo, spc, tau_0=2 ** -8, order=4):
    """Build rho and an initialized mpo_c for sweep tests."""
    rho = thermal_mpo(mpo, tau_0, order, spc)
    mpo_c = NormalMPO.from_mpo(rho)
    mpo_c.canonical(0)
    return rho, mpo_c


def _make_envs(rho, mpo_c, scheme='1s'):
    """Build initial left boundary and all right envs for mpo_c."""
    L = rho.L
    _2s = (scheme == '2s')
    env_left = Environment(
        L, fetch_lo=0, fetch_hi=L - 2 if _2s else L - 1
    )
    env_right = Environment(
        L, fetch_lo=1 if _2s else 0, fetch_hi=L - 1
    )
    env_left[0] = left_env_boundary(rho, rho, mpo_c)
    build_right_envs(rho, rho, mpo_c, env_right)
    return env_left, env_right


def _rescaled(rho, factor):
    """Copy `rho` with every site tensor scaled by `factor`.

    Scaling the site tensors rather than `log_scale` is what makes this a test
    of the truncation scale: the sweeps read the tensor data and never the
    tracked scale, so the fitted product changes magnitude by `factor^(2L)`
    while still describing the same operator up to a constant.
    """
    return NormalMPO(
        [t * factor for t in rho],
        log_scale=rho.log_scale,
        bc=rho.bc,
        center=rho.center,
    )


def _sweep_once(rho, scheme, opts):
    """Run one full sweep on a fresh mpo_c; return its bond dims and the dw."""
    mpo_c = NormalMPO.from_mpo(rho)
    mpo_c.canonical(0)
    env_left, env_right = _make_envs(rho, mpo_c, scheme=scheme)
    forward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
    dw = backward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
    return list(mpo_c.bond_dims), dw


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForwardSweep:
    """Tests for forward_sweep."""

    def test_center_moves_to_rightmost_site_1s(self, spinless_fermion_L4):
        """After a 1s forward sweep, mpo_c.center == L-1."""
        mpo, spc, _ = spinless_fermion_L4
        rho, mpo_c = _setup_compression(mpo, spc)
        opts = Options(scheme='1s', max_bond=None)
        env_left, env_right = _make_envs(rho, mpo_c, scheme='1s')
        forward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        assert mpo_c.center == mpo_c.L - 1

    def test_center_moves_to_rightmost_site_2s(self, spinless_fermion_L4):
        """After a 2s forward sweep, mpo_c.center == L-1."""
        mpo, spc, _ = spinless_fermion_L4
        rho, mpo_c = _setup_compression(mpo, spc)
        opts = Options(scheme='2s', max_bond=None)
        env_left, env_right = _make_envs(rho, mpo_c, scheme='2s')
        forward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        assert mpo_c.center == mpo_c.L - 1

    def test_center_moves_to_rightmost_site_1sp(self, spinless_fermion_L4):
        """After a 1sp forward sweep, mpo_c.center == L-1."""
        mpo, spc, _ = spinless_fermion_L4
        rho, mpo_c = _setup_compression(mpo, spc)
        opts = Options(scheme='1sp', max_bond=8, expand_k=2, expand_alpha=2)
        env_left, env_right = _make_envs(rho, mpo_c, scheme='1sp')
        forward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        assert mpo_c.center == mpo_c.L - 1

    def test_mpo_c_tensors_updated(self, spinless_fermion_L4):
        """forward_sweep updates at least one site tensor of mpo_c."""
        mpo, spc, _ = spinless_fermion_L4
        rho, mpo_c = _setup_compression(mpo, spc)
        # Store original tensors.
        originals = [t.clone() for t in mpo_c]
        opts = Options(scheme='1s', max_bond=None)
        env_left, env_right = _make_envs(rho, mpo_c, scheme='1s')
        forward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        # At least one tensor should have changed (since rho ≠ rho²).
        # After forward_sweep, canonical() inside the sweep may also change bond
        # conventions, causing allclose to raise ValueError for incompatible
        # index structure — that counts as "changed" as well.
        from nicole import allclose
        def _tensors_differ(a, b):
            try:
                return not allclose(a, b, atol=1e-10)
            except ValueError:
                return True  # incompatible structure → definitely changed

        changed = any(_tensors_differ(mpo_c[i], originals[i]) for i in range(mpo_c.L))
        assert changed


class TestBackwardSweep:
    """Tests for backward_sweep."""

    def test_center_moves_to_site0_1s(self, spinless_fermion_L4):
        """After a 1s backward sweep, mpo_c.center == 0."""
        mpo, spc, _ = spinless_fermion_L4
        rho, mpo_c = _setup_compression(mpo, spc)
        opts = Options(scheme='1s', max_bond=None)
        env_left, env_right = _make_envs(rho, mpo_c, scheme='1s')
        forward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        backward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        assert mpo_c.center == 0

    def test_center_moves_to_site0_2s(self, spinless_fermion_L4):
        """After a 2s backward sweep, mpo_c.center == 0."""
        mpo, spc, _ = spinless_fermion_L4
        rho, mpo_c = _setup_compression(mpo, spc)
        opts = Options(scheme='2s', max_bond=None)
        env_left, env_right = _make_envs(rho, mpo_c, scheme='2s')
        forward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        backward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        assert mpo_c.center == 0

    def test_backward_returns_dw_for_2s(self, spinless_fermion_L4):
        """backward_sweep returns a non-negative discarded weight for 2s."""
        mpo, spc, _ = spinless_fermion_L4
        rho, mpo_c = _setup_compression(mpo, spc)
        opts = Options(scheme='2s', max_bond=2, trunc_thresh=1e-15)
        env_left, env_right = _make_envs(rho, mpo_c, scheme='2s')
        forward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        dw = backward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        assert dw >= 0.0

    def test_backward_returns_zero_dw_for_1s(self, spinless_fermion_L4):
        """backward_sweep returns 0.0 discarded weight for 1s scheme."""
        mpo, spc, _ = spinless_fermion_L4
        rho, mpo_c = _setup_compression(mpo, spc)
        opts = Options(scheme='1s', max_bond=None)
        env_left, env_right = _make_envs(rho, mpo_c, scheme='1s')
        forward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        dw = backward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        assert dw == 0.0

    def test_center_moves_to_site0_1sp(self, spinless_fermion_L4):
        """After a 1sp backward sweep, mpo_c.center == 0."""
        mpo, spc, _ = spinless_fermion_L4
        rho, mpo_c = _setup_compression(mpo, spc)
        opts = Options(scheme='1sp', max_bond=8, expand_k=2, expand_alpha=2)
        env_left, env_right = _make_envs(rho, mpo_c, scheme='1sp')
        forward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        backward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        assert mpo_c.center == 0

    def test_backward_returns_dw_for_1sp(self, spinless_fermion_L4):
        """backward_sweep returns a non-negative discarded weight for 1sp."""
        mpo, spc, _ = spinless_fermion_L4
        rho, mpo_c = _setup_compression(mpo, spc)
        opts = Options(
            scheme='1sp', max_bond=2, trunc_thresh=1e-15, expand_k=2, expand_alpha=2,
        )
        env_left, env_right = _make_envs(rho, mpo_c, scheme='1sp')
        forward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        dw = backward_sweep(rho, rho, mpo_c, env_left, env_right, opts)
        assert dw >= 0.0

    def test_1sp_matches_2s_accuracy_after_one_sweep(self, spinless_fermion_L4):
        """1sp and 2s traces after one full sweep agree closely (no bond truncation).

        With `max_bond=None` (no truncation), the 1s+ CBE expansion and the
        exact 2-site update should both recover essentially the same trace,
        since both explore the full 2-site variational space.
        """
        mpo, spc, _ = spinless_fermion_L4
        rho, mpo_c_1sp = _setup_compression(mpo, spc)
        mpo_c_2s = NormalMPO.from_mpo(rho)
        mpo_c_2s.canonical(0)

        opts_1sp = Options(scheme='1sp', max_bond=None, expand_k=4, expand_alpha=4)
        env_left, env_right = _make_envs(rho, mpo_c_1sp, scheme='1sp')
        forward_sweep(rho, rho, mpo_c_1sp, env_left, env_right, opts_1sp)
        backward_sweep(rho, rho, mpo_c_1sp, env_left, env_right, opts_1sp)

        opts_2s = Options(scheme='2s', max_bond=None)
        env_left, env_right = _make_envs(rho, mpo_c_2s, scheme='2s')
        forward_sweep(rho, rho, mpo_c_2s, env_left, env_right, opts_2s)
        backward_sweep(rho, rho, mpo_c_2s, env_left, env_right, opts_2s)

        trace_1sp = mpo_c_1sp.trace()
        trace_2s = mpo_c_2s.trace()
        rel_err = abs(trace_1sp - trace_2s) / abs(trace_2s)
        assert rel_err < 1e-6, (
            f"1sp trace={trace_1sp:.10g} vs. 2s trace={trace_2s:.10g}, "
            f"rel err={rel_err:.2e}"
        )


class _StubRef:
    """Stand-in for the tensor `_Truncation` measures; it only reads the norm."""

    def __init__(self, norm):
        self._norm = norm
        self.reads = 0

    def norm(self):
        self.reads += 1
        return self._norm


class TestTruncation:
    """Tests for the per-sweep resolution of the relative threshold."""

    def test_thresh_scaled_by_reference_norm(self):
        """`thresh` comes out multiplied by the norm; `nkeep` is untouched."""
        trunc = _Truncation({'thresh': 1e-8, 'nkeep': 16})
        assert trunc.at(_StubRef(4.0)) == {'thresh': 4e-8, 'nkeep': 16}

    def test_scale_fixed_on_first_use(self):
        """Later bonds reuse the first scale and never read another norm."""
        trunc = _Truncation({'thresh': 1e-8})
        first = trunc.at(_StubRef(4.0))
        later = _StubRef(1e3)
        assert trunc.at(later) == first
        assert later.reads == 0

    def test_zero_thresh_needs_no_scale(self):
        """Nothing to resolve without a relative threshold, so nothing is read."""
        trunc = _Truncation({'thresh': 0.0, 'nkeep': 8})
        ref = _StubRef(4.0)
        assert trunc.at(ref) == {'thresh': 0.0, 'nkeep': 8}
        assert ref.reads == 0

    @pytest.mark.parametrize('norm', [0.0, float('inf'), float('nan')])
    def test_degenerate_scale_drops_the_threshold(self, norm):
        """With no scale to measure against, only `nkeep` survives."""
        assert _Truncation({'thresh': 1e-8, 'nkeep': 8}).at(_StubRef(norm)) == {'nkeep': 8}
        assert _Truncation({'thresh': 1e-8}).at(_StubRef(norm)) is None


class TestTruncationScale:
    """Tests that `trunc_thresh` is measured relative to the local scale."""

    @pytest.mark.parametrize('scheme', ['1s', '2s', '1sp'])
    def test_truncation_is_scale_invariant(self, scheme, spinless_fermion_L4):
        """Rescaling the operands leaves bond dims and discarded weight alone.

        An absolute cutoff fails here: the fitted product carries a factor
        `factor^(2L)`, so a threshold that trims the tail of the spectrum at
        `factor = 1` would wipe out the whole spectrum at `factor = 1e-2`.
        Measuring the scale too early fails too, since `mpo_c` starts every
        fit at unit norm, where the rescaling leaves no trace.
        """
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _setup_compression(mpo, spc)
        opts = Options(
            scheme=scheme, max_bond=8, trunc_thresh=1e-6,
            expand_k=2, expand_alpha=2,
        )

        dims, dw = _sweep_once(rho, scheme, opts)
        dims_small, dw_small = _sweep_once(_rescaled(rho, 1e-2), scheme, opts)
        dims_large, dw_large = _sweep_once(_rescaled(rho, 1e2), scheme, opts)

        assert dims_small == dims, f"{dims_small} != {dims} under a 1e-2 rescaling"
        assert dims_large == dims, f"{dims_large} != {dims} under a 1e2 rescaling"
        assert dw_small == pytest.approx(dw, abs=1e-12)
        assert dw_large == pytest.approx(dw, abs=1e-12)

    def test_threshold_actually_truncates(self, spinless_fermion_L4):
        """A loose relative threshold cuts bonds a zero threshold would keep.

        Guards the test above against passing trivially: scale invariance is
        uninteresting if the threshold never removes anything.
        """
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _setup_compression(mpo, spc)
        base = {'scheme': '2s', 'max_bond': 16}

        dims_loose, _ = _sweep_once(rho, '2s', Options(**base, trunc_thresh=1e-3))
        dims_zero, _ = _sweep_once(rho, '2s', Options(**base, trunc_thresh=0.0))

        assert all(a <= b for a, b in zip(dims_loose, dims_zero))
        assert sum(dims_loose) < sum(dims_zero)

    def test_zero_threshold_truncates_only_by_max_bond(self, spinless_fermion_L4):
        """With `trunc_thresh = 0.0`, `max_bond` is the only limit in force."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _setup_compression(mpo, spc)
        max_bond = 4

        dims_capped, _ = _sweep_once(
            rho, '2s', Options(scheme='2s', max_bond=max_bond, trunc_thresh=0.0),
        )
        dims_free, _ = _sweep_once(
            rho, '2s', Options(scheme='2s', max_bond=None, trunc_thresh=0.0),
        )

        assert all(d <= max_bond for d in dims_capped)
        assert dims_capped == [min(d, max_bond) for d in dims_free]


class TestFullFit:
    """Integration tests for the full variational compression."""

    def test_fit_mpo_1s_trace_close_to_exact(self, spinless_fermion_L4):
        """_fit_mpo with 1s scheme gives correct log Z after 1 doubling step.

        After squaring rho(τ₀) → rho(2τ₀), the trace should match the
        exact grand-canonical partition function Z(2τ₀) for free fermions.
        """
        mpo, spc, exact_log_z_fn = spinless_fermion_L4
        import math
        tau_0 = 2 ** -6  # larger τ₀ for a faster test.
        opts = Options(scheme='1s', tau_0=tau_0, n_sweeps=6, max_bond=None)

        rho = thermal_mpo(mpo, tau_0, opts.taylor_order, spc)
        rho_sq, dw = _fit_mpo(rho, rho, opts)

        beta = 2 * tau_0
        log_z_xtrg = math.log(rho_sq.trace())
        log_z_exact = exact_log_z_fn(beta)

        rel_err = abs(log_z_xtrg - log_z_exact) / abs(log_z_exact)
        assert rel_err < 1e-3, (
            f"XTRG 1s log Z = {log_z_xtrg:.8g}, exact = {log_z_exact:.8g}, "
            f"rel err = {rel_err:.2e}"
        )

    def test_fit_mpo_2s_trace_close_to_exact(self, spinless_fermion_L4):
        """_fit_mpo with 2s scheme gives correct log Z after 1 doubling step."""
        mpo, spc, exact_log_z_fn = spinless_fermion_L4
        import math
        tau_0 = 2 ** -6
        opts = Options(scheme='2s', tau_0=tau_0, n_sweeps=4, max_bond=None)

        rho = thermal_mpo(mpo, tau_0, opts.taylor_order, spc)
        rho_sq, dw = _fit_mpo(rho, rho, opts)

        beta = 2 * tau_0
        log_z_xtrg = math.log(rho_sq.trace())
        log_z_exact = exact_log_z_fn(beta)

        rel_err = abs(log_z_xtrg - log_z_exact) / abs(log_z_exact)
        assert rel_err < 1e-3, (
            f"XTRG 2s log Z = {log_z_xtrg:.8g}, exact = {log_z_exact:.8g}, "
            f"rel err = {rel_err:.2e}"
        )

    def test_fit_mpo_1sp_trace_close_to_exact(self, spinless_fermion_L4):
        """_fit_mpo with 1sp scheme gives correct log Z after 1 doubling step."""
        mpo, spc, exact_log_z_fn = spinless_fermion_L4
        import math
        tau_0 = 2 ** -6
        opts = Options(
            scheme='1sp', tau_0=tau_0, n_sweeps=4, max_bond=None,
            expand_k=4, expand_alpha=4,
        )

        rho = thermal_mpo(mpo, tau_0, opts.taylor_order, spc)
        rho_sq, dw = _fit_mpo(rho, rho, opts)

        beta = 2 * tau_0
        log_z_xtrg = math.log(rho_sq.trace())
        log_z_exact = exact_log_z_fn(beta)

        rel_err = abs(log_z_xtrg - log_z_exact) / abs(log_z_exact)
        assert rel_err < 1e-3, (
            f"XTRG 1sp log Z = {log_z_xtrg:.8g}, exact = {log_z_exact:.8g}, "
            f"rel err = {rel_err:.2e}"
        )
