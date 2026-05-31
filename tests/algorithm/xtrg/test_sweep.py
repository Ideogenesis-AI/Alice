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
from alice.algorithm.xtrg.sweep import backward_sweep, forward_sweep
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
