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


"""End-to-end tests for the XTRG algorithm driver."""

from __future__ import annotations

import io
import math

import pytest

from alice.algorithm.xtrg import Options, Summary, run


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    """Tests for Options."""

    def test_defaults(self):
        """Default Options can be constructed without arguments."""
        opts = Options()
        assert opts.scheme == '2s'
        assert opts.tau_0 == 2 ** -12
        assert opts.n_steps == 20
        assert opts.n_sweeps == 4

    def test_scheme_aliases(self):
        """All scheme aliases resolve to canonical names."""
        for alias in ('1s', '1-site', 'one-site'):
            assert Options(scheme=alias).scheme == '1s'
        for alias in ('2s', '2-site', 'two-site'):
            assert Options(scheme=alias).scheme == '2s'
        for alias in ('1sp', '1-site-plus', 'one-site-plus'):
            assert Options(scheme=alias).scheme == '1sp'

    def test_expand_defaults(self):
        """expand_k/expand_alpha have the documented default values."""
        opts = Options()
        assert opts.expand_k == 4
        assert opts.expand_alpha is None

    def test_unknown_scheme_raises(self):
        """Unknown scheme raises ValueError."""
        with pytest.raises(ValueError, match='scheme'):
            Options(scheme='invalid')

    def test_toml_roundtrip(self, tmp_path):
        """Options survives a TOML serialize → deserialize round-trip."""
        opts = Options(scheme='1s', tau_0=0.01, n_steps=5, max_bond=10)
        path = tmp_path / 'opts.toml'
        opts.to_toml(path, section='xtrg')
        opts2 = Options.load_toml(path, section='xtrg')
        assert opts2.scheme == '1s'
        assert math.isclose(opts2.tau_0, 0.01)
        assert opts2.n_steps == 5
        assert opts2.max_bond == 10

    def test_toml_roundtrip_1sp_expand_fields(self, tmp_path):
        """expand_k/expand_alpha survive a TOML round-trip under scheme='1sp'."""
        opts = Options(scheme='1sp', expand_k=6, expand_alpha=3)
        path = tmp_path / 'opts.toml'
        opts.to_toml(path, section='xtrg')
        opts2 = Options.load_toml(path, section='xtrg')
        assert opts2.scheme == '1sp'
        assert opts2.expand_k == 6
        assert opts2.expand_alpha == 3


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    """Tests for Summary serialize/deserialize round-trip."""

    def test_roundtrip(self, spinless_fermion_L4, tmp_path):
        """Summary survives a save → load round-trip."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(scheme='1s', tau_0=2 ** -4, n_steps=2, n_sweeps=2)
        summary = run(mpo, spc, opts)

        path = tmp_path / 'xtrg.ckpt'
        summary.save(path)
        loaded = Summary.load(path)

        assert loaded.n_steps == summary.n_steps
        assert math.isclose(loaded.betas[-1], summary.betas[-1])
        assert math.isclose(loaded.log_z[-1], summary.log_z[-1], rel_tol=1e-10)
        assert loaded.converged == summary.converged


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    """Tests for the per-step checkpoint writing logic."""

    def test_checkpoint_file_created(self, spinless_fermion_L4, tmp_path):
        """xtrg.ckpt is written to checkpoint_dir after run()."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=2, n_sweeps=2,
            checkpoint_dir=str(tmp_path),
        )
        run(mpo, spc, opts)
        assert (tmp_path / 'xtrg.ckpt').exists()

    def test_lock_file_not_present(self, spinless_fermion_L4, tmp_path):
        """xtrg_lock.ckpt is renamed away on success and must not exist afterwards."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=2, n_sweeps=2,
            checkpoint_dir=str(tmp_path),
        )
        run(mpo, spc, opts)
        assert not (tmp_path / 'xtrg_lock.ckpt').exists()

    def test_checkpoint_loadable(self, spinless_fermion_L4, tmp_path):
        """Checkpoint loads via Summary.load and matches the run summary."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=2, n_sweeps=2,
            checkpoint_dir=str(tmp_path),
        )
        summary = run(mpo, spc, opts)
        loaded = Summary.load(tmp_path / 'xtrg.ckpt')
        assert loaded.n_steps == summary.n_steps
        assert math.isclose(loaded.betas[-1], summary.betas[-1])
        assert math.isclose(loaded.log_z[-1], summary.log_z[-1], rel_tol=1e-10)

    def test_checkpoint_written_to_cwd_by_default(self, spinless_fermion_L4, tmp_path):
        """With checkpoint_dir=None, xtrg.ckpt is written to Path.cwd()."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(scheme='1s', tau_0=2 ** -4, n_steps=1, n_sweeps=1)
        run(mpo, spc, opts)
        assert (tmp_path / 'xtrg.ckpt').exists()

    def test_checkpoint_written_each_step(self, spinless_fermion_L4, tmp_path):
        """Checkpoint reflects the step count of the last cooling step performed."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=3, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
        )
        summary = run(mpo, spc, opts)
        loaded = Summary.load(tmp_path / 'xtrg.ckpt')
        assert loaded.n_steps == summary.n_steps

    def test_checkpoint_dir_toml_round_trip(self, tmp_path):
        """checkpoint_dir survives a to_toml / load_toml round trip."""
        original = Options(checkpoint_dir='/tmp/ckpt')
        path = tmp_path / 'opts.toml'
        original.to_toml(path)
        loaded = Options.load_toml(path)
        assert loaded.checkpoint_dir == '/tmp/ckpt'


# ---------------------------------------------------------------------------
# Regression: log-scale overflow
# ---------------------------------------------------------------------------

class TestXtrgLogScaleOverflow:
    """`Summary.log_z` stays finite even when `Z(β)` itself would not.

    `NormalMPO` tracks `Tr[ρ]`'s magnitude as `log_scale`, so `log Z` stays
    representable even once the raw partition function `Z` would exceed
    float64's `~1.8e308` range (i.e. `log Z ≳ 709.78`) — this regime is
    routinely reached by real XTRG runs at low temperature (e.g. β=256 for
    an L=8 spinless chain). This test drives a small L=4 chain far enough
    (β_max = 2^15 × 2^-6 = 512) that the exact `log Z` is comfortably past
    that threshold, and asserts `Summary.log_z` stays finite throughout.
    """

    def test_log_z_finite_deep_into_cooling(self, spinless_fermion_L4):
        """`log_z` stays finite even once `Z(β)` would overflow float64."""
        mpo, spc, exact_log_z_fn = spinless_fermion_L4
        opts = Options(
            scheme='1s',
            tau_0=2 ** -6,
            n_steps=15,
            taylor_order=10,
            n_sweeps=2,
        )
        summary = run(mpo, spc, opts)

        assert all(math.isfinite(lz) for lz in summary.log_z), (
            f"non-finite log_z encountered: {summary.log_z}"
        )
        # Sanity check that this run actually reaches into the regime where
        # the raw partition function Z would exceed float64 range.
        assert summary.log_z[-1] > 709.78

        # The finite log_z should still be numerically accurate. The exact
        # reference function itself uses a naive exp() that overflows at the
        # very last (most extreme) grid point, so the accuracy check uses
        # the second-to-last step (β=256), which is still safely past the
        # float64 overflow threshold but within the reference function's
        # own numerical range.
        beta_check, lz_check = summary.betas[-2], summary.log_z[-2]
        lz_exact = exact_log_z_fn(beta_check)
        rel_err = abs(lz_check - lz_exact) / abs(lz_exact)
        assert rel_err < 0.05, (
            f"β={beta_check:.4g}: XTRG log Z={lz_check:.8g}, "
            f"exact={lz_exact:.8g}, rel err={rel_err:.2e}"
        )


# ---------------------------------------------------------------------------
# End-to-end thermodynamics
# ---------------------------------------------------------------------------

class TestXtrg1s:
    """End-to-end XTRG tests with the 1-site scheme."""

    def test_log_z_matches_exact_spinless(self, spinless_fermion_L4):
        """XTRG (1s) log Z matches exact grand-canonical log Z for free fermions.

        Uses n_steps=6 (β_max ≈ 64 × 2^{-6} = 1.0) with max_bond=None for
        essentially exact compression. Tolerance: 1% relative error.
        """
        mpo, spc, exact_log_z_fn = spinless_fermion_L4
        opts = Options(
            scheme='1s',
            tau_0=2 ** -6,
            n_steps=6,
            taylor_order=10,
            max_bond=None,
            n_sweeps=4,
        )
        summary = run(mpo, spc, opts)

        for n, (beta, lz) in enumerate(zip(summary.betas, summary.log_z)):
            lz_exact = exact_log_z_fn(beta)
            rel_err = abs(lz - lz_exact) / abs(lz_exact)
            assert rel_err < 0.01, (
                f"step {n}: β={beta:.4g}, XTRG log Z={lz:.8g}, "
                f"exact={lz_exact:.8g}, rel err={rel_err:.2e}"
            )

    def test_free_energy_increasing(self, spinless_fermion_L4):
        """Free energy per site should increase (become less negative) as β grows.

        Since (∂f/∂T) = −s ≤ 0, free energy decreases with temperature, i.e.
        f(β) is a non-decreasing function of β.
        """
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(scheme='1s', tau_0=2 ** -6, n_steps=4, n_sweeps=2)
        summary = run(mpo, spc, opts)
        fs = summary.free_energies
        for i in range(len(fs) - 1):
            assert fs[i + 1] >= fs[i] - 1e-8, (
                f"Free energy not increasing: f[{i}]={fs[i]:.6g}, "
                f"f[{i+1}]={fs[i+1]:.6g}"
            )


class TestXtrg2s:
    """End-to-end XTRG tests with the 2-site scheme."""

    def test_log_z_matches_exact_spinless(self, spinless_fermion_L4):
        """XTRG (2s) log Z matches exact grand-canonical log Z for free fermions."""
        mpo, spc, exact_log_z_fn = spinless_fermion_L4
        opts = Options(
            scheme='2s',
            tau_0=2 ** -6,
            n_steps=6,
            taylor_order=10,
            max_bond=None,
            n_sweeps=4,
        )
        summary = run(mpo, spc, opts)

        for n, (beta, lz) in enumerate(zip(summary.betas, summary.log_z)):
            lz_exact = exact_log_z_fn(beta)
            rel_err = abs(lz - lz_exact) / abs(lz_exact)
            assert rel_err < 0.01, (
                f"step {n}: β={beta:.4g}, XTRG log Z={lz:.8g}, "
                f"exact={lz_exact:.8g}, rel err={rel_err:.2e}"
            )

    def test_discarded_weights_nonnegative(self, spinless_fermion_L4):
        """All discarded weights should be ≥ 0."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='2s', tau_0=2 ** -6, n_steps=4, max_bond=4, n_sweeps=2
        )
        summary = run(mpo, spc, opts)
        for dw in summary.discarded_weights:
            assert dw >= 0.0


class TestXtrg1sp:
    """End-to-end XTRG tests with the 1-site-plus (CBE) scheme."""

    def test_log_z_matches_exact_spinless(self, spinless_fermion_L4):
        """XTRG (1sp) log Z matches exact grand-canonical log Z for free fermions."""
        mpo, spc, exact_log_z_fn = spinless_fermion_L4
        opts = Options(
            scheme='1sp',
            tau_0=2 ** -6,
            n_steps=6,
            taylor_order=10,
            max_bond=None,
            n_sweeps=4,
            expand_k=4,
            expand_alpha=4,
        )
        summary = run(mpo, spc, opts)

        for n, (beta, lz) in enumerate(zip(summary.betas, summary.log_z)):
            lz_exact = exact_log_z_fn(beta)
            rel_err = abs(lz - lz_exact) / abs(lz_exact)
            assert rel_err < 0.01, (
                f"step {n}: β={beta:.4g}, XTRG log Z={lz:.8g}, "
                f"exact={lz_exact:.8g}, rel err={rel_err:.2e}"
            )

    def test_discarded_weights_nonnegative(self, spinless_fermion_L4):
        """All discarded weights should be ≥ 0."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1sp', tau_0=2 ** -6, n_steps=4, max_bond=4, n_sweeps=2,
            expand_k=2, expand_alpha=2,
        )
        summary = run(mpo, spc, opts)
        for dw in summary.discarded_weights:
            assert dw >= 0.0

    def test_expand_alpha_none_falls_back_to_exact(self, spinless_fermion_L4):
        """expand_alpha=None (skip cheap compression) still gives accurate log Z."""
        mpo, spc, exact_log_z_fn = spinless_fermion_L4
        opts = Options(
            scheme='1sp',
            tau_0=2 ** -6,
            n_steps=4,
            taylor_order=10,
            max_bond=None,
            n_sweeps=4,
            expand_k=4,
            expand_alpha=None,
        )
        summary = run(mpo, spc, opts)

        beta, lz = summary.betas[-1], summary.log_z[-1]
        lz_exact = exact_log_z_fn(beta)
        rel_err = abs(lz - lz_exact) / abs(lz_exact)
        assert rel_err < 0.01, (
            f"β={beta:.4g}, XTRG log Z={lz:.8g}, exact={lz_exact:.8g}, "
            f"rel err={rel_err:.2e}"
        )


class TestXtrgSpinful:
    """End-to-end XTRG test with the spinful (U=0 Hubbard) model."""

    @pytest.mark.slow
    def test_log_z_matches_exact(self, spinful_fermion_L4):
        """XTRG log Z matches the exact grand-canonical log Z for U=0 Hubbard."""
        mpo, spc, exact_log_z_fn = spinful_fermion_L4
        opts = Options(
            scheme='2s',
            tau_0=2 ** -6,
            n_steps=4,
            taylor_order=8,
            max_bond=None,
            n_sweeps=4,
        )
        summary = run(mpo, spc, opts)

        for n, (beta, lz) in enumerate(zip(summary.betas, summary.log_z)):
            lz_exact = exact_log_z_fn(beta)
            rel_err = abs(lz - lz_exact) / abs(lz_exact)
        assert rel_err < 0.02, (
            f"step {n}: β={beta:.4g}, XTRG log Z={lz:.8g}, "
            f"exact={lz_exact:.8g}, rel err={rel_err:.2e}"
        )

    @pytest.mark.slow
    def test_log_z_matches_exact_1sp(self, spinful_fermion_L4):
        """XTRG (1sp) log Z matches the exact grand-canonical log Z for U=0 Hubbard."""
        mpo, spc, exact_log_z_fn = spinful_fermion_L4
        opts = Options(
            scheme='1sp',
            tau_0=2 ** -6,
            n_steps=4,
            taylor_order=8,
            max_bond=None,
            n_sweeps=4,
            expand_k=4,
            expand_alpha=4,
        )
        summary = run(mpo, spc, opts)

        beta, lz = summary.betas[-1], summary.log_z[-1]
        lz_exact = exact_log_z_fn(beta)
        rel_err = abs(lz - lz_exact) / abs(lz_exact)
        assert rel_err < 0.02, (
            f"β={beta:.4g}: XTRG log Z={lz:.8g}, exact={lz_exact:.8g}, "
            f"rel err={rel_err:.2e}"
        )
