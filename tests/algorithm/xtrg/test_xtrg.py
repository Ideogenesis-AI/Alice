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

import logging
import math
import shutil as _shutil
from unittest.mock import patch

import pytest

from alice.algorithm.xtrg import Artifact, Options, Summary, run
from alice.algorithm.xtrg import xtrg as _xtrg_module
from alice.algorithm.xtrg.xtrg import _compute_observables, _fit_mpo
from alice.network.thermal import thermal_mpo


# ---------------------------------------------------------------------------
# Test helper: build the fresh-start Artifact that run() now expects
# ---------------------------------------------------------------------------

def _initial_state(mpo, spc, opts: Options) -> Artifact:
    """Build the step-0 Artifact run() expects, mirroring a fresh caller."""
    rho0 = thermal_mpo(mpo, opts.tau_0, opts.taylor_order, spc)
    return Artifact(rho=rho0, beta=opts.tau_0, step=0)


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
        assert opts.z_tol == 1e-10
        assert opts.save_artifacts is True
        assert opts.save_artifacts_since == 0

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

    def test_save_artifacts_since_negative_raises(self):
        """Negative save_artifacts_since raises ValueError."""
        with pytest.raises(ValueError, match='save_artifacts_since'):
            Options(save_artifacts_since=-1)

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

    def test_toml_roundtrip_artifact_fields(self, tmp_path):
        """save_artifacts / save_artifacts_since survive a TOML round-trip."""
        opts = Options(save_artifacts=False, save_artifacts_since=3)
        path = tmp_path / 'opts.toml'
        opts.to_toml(path, section='xtrg')
        opts2 = Options.load_toml(path, section='xtrg')
        assert opts2.save_artifacts is False
        assert opts2.save_artifacts_since == 3

    def test_artifacts_dir_defaults_to_none(self):
        """artifacts_dir defaults to None (nested under checkpoint_dir)."""
        assert Options().artifacts_dir is None

    def test_artifacts_dir_toml_round_trip(self, tmp_path):
        """artifacts_dir survives a to_toml / load_toml round trip."""
        original = Options(artifacts_dir='/tmp/artifacts')
        path = tmp_path / 'opts.toml'
        original.to_toml(path, section='xtrg')
        loaded = Options.load_toml(path, section='xtrg')
        assert loaded.artifacts_dir == '/tmp/artifacts'


# ---------------------------------------------------------------------------
# _fit_mpo: z_tol early stopping
# ---------------------------------------------------------------------------

class TestFitMpoZTol:
    """Tests for the z_tol early-stopping criterion in _fit_mpo."""

    def _rho(self, mpo, spc, opts: Options):
        """Build the same initial Taylor-expanded rho that run() would use."""
        return thermal_mpo(mpo, opts.tau_0, opts.taylor_order, spc)

    def test_loose_z_tol_stops_before_n_sweeps(self, spinless_fermion_L4):
        """A very loose z_tol should trigger early stopping well before n_sweeps."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='2s', tau_0=2 ** -6, taylor_order=10, n_sweeps=20, z_tol=1.0,
        )
        rho = self._rho(mpo, spc, opts)

        with patch.object(
            _xtrg_module, 'forward_sweep', wraps=_xtrg_module.forward_sweep,
        ) as spy:
            _fit_mpo(rho, rho, opts)

        assert spy.call_count < opts.n_sweeps

    def test_z_tol_zero_runs_full_n_sweeps(self, spinless_fermion_L4):
        """z_tol=0.0 can never be satisfied (a norm difference is always >= 0),
        so every one of the n_sweeps sweeps should run."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='2s', tau_0=2 ** -6, taylor_order=10, n_sweeps=3, z_tol=0.0,
        )
        rho = self._rho(mpo, spc, opts)

        with patch.object(
            _xtrg_module, 'forward_sweep', wraps=_xtrg_module.forward_sweep,
        ) as spy:
            _fit_mpo(rho, rho, opts)

        assert spy.call_count == opts.n_sweeps

    def test_early_stop_matches_full_sweep_result(self, spinless_fermion_L4):
        """Early-stopped fits reproduce the log Z of an equivalent full-sweep run.

        With a generous n_sweeps budget, a tight z_tol should stop once the fit
        has genuinely converged, giving thermodynamic observables indistinguishable
        from a run that never stops early (z_tol=0.0).
        """
        mpo, spc, _ = spinless_fermion_L4
        base = dict(
            scheme='2s', tau_0=2 ** -6, n_steps=4, taylor_order=10,
            max_bond=None, n_sweeps=20,
        )
        opts_early = Options(**base, z_tol=1e-10)
        opts_full = Options(**base, z_tol=0.0)
        summary_early, _ = run(_initial_state(mpo, spc, opts_early), opts_early)
        summary_full, _ = run(_initial_state(mpo, spc, opts_full), opts_full)

        for lz_early, lz_full in zip(summary_early.log_z, summary_full.log_z):
            assert math.isclose(lz_early, lz_full, rel_tol=1e-8, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

class TestComputeObservables:
    """Tests for thermodynamic observable extraction from the log-Z grid."""

    def test_specific_heat_has_thermodynamic_sign(self):
        """c_V = −β ∂u/∂(ln β) is positive when energy falls under cooling.

        A two-level system (E = 0, 1) has monotonically decreasing internal
        energy as β grows, so the finite-difference c_V must be positive at
        every interior point.
        """
        tau0 = 2 ** -4
        n_steps = 8
        betas = [tau0 * 2 ** n for n in range(n_steps + 1)]
        log_z = [math.log1p(math.exp(-beta)) for beta in betas]
        _f, energies, specific_heats, _S = _compute_observables(betas, log_z, L=1)
        n_positive = 0
        for n in range(len(betas) - 1):
            if energies[n + 1] < energies[n]:
                assert specific_heats[n] > 0
                n_positive += 1
        assert n_positive > 0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    """Tests for Summary serialize/deserialize round-trip."""

    def test_roundtrip(self, spinless_fermion_L4, tmp_path):
        """Summary survives a save → load round-trip."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=2, n_sweeps=2,
            save_artifacts=False,
        )
        summary, _artifact = run(_initial_state(mpo, spc, opts), opts)

        path = tmp_path / 'thermal.ckpt'
        summary.save(path)
        loaded = Summary.load(path)

        assert loaded.n_steps == summary.n_steps
        assert math.isclose(loaded.betas[-1], summary.betas[-1])
        assert math.isclose(loaded.log_z[-1], summary.log_z[-1], rel_tol=1e-10)
        assert loaded.finished == summary.finished
        assert 'rho' not in Summary.__dataclass_fields__

    def test_rejects_version_1(self):
        """Version-1 payloads (with rho) are rejected."""
        with pytest.raises(ValueError, match='version'):
            Summary.deserialize({'version': 1, 'betas': [], 'log_z': []})


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

class TestArtifact:
    """Tests for Artifact serialize/deserialize and on-disk archives."""

    def test_roundtrip(self, spinless_fermion_L4, tmp_path):
        """Artifact survives a save → load round-trip."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=1, n_sweeps=1,
            save_artifacts=False,
        )
        _summary, artifact = run(_initial_state(mpo, spc, opts), opts)

        path = tmp_path / 'art.ckpt'
        artifact.save(path)
        loaded = Artifact.load(path)

        assert loaded.step == artifact.step
        assert math.isclose(loaded.beta, artifact.beta)
        assert loaded.rho.L == artifact.rho.L

    def test_checkpoint_removed_after_success(self, spinless_fermion_L4, tmp_path):
        """xtrg.ckpt is deleted after a successful run."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=1, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
            save_artifacts=False,
        )
        run(_initial_state(mpo, spc, opts), opts)
        assert not (tmp_path / 'xtrg.ckpt').exists()
        assert not (tmp_path / 'xtrg_lock.ckpt').exists()

    def test_artifacts_archived_by_default(self, spinless_fermion_L4, tmp_path):
        """Default save_artifacts writes step_00 … step_n under artifacts/."""
        mpo, spc, _ = spinless_fermion_L4
        n_steps = 2
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=n_steps, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
        )
        summary, artifact = run(_initial_state(mpo, spc, opts), opts)

        for k in range(n_steps + 1):
            path = tmp_path / 'artifacts' / f'step_{k:02d}.ckpt'
            assert path.exists(), f'missing {path}'
            loaded = Artifact.load(path)
            assert loaded.step == k
            assert math.isclose(loaded.beta, summary.betas[k])

        assert artifact.step == n_steps
        assert math.isclose(artifact.beta, summary.betas[-1])

    def test_save_artifacts_since(self, spinless_fermion_L4, tmp_path):
        """Only steps >= save_artifacts_since are archived."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=3, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
            save_artifacts=True,
            save_artifacts_since=2,
        )
        run(_initial_state(mpo, spc, opts), opts)

        arts = tmp_path / 'artifacts'
        assert not (arts / 'step_00.ckpt').exists()
        assert not (arts / 'step_01.ckpt').exists()
        assert (arts / 'step_02.ckpt').exists()
        assert (arts / 'step_03.ckpt').exists()

    def test_save_artifacts_false_skips_directory(self, spinless_fermion_L4, tmp_path):
        """save_artifacts=False does not create an artifacts/ directory."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=1, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
            save_artifacts=False,
        )
        summary, artifact = run(_initial_state(mpo, spc, opts), opts)
        assert not (tmp_path / 'artifacts').exists()
        assert artifact.step == summary.n_steps

    def test_artifacts_dir_decoupled_from_checkpoint_dir(self, spinless_fermion_L4, tmp_path):
        """An explicit artifacts_dir is used verbatim, independent of checkpoint_dir."""
        mpo, spc, _ = spinless_fermion_L4
        ckpt_dir = tmp_path / 'ckpt'
        artifacts_dir = tmp_path / 'elsewhere' / 'artifacts'
        n_steps = 2
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=n_steps, n_sweeps=1,
            checkpoint_dir=str(ckpt_dir),
            artifacts_dir=str(artifacts_dir),
        )
        summary, artifact = run(_initial_state(mpo, spc, opts), opts)

        assert (ckpt_dir / 'thermal.ckpt').exists()
        assert not (ckpt_dir / 'artifacts').exists()
        for k in range(n_steps + 1):
            path = artifacts_dir / f'step_{k:02d}.ckpt'
            assert path.exists(), f'missing {path}'

        assert artifact.step == summary.n_steps


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    """Tests for the per-step thermal.ckpt writing logic."""

    def test_checkpoint_file_created(self, spinless_fermion_L4, tmp_path):
        """thermal.ckpt is written to checkpoint_dir after run()."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=2, n_sweeps=2,
            checkpoint_dir=str(tmp_path),
            save_artifacts=False,
        )
        run(_initial_state(mpo, spc, opts), opts)
        assert (tmp_path / 'thermal.ckpt').exists()

    def test_lock_file_not_present(self, spinless_fermion_L4, tmp_path):
        """thermal_lock.ckpt is renamed away on success and must not exist afterwards."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=2, n_sweeps=2,
            checkpoint_dir=str(tmp_path),
            save_artifacts=False,
        )
        run(_initial_state(mpo, spc, opts), opts)
        assert not (tmp_path / 'thermal_lock.ckpt').exists()

    def test_checkpoint_loadable(self, spinless_fermion_L4, tmp_path):
        """Checkpoint loads via Summary.load and matches the run summary."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=2, n_sweeps=2,
            checkpoint_dir=str(tmp_path),
            save_artifacts=False,
        )
        summary, _artifact = run(_initial_state(mpo, spc, opts), opts)
        loaded = Summary.load(tmp_path / 'thermal.ckpt')
        assert loaded.n_steps == summary.n_steps
        assert math.isclose(loaded.betas[-1], summary.betas[-1])
        assert math.isclose(loaded.log_z[-1], summary.log_z[-1], rel_tol=1e-10)
        assert loaded.finished is True

    def test_checkpoint_written_to_cwd_by_default(self, spinless_fermion_L4, tmp_path):
        """With checkpoint_dir=None, thermal.ckpt is written to Path.cwd()."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=1, n_sweeps=1,
            save_artifacts=False,
        )
        run(_initial_state(mpo, spc, opts), opts)
        assert (tmp_path / 'thermal.ckpt').exists()

    def test_checkpoint_written_each_step(self, spinless_fermion_L4, tmp_path):
        """Checkpoint reflects the step count of the last cooling step performed."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=3, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
            save_artifacts=False,
        )
        summary, _artifact = run(_initial_state(mpo, spc, opts), opts)
        loaded = Summary.load(tmp_path / 'thermal.ckpt')
        assert loaded.n_steps == summary.n_steps

    def test_checkpoint_dir_toml_round_trip(self, tmp_path):
        """checkpoint_dir survives a to_toml / load_toml round trip."""
        original = Options(checkpoint_dir='/tmp/ckpt')
        path = tmp_path / 'opts.toml'
        original.to_toml(path)
        loaded = Options.load_toml(path)
        assert loaded.checkpoint_dir == '/tmp/ckpt'


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

class TestResume:
    """Tests for resuming run() from a previously-checkpointed Artifact.

    run() takes state directly (mirroring DMRG's run(mps, mpo, opts)), so
    resuming is just calling run() again with an Artifact whose step > 0.
    The β/log Z history needed to continue correctly is recovered from
    thermal.ckpt in the resolved checkpoint directory, not passed in.
    """

    def test_resume_matches_uninterrupted_run(self, spinless_fermion_L4, tmp_path):
        """Interrupted (n_steps=2) + resumed (n_steps=4) matches an uninterrupted run."""
        mpo, spc, _ = spinless_fermion_L4
        base = dict(scheme='1s', tau_0=2 ** -6, taylor_order=10, n_sweeps=2)

        # Reference: uninterrupted run straight to n_steps=4.
        ref_ckpt = tmp_path / 'ref'
        ref_opts = Options(**base, n_steps=4, checkpoint_dir=str(ref_ckpt))
        reference, _ = run(_initial_state(mpo, spc, ref_opts), ref_opts)

        # Interrupted: run to n_steps=2 first (completes successfully).
        resume_ckpt = tmp_path / 'resume'
        opts2 = Options(**base, n_steps=2, checkpoint_dir=str(resume_ckpt))
        _summary2, artifact2 = run(_initial_state(mpo, spc, opts2), opts2)
        assert artifact2.step == 2

        # Resume: same checkpoint_dir, n_steps raised to 4.
        opts4 = Options(**base, n_steps=4, checkpoint_dir=str(resume_ckpt))
        resumed, _artifact4 = run(artifact2, opts4)

        assert resumed.n_steps == reference.n_steps
        for lz_resumed, lz_ref in zip(resumed.log_z, reference.log_z):
            assert math.isclose(lz_resumed, lz_ref, rel_tol=1e-10)
        for beta_resumed, beta_ref in zip(resumed.betas, reference.betas):
            assert math.isclose(beta_resumed, beta_ref)

    def test_resume_does_not_redo_earlier_steps(self, spinless_fermion_L4, tmp_path):
        """Resuming only performs the remaining squaring steps."""
        mpo, spc, _ = spinless_fermion_L4
        opts2 = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=2, n_sweeps=2,
            checkpoint_dir=str(tmp_path),
        )
        _summary2, artifact2 = run(_initial_state(mpo, spc, opts2), opts2)

        opts4 = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=4, n_sweeps=2,
            checkpoint_dir=str(tmp_path),
        )
        with patch.object(
            _xtrg_module, '_fit_mpo', wraps=_xtrg_module._fit_mpo,
        ) as spy:
            run(artifact2, opts4)

        assert spy.call_count == opts4.n_steps - artifact2.step

    def test_continue_finished_run_from_archived_artifact(
        self, spinless_fermion_L4, tmp_path,
    ):
        """A finished run continues from its archived final artifact on disk."""
        mpo, spc, _ = spinless_fermion_L4
        base = dict(scheme='1s', tau_0=2 ** -6, taylor_order=10, n_sweeps=2)

        opts2 = Options(**base, n_steps=2, checkpoint_dir=str(tmp_path))
        summary2, _artifact2 = run(_initial_state(mpo, spc, opts2), opts2)

        # Reload from disk rather than reusing the returned Artifact, as a
        # separate process continuing the run would have to.
        state = Artifact.load(tmp_path / 'artifacts' / 'step_02.ckpt')
        opts4 = Options(**base, n_steps=4, max_bond=8, checkpoint_dir=str(tmp_path))
        continued, artifact4 = run(state, opts4)

        assert continued.n_steps == 4
        assert len(continued.betas) == 5
        assert artifact4.step == 4
        # The first three grid points come from the earlier run untouched.
        for n in range(3):
            assert math.isclose(continued.betas[n], summary2.betas[n])
            assert math.isclose(continued.log_z[n], summary2.log_z[n])
        for n in range(3, 5):
            assert math.isclose(continued.betas[n], summary2.betas[-1] * 2 ** (n - 2))

    def test_continue_from_earlier_step_truncates_history(
        self, spinless_fermion_L4, tmp_path, caplog,
    ):
        """Restarting from an earlier archived step truncates and redoes the tail."""
        mpo, spc, _ = spinless_fermion_L4
        base = dict(scheme='1s', tau_0=2 ** -6, taylor_order=10, n_sweeps=2)

        opts4 = Options(**base, n_steps=4, checkpoint_dir=str(tmp_path))
        summary4, _artifact4 = run(_initial_state(mpo, spc, opts4), opts4)
        assert summary4.n_steps == 4

        # Re-cool steps 3 and 4 starting from the step-2 archive.
        state = Artifact.load(tmp_path / 'artifacts' / 'step_02.ckpt')
        with caplog.at_level(logging.WARNING, logger='alice.algorithm.xtrg.xtrg'):
            redone, artifact = run(state, opts4)

        assert redone.n_steps == 4
        assert len(redone.betas) == 5
        assert len(redone.discarded_weights) == 4
        assert artifact.step == 4
        for n in range(5):
            assert math.isclose(redone.betas[n], summary4.betas[n])
        assert any(
            record.levelno >= logging.WARNING and 'overwritten' in record.getMessage()
            for record in caplog.records
        ), 'expected a WARNING that the truncated entries are overwritten'

    def test_history_shorter_than_state_step_raises(self, spinless_fermion_L4, tmp_path):
        """A thermal.ckpt stopping before state.step raises ValueError."""
        mpo, spc, _ = spinless_fermion_L4
        opts2 = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=2, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
        )
        _summary2, artifact2 = run(_initial_state(mpo, spc, opts2), opts2)

        # thermal.ckpt reaches step 2; claim the state is at step 3.
        ahead_state = Artifact(
            rho=artifact2.rho, beta=artifact2.beta * 2, step=3,
        )
        opts_resume = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=5, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
        )
        with pytest.raises(ValueError, match='inconsistent'):
            run(ahead_state, opts_resume)

    def test_mismatched_tau_0_raises(self, spinless_fermion_L4, tmp_path):
        """Resuming with a tau_0 that differs from the history's raises ValueError."""
        mpo, spc, _ = spinless_fermion_L4
        opts2 = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=2, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
        )
        _summary2, artifact2 = run(_initial_state(mpo, spc, opts2), opts2)

        opts_resume = Options(
            scheme='1s', tau_0=2 ** -5, n_steps=4, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
        )
        with pytest.raises(ValueError, match='inconsistent'):
            run(artifact2, opts_resume)

    def test_state_step_past_n_steps_raises(self, spinless_fermion_L4, tmp_path):
        """state.step > opts.n_steps raises ValueError."""
        mpo, spc, _ = spinless_fermion_L4
        opts4 = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=4, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
        )
        _summary4, artifact4 = run(_initial_state(mpo, spc, opts4), opts4)

        opts2 = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=2, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
        )
        with pytest.raises(ValueError, match='n_steps'):
            run(artifact4, opts2)

    def test_missing_thermal_ckpt_raises(self, spinless_fermion_L4, tmp_path):
        """Resuming with step > 0 in an empty checkpoint_dir raises FileNotFoundError."""
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=1, n_sweeps=1,
        )
        state0 = _initial_state(mpo, spc, opts)
        rho1, _dw = _fit_mpo(state0.rho, state0.rho, opts)
        stale_state = Artifact(rho=rho1, beta=state0.beta * 2, step=1)

        empty_ckpt = tmp_path / 'empty'
        resume_opts = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=4, n_sweeps=1,
            checkpoint_dir=str(empty_ckpt),
        )
        with pytest.raises(FileNotFoundError, match='thermal.ckpt'):
            run(stale_state, resume_opts)

    def test_mismatched_thermal_ckpt_step_raises(self, spinless_fermion_L4, tmp_path):
        """A thermal.ckpt whose n_steps disagrees with state.step raises ValueError."""
        mpo, spc, _ = spinless_fermion_L4
        opts2 = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=2, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
        )
        _summary2, artifact2 = run(_initial_state(mpo, spc, opts2), opts2)

        # thermal.ckpt on disk reflects step 2; claim the state is at step 1.
        mismatched_state = Artifact(rho=artifact2.rho, beta=artifact2.beta, step=1)
        opts_resume = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=4, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
        )
        with pytest.raises(ValueError, match='inconsistent'):
            run(mismatched_state, opts_resume)

    def test_mismatched_thermal_ckpt_beta_raises(self, spinless_fermion_L4, tmp_path):
        """A thermal.ckpt whose last beta disagrees with state.beta raises ValueError."""
        mpo, spc, _ = spinless_fermion_L4
        opts2 = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=2, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
        )
        _summary2, artifact2 = run(_initial_state(mpo, spc, opts2), opts2)

        mismatched_state = Artifact(
            rho=artifact2.rho, beta=artifact2.beta * 3, step=artifact2.step,
        )
        opts_resume = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=4, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
        )
        with pytest.raises(ValueError, match='inconsistent'):
            run(mismatched_state, opts_resume)

    def test_already_at_target_skips_squaring(self, spinless_fermion_L4, tmp_path):
        """state.step >= opts.n_steps returns without any further squaring."""
        mpo, spc, _ = spinless_fermion_L4
        opts2 = Options(
            scheme='1s', tau_0=2 ** -6, n_steps=2, n_sweeps=1,
            checkpoint_dir=str(tmp_path),
        )
        _summary2, artifact2 = run(_initial_state(mpo, spc, opts2), opts2)

        with patch.object(
            _xtrg_module, '_fit_mpo', wraps=_xtrg_module._fit_mpo,
        ) as spy:
            resumed, resumed_artifact = run(artifact2, opts2)

        assert spy.call_count == 0
        assert resumed.n_steps == opts2.n_steps
        assert math.isclose(resumed_artifact.beta, artifact2.beta)


# ---------------------------------------------------------------------------
# Environment caching
# ---------------------------------------------------------------------------

class TestEnvCache:
    """Tests for the unique-subdirectory environment cache created by run()."""

    def test_env_cache_dir_toml_round_trip(self, tmp_path):
        """env_cache_dir survives a to_toml / load_toml round trip."""
        original = Options(env_cache_dir='/tmp/envs')
        path = tmp_path / 'opts.toml'
        original.to_toml(path)
        loaded = Options.load_toml(path)
        assert loaded.env_cache_dir == '/tmp/envs'

    def test_unique_subdir_removed_on_success(self, spinless_fermion_L4, tmp_path):
        """run() removes the unique cache subdirectory on successful completion."""
        mpo, spc, _ = spinless_fermion_L4
        cache_dir = tmp_path / 'envs'
        cache_dir.mkdir()
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=1, n_sweeps=1,
            env_cache_dir=str(cache_dir),
            checkpoint_dir=str(tmp_path / 'ckpt'),
            save_artifacts=False,
        )
        run(_initial_state(mpo, spc, opts), opts)
        subdirs = [p for p in cache_dir.iterdir() if p.is_dir()]
        assert subdirs == []

    def test_unique_subdir_created_inside_cache_dir(self, spinless_fermion_L4, tmp_path):
        """run() creates a unique subdirectory inside env_cache_dir during execution."""
        seen_subdirs: list = []
        original_rmtree = _shutil.rmtree

        def capturing_rmtree(path, **kwargs):
            seen_subdirs.append(path)
            original_rmtree(path, **kwargs)

        cache_dir = tmp_path / 'envs'
        cache_dir.mkdir()
        mpo, spc, _ = spinless_fermion_L4
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=1, n_sweeps=1,
            env_cache_dir=str(cache_dir),
            checkpoint_dir=str(tmp_path / 'ckpt'),
            save_artifacts=False,
        )
        with patch('alice.algorithm.xtrg.xtrg.shutil.rmtree', side_effect=capturing_rmtree):
            run(_initial_state(mpo, spc, opts), opts)

        assert len(seen_subdirs) == 1
        subdir = seen_subdirs[0]
        assert subdir.parent == cache_dir
        # Name is exactly 8 hex characters.
        assert len(subdir.name) == 8
        assert all(c in '0123456789abcdef' for c in subdir.name)

    def test_two_runs_use_distinct_subdirs(self, spinless_fermion_L4, tmp_path):
        """Two sequential runs with the same env_cache_dir use different subdirectories."""
        seen_subdirs: list = []
        original_rmtree = _shutil.rmtree

        def capturing_rmtree(path, **kwargs):
            seen_subdirs.append(path)
            original_rmtree(path, **kwargs)

        cache_dir = tmp_path / 'envs'
        cache_dir.mkdir()
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=1, n_sweeps=1,
            env_cache_dir=str(cache_dir),
            checkpoint_dir=str(tmp_path / 'ckpt'),
            save_artifacts=False,
        )
        mpo, spc, _ = spinless_fermion_L4
        with patch('alice.algorithm.xtrg.xtrg.shutil.rmtree', side_effect=capturing_rmtree):
            run(_initial_state(mpo, spc, opts), opts)
            run(_initial_state(mpo, spc, opts), opts)

        assert len(seen_subdirs) == 2
        assert seen_subdirs[0] != seen_subdirs[1]

    def test_no_leftover_files_after_two_runs(self, spinless_fermion_L4, tmp_path):
        """env_cache_dir has no subdirectories after two sequential runs."""
        cache_dir = tmp_path / 'envs'
        cache_dir.mkdir()
        opts = Options(
            scheme='1s', tau_0=2 ** -4, n_steps=1, n_sweeps=1,
            env_cache_dir=str(cache_dir),
            checkpoint_dir=str(tmp_path / 'ckpt'),
            save_artifacts=False,
        )
        mpo, spc, _ = spinless_fermion_L4
        run(_initial_state(mpo, spc, opts), opts)
        run(_initial_state(mpo, spc, opts), opts)
        assert list(cache_dir.iterdir()) == []

    def test_cached_run_matches_in_memory_run(self, spinless_fermion_L4, tmp_path):
        """Disk-cached environments give the same log Z as the in-memory path."""
        mpo, spc, _ = spinless_fermion_L4
        cache_dir = tmp_path / 'envs'
        cache_dir.mkdir()
        kwargs = dict(
            scheme='1s', tau_0=2 ** -4, n_steps=2, n_sweeps=2,
            checkpoint_dir=str(tmp_path / 'ckpt'),
            save_artifacts=False,
        )
        opts_plain = Options(**kwargs)
        opts_cached = Options(env_cache_dir=str(cache_dir), **kwargs)
        plain, _ = run(_initial_state(mpo, spc, opts_plain), opts_plain)
        cached, _ = run(_initial_state(mpo, spc, opts_cached), opts_cached)
        assert math.isclose(cached.log_z[-1], plain.log_z[-1], rel_tol=1e-10)


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
        summary, _artifact = run(_initial_state(mpo, spc, opts), opts)

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
        summary, _artifact = run(_initial_state(mpo, spc, opts), opts)

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
        summary, _artifact = run(_initial_state(mpo, spc, opts), opts)
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
        summary, _artifact = run(_initial_state(mpo, spc, opts), opts)

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
        summary, _artifact = run(_initial_state(mpo, spc, opts), opts)
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
        summary, _artifact = run(_initial_state(mpo, spc, opts), opts)

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
        summary, _artifact = run(_initial_state(mpo, spc, opts), opts)
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
        summary, _artifact = run(_initial_state(mpo, spc, opts), opts)

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
        summary, _artifact = run(_initial_state(mpo, spc, opts), opts)

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
        summary, _artifact = run(_initial_state(mpo, spc, opts), opts)

        beta, lz = summary.betas[-1], summary.log_z[-1]
        lz_exact = exact_log_z_fn(beta)
        rel_err = abs(lz - lz_exact) / abs(lz_exact)
        assert rel_err < 0.02, (
            f"β={beta:.4g}: XTRG log Z={lz:.8g}, exact={lz_exact:.8g}, "
            f"rel err={rel_err:.2e}"
        )
