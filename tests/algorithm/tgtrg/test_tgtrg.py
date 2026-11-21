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


"""Tests for alice.algorithm.tgtrg.tgtrg.

The physics is judged against the analytic grand-canonical solution of the
free-fermion chain: `log Z`, the free energy, and the internal energy at
every point of the cooling grid. On this chain the 2-site scheme reaches full
rank, so the only error left is the Taylor initialization and roundoff.

The rest covers the driver's contract: the two grid modes and their
stacking, the checkpoint and artifact layout, and the resume path.
"""

from __future__ import annotations

import copy
import math

import pytest

from alice.algorithm.tgtrg import Artifact, Options, Summary, run
from alice.algorithm.tgtrg.tgtrg import _build_grid, _derivative
from alice.network.thermal import thermal_mpo

from .conftest import exact_energy_spinless


@pytest.fixture
def start(spinless_fermion_L4):
    """A fresh `ρ(τ₀)` artifact at step 0, with its Hamiltonian."""
    mpo, spc, exact_log_z = spinless_fermion_L4
    tau_0 = 2 ** -8
    rho = thermal_mpo(mpo, tau_0, 12, spc)
    return Artifact(rho=rho, beta=tau_0, step=0), mpo, exact_log_z


def _cool(start, tmp_path, **kwargs):
    """Run a cooling job in `tmp_path` with the given option overrides."""
    state, mpo, _ = start
    kwargs.setdefault('tau_0', state.beta)
    kwargs.setdefault('n_steps', 6)
    kwargs.setdefault('save_artifacts', False)
    opts = Options(checkpoint_dir=str(tmp_path), **kwargs)
    return run(copy.deepcopy(state), mpo, opts)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    """Tests for option defaults, validation and TOML round-tripping."""

    def test_defaults(self):
        opts = Options()
        assert opts.scheme == '2s'
        assert opts.grid == 'exp'
        assert opts.order == 2
        assert opts.obs_exact is False

    @pytest.mark.parametrize('alias,expected', [
        ('1s', '1s'), ('2s', '2s'), ('1sp', '1sp'),
        ('1-site', '1s'), ('2-site', '2s'), ('1-site-plus', '1sp'),
        ('one-site', '1s'), ('two-site', '2s'), ('ONE-SITE-PLUS', '1sp'),
    ])
    def test_scheme_aliases(self, alias, expected):
        assert Options(scheme=alias).scheme == expected

    @pytest.mark.parametrize('alias,expected', [
        ('exp', 'exp'), ('exponential', 'exp'), ('doubling', 'exp'),
        ('lin', 'lin'), ('linear', 'lin'), ('UNIFORM', 'lin'),
    ])
    def test_grid_aliases(self, alias, expected):
        assert Options(grid=alias).grid == expected

    def test_unknown_scheme_is_rejected(self):
        with pytest.raises(ValueError, match='scheme'):
            Options(scheme='3s')

    def test_unknown_grid_is_rejected(self):
        with pytest.raises(ValueError, match='grid'):
            Options(grid='logarithmic')

    @pytest.mark.parametrize('order', [0, 3, -1])
    def test_bad_order_is_rejected(self, order):
        with pytest.raises(ValueError, match='order'):
            Options(order=order)

    @pytest.mark.parametrize('tau_0', [0.0, -1.0])
    def test_non_positive_tau_0_is_rejected(self, tau_0):
        with pytest.raises(ValueError, match='tau_0'):
            Options(tau_0=tau_0)

    def test_non_positive_tau_is_rejected_on_a_linear_grid(self):
        with pytest.raises(ValueError, match='tau'):
            Options(grid='lin', tau=0.0)

    def test_tau_is_ignored_on_an_exponential_grid(self):
        """The step size there is β itself, so `tau` is not consulted."""
        assert Options(grid='exp', tau=0.0).tau == 0.0

    @pytest.mark.parametrize('field,value', [
        ('n_steps', -1), ('taylor_order', 0), ('krylov_dim', 0),
        ('max_bond', 0), ('save_artifacts_every', 0),
    ])
    def test_out_of_range_fields_are_rejected(self, field, value):
        with pytest.raises(ValueError, match=field):
            Options(**{field: value})

    def test_toml_roundtrip(self, tmp_path):
        opts = Options(scheme='1s', grid='lin', tau=0.05, n_steps=8, order=1,
                       max_bond=64, krylov_dim=12, obs_exact=True,
                       save_artifacts_every=2)
        path = tmp_path / 'tgtrg.toml'
        opts.to_toml(path, section='tgtrg')
        assert Options.load_toml(path, section='tgtrg') == opts


# ---------------------------------------------------------------------------
# The β grid
# ---------------------------------------------------------------------------

class TestGrid:
    """Tests for `_build_grid`, which fixes the schedule of a run."""

    def _state(self, beta, step):
        return Artifact(rho=None, beta=beta, step=step)

    def test_exponential_grid_doubles(self):
        grid = _build_grid(self._state(1.0, 0),
                           Options(grid='exp', tau_0=1.0, n_steps=3))
        assert grid == [(2.0, 1.0), (4.0, 2.0), (8.0, 4.0)]

    def test_linear_grid_is_uniform(self):
        grid = _build_grid(self._state(1.0, 0),
                           Options(grid='lin', tau=0.5, n_steps=3))
        assert grid == [(1.5, 0.5), (2.0, 0.5), (2.5, 0.5)]

    def test_grid_covers_only_the_remaining_steps(self):
        """`n_steps` is an absolute index, so resuming shortens the grid."""
        grid = _build_grid(self._state(4.0, 2),
                           Options(grid='exp', tau_0=1.0, n_steps=4))
        assert grid == [(8.0, 4.0), (16.0, 8.0)]

    def test_grid_is_empty_at_the_target(self):
        assert _build_grid(self._state(4.0, 2),
                           Options(grid='exp', tau_0=1.0, n_steps=2)) == []

    def test_exponential_grid_checks_the_state(self):
        """β must be `2^step τ₀`, or the doubling schedule is not this state's."""
        with pytest.raises(ValueError, match='exponential grid'):
            _build_grid(self._state(3.0, 2), Options(grid='exp', tau_0=1.0))

    def test_linear_grid_starts_wherever_it_is_handed(self):
        """A linear phase continues from the β it was given, τ₀ notwithstanding."""
        grid = _build_grid(self._state(3.0, 2),
                           Options(grid='lin', tau_0=1.0, tau=1.0, n_steps=3))
        assert grid == [(4.0, 1.0)]


# ---------------------------------------------------------------------------
# Finite differences
# ---------------------------------------------------------------------------

class TestDerivative:
    """Tests for the non-uniform finite-difference helper."""

    def test_linear_function_is_exact(self):
        xs = [0.0, 0.5, 1.5, 3.5]
        ys = [3.0 * x + 1.0 for x in xs]
        assert _derivative(xs, ys) == pytest.approx([3.0] * 4)

    def test_quadratic_is_exact_in_the_interior(self):
        """The three-point formula for unequal spacing is exact on a parabola."""
        xs = [0.0, 1.0, 3.0, 7.0]
        ys = [x * x for x in xs]
        assert _derivative(xs, ys)[1:-1] == pytest.approx([2.0, 6.0])

    def test_endpoints_use_a_one_sided_difference(self):
        xs = [0.0, 1.0, 3.0]
        ys = [x * x for x in xs]
        d = _derivative(xs, ys)
        assert d[0] == pytest.approx(1.0)
        assert d[-1] == pytest.approx(4.0)

    def test_single_point_has_no_derivative(self):
        assert _derivative([1.0], [2.0]) == [0.0]

    def test_empty_grid_is_handled(self):
        assert _derivative([], []) == []


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

class TestPhysics:
    """A cooling run must reproduce the analytic solution."""

    @pytest.mark.parametrize('order', [1, 2])
    def test_log_z_is_exact_at_full_rank(self, start, tmp_path, order):
        """Every β on the grid reproduces the analytic log Z."""
        _, _, exact_log_z = start
        summary, _ = _cool(start, tmp_path, scheme='2s', order=order, n_steps=8)

        for beta, log_z in zip(summary.betas, summary.log_z):
            assert log_z == pytest.approx(exact_log_z(beta), rel=1e-9)

    def test_free_energy_follows_log_z(self, start, tmp_path):
        """`f = -log Z / (βL)`, per site."""
        _, mpo, _ = start
        summary, _ = _cool(start, tmp_path)

        for beta, log_z, f in zip(summary.betas, summary.log_z,
                                  summary.free_energies):
            assert f == pytest.approx(-log_z / (beta * mpo.L), rel=1e-12)

    def test_exact_energy_matches_the_analytic_value(self, start, tmp_path):
        """The directly measured `u` is the analytic one, free of grid error."""
        _, mpo, _ = start
        summary, _ = _cool(start, tmp_path, obs_exact=True, n_steps=8)
        reference = exact_energy_spinless(mpo.L)

        for beta, u in zip(summary.betas, summary.exact_energies):
            assert u == pytest.approx(reference(beta), rel=1e-8)

    def test_finite_difference_energy_tracks_the_exact_one(self, start, tmp_path):
        """`u` from the log Z grid agrees with the measured `u` to grid error.

        The exponential grid is coarse by construction — the spacing is β
        itself — so the agreement here is the discretization floor, not
        roundoff. That gap is exactly what `obs_exact` exists to expose.
        """
        _, _, _ = start
        summary, _ = _cool(start, tmp_path, obs_exact=True, n_steps=8)

        # Compare where the grid is fine relative to the scale of u, i.e.
        # while the chain is still hot.
        for u, u_exact in zip(summary.energies[:6], summary.exact_energies[:6]):
            assert u == pytest.approx(u_exact, abs=5e-3)

    def test_linear_grid_sharpens_the_finite_differences(self, start, tmp_path):
        """On a uniform grid the derivative error drops to the step size."""
        summary, _ = _cool(start, tmp_path, grid='lin', tau=0.05, n_steps=8,
                           obs_exact=True)

        for u, u_exact in zip(summary.energies[1:-1], summary.exact_energies[1:-1]):
            assert u == pytest.approx(u_exact, abs=1e-4)

    def test_entropy_is_the_thermodynamic_combination(self, start, tmp_path):
        """`S = β(u - f)` per site, from the recorded columns."""
        summary, _ = _cool(start, tmp_path)

        for beta, u, f, s in zip(summary.betas, summary.energies,
                                 summary.free_energies, summary.entropies):
            assert s == pytest.approx(beta * (u - f), rel=1e-12)

    def test_entropy_falls_as_the_chain_cools(self, start, tmp_path):
        """Physical sanity: cooling removes entropy."""
        summary, _ = _cool(start, tmp_path, n_steps=9)
        assert summary.entropies[-1] < summary.entropies[0]

    def test_one_site_scheme_runs_and_stays_close(self, start, tmp_path):
        """The fixed-rank scheme cools correctly to the accuracy of its rank.

        `'1s'` keeps the bond dimension of the Taylor state, which is below
        what ρ needs as the chain cools, so its error grows with β where the
        2-site scheme's does not. That gap is the reason `'2s'` is the
        default.
        """
        _, _, exact_log_z = start
        summary, _ = _cool(start, tmp_path, scheme='1s', n_steps=8)

        for beta, log_z in zip(summary.betas, summary.log_z):
            assert log_z == pytest.approx(exact_log_z(beta), rel=1e-4)
        # Still visibly worse than the adaptive scheme at the cold end.
        assert abs(summary.log_z[-1] - exact_log_z(summary.betas[-1])) > 1e-12

    def test_spinful_chain_cools_correctly(self, spinful_fermion_L4, tmp_path):
        """Two symmetry groups, and log Z is twice the spinless value."""
        mpo, spc, exact_log_z = spinful_fermion_L4
        tau_0 = 2 ** -8
        rho = thermal_mpo(mpo, tau_0, 6, spc)
        opts = Options(tau_0=tau_0, n_steps=4, checkpoint_dir=str(tmp_path),
                       save_artifacts=False)
        summary, _ = run(Artifact(rho=rho, beta=tau_0, step=0), mpo, opts)

        for beta, log_z in zip(summary.betas, summary.log_z):
            assert log_z == pytest.approx(exact_log_z(beta), rel=1e-8)


# ---------------------------------------------------------------------------
# Run mechanics
# ---------------------------------------------------------------------------

class TestRun:
    """Tests for the shape of what a run returns."""

    def test_returns_a_summary_and_an_artifact(self, start, tmp_path):
        summary, final = _cool(start, tmp_path, n_steps=2)
        assert isinstance(summary, Summary)
        assert isinstance(final, Artifact)

    def test_advances_the_grid(self, start, tmp_path):
        state, _, _ = start
        summary, final = _cool(start, tmp_path, n_steps=4)

        assert final.step == 4
        assert final.beta == pytest.approx(state.beta * 2 ** 4)
        # The history opens with the Taylor state, so there are n_steps + 1.
        assert summary.betas == pytest.approx(
            [state.beta * 2 ** k for k in range(5)])
        assert summary.n_steps == 4
        assert summary.finished is True

    def test_records_one_entry_per_step(self, start, tmp_path):
        summary, _ = _cool(start, tmp_path, n_steps=3, obs_exact=True)

        assert len(summary.betas) == 4
        assert len(summary.taus) == 4
        assert len(summary.log_z) == 4
        assert len(summary.max_bonds) == 4
        assert len(summary.exact_energies) == 4
        # Per-step diagnostics do not exist at the Taylor state.
        assert len(summary.krylov_errors) == 3
        assert len(summary.discarded_weights) == 3

    def test_taus_record_the_increments(self, start, tmp_path):
        """The first entry is zero: the Taylor state was not reached by a step."""
        state, _, _ = start
        summary, _ = _cool(start, tmp_path, n_steps=3)
        assert summary.taus == pytest.approx(
            [0.0, state.beta, 2 * state.beta, 4 * state.beta])

    def test_exact_energies_are_empty_by_default(self, start, tmp_path):
        """A run pays for nothing but the cooling unless asked."""
        summary, _ = _cool(start, tmp_path, n_steps=2)
        assert summary.exact_energies == []

    def test_cools_the_input_state_in_place(self, start, tmp_path):
        """The density matrix is cooled in place and handed back.

        Pinned deliberately: a ρ at the bond dimensions a cooling run reaches
        is expensive to copy, so the driver does not, and a caller who needs
        the initial state afterwards has to deep-copy it first.
        """
        state, mpo, _ = start
        _, final = run(state, mpo, Options(tau_0=state.beta, n_steps=2,
                                           checkpoint_dir=str(tmp_path),
                                           save_artifacts=False))
        assert final.rho is state.rho

    def test_zero_steps_is_a_no_op(self, start, tmp_path):
        state, _, _ = start
        summary, final = _cool(start, tmp_path, n_steps=0)

        assert summary.n_steps == 0
        assert final.step == 0
        assert len(summary.betas) == 1
        assert summary.krylov_errors == []

    def test_default_options_are_usable(self, spinless_fermion_L4, tmp_path,
                                        monkeypatch):
        """`run` must work without an options object at all.

        With no `checkpoint_dir` the files land in the working directory, so
        the test runs in a temporary one.
        """
        monkeypatch.chdir(tmp_path)
        mpo, spc, exact_log_z = spinless_fermion_L4
        defaults = Options()
        rho = thermal_mpo(mpo, defaults.tau_0, defaults.taylor_order, spc)

        summary, final = run(Artifact(rho=rho, beta=defaults.tau_0, step=0), mpo)

        assert summary.n_steps == defaults.n_steps
        assert final.beta == pytest.approx(defaults.tau_0 * 2 ** defaults.n_steps)
        assert summary.log_z[-1] == pytest.approx(exact_log_z(final.beta),
                                                  rel=1e-9)

    def test_exponential_grid_checks_the_handed_in_state(self, start, tmp_path):
        """A state that is not on `2^step τ₀` cannot be continued by doubling."""
        state, mpo, _ = start
        with pytest.raises(ValueError, match='exponential grid'):
            run(state, mpo, Options(tau_0=state.beta / 3, n_steps=2,
                                    checkpoint_dir=str(tmp_path)))

    def test_state_past_the_target_is_rejected(self, start, tmp_path):
        state, mpo, _ = start
        state.step = 5
        state.beta = state.beta * 2 ** 5
        with pytest.raises(ValueError, match='past the target'):
            run(state, mpo, Options(tau_0=2 ** -8, n_steps=3,
                                    checkpoint_dir=str(tmp_path)))

    def test_unimplemented_scheme_is_rejected(self, start, tmp_path):
        state, mpo, _ = start
        with pytest.raises(NotImplementedError, match='1sp'):
            run(state, mpo, Options(scheme='1sp', tau_0=state.beta,
                                    checkpoint_dir=str(tmp_path)))

    def test_max_bond_caps_the_density_matrix(self, start, tmp_path):
        """The cap holds from the handed-in state onward, not just at splits."""
        summary, _ = _cool(start, tmp_path, n_steps=6, max_bond=6)
        assert max(summary.max_bonds) <= 6

    def test_max_bond_fixes_the_rank_of_the_one_site_scheme(self, start, tmp_path):
        """`'1s'` cannot change a bond, so the initial compaction is the cap."""
        summary, _ = _cool(start, tmp_path, scheme='1s', n_steps=4, max_bond=4)
        assert set(summary.max_bonds) == {4}

    def test_large_step_is_warned_about(self, start, tmp_path, caplog):
        """The exponential grid walks into its own step-size limit silently.

        Nothing downstream flags it — the Krylov expansions converge — so the
        warning is the only notice a user gets.
        """
        with caplog.at_level('WARNING'):
            _cool(start, tmp_path, n_steps=14)
        assert 'past the soft limit' in caplog.text

    def test_a_moderate_schedule_is_not_warned_about(self, start, tmp_path,
                                                     caplog):
        with caplog.at_level('WARNING'):
            _cool(start, tmp_path, n_steps=6)
        assert 'soft limit' not in caplog.text

    def test_krylov_errors_stay_small(self, start, tmp_path):
        summary, _ = _cool(start, tmp_path, n_steps=6)
        assert max(summary.krylov_errors) < 1e-6


# ---------------------------------------------------------------------------
# Stacked schedules
# ---------------------------------------------------------------------------

class TestStackedSchedule:
    """Exponential down to the crossover, then linear — the paper's recipe."""

    def test_linear_phase_continues_the_exponential_one(self, start, tmp_path):
        """A continuation keeps one history, one step counter, one grid."""
        _, mpo, exact_log_z = start

        hot, mid = _cool(start, tmp_path, n_steps=6)
        cold, final = run(mid, mpo, Options(
            grid='lin', tau=0.5, n_steps=9,
            checkpoint_dir=str(tmp_path), save_artifacts=False))

        assert final.step == 9
        assert cold.betas[:7] == pytest.approx(hot.betas)
        assert cold.betas[7:] == pytest.approx(
            [mid.beta + 0.5 * k for k in (1, 2, 3)])
        # The phase boundary shows up in `taus`, not in a break in the physics.
        assert cold.taus[7:] == pytest.approx([0.5, 0.5, 0.5])
        for beta, log_z in zip(cold.betas, cold.log_z):
            assert log_z == pytest.approx(exact_log_z(beta), rel=1e-8)

    def test_observables_cross_the_phase_boundary(self, start, tmp_path):
        """The finite differences handle the change of spacing in stride."""
        _, mpo, _ = start
        reference = exact_energy_spinless(mpo.L)

        _, mid = _cool(start, tmp_path, n_steps=6, obs_exact=True)
        cold, _ = run(mid, mpo, Options(
            grid='lin', tau=0.1, n_steps=12,
            obs_exact=True, checkpoint_dir=str(tmp_path), save_artifacts=False))

        for beta, u in zip(cold.betas, cold.exact_energies):
            assert u == pytest.approx(reference(beta), rel=1e-7)
        # The energy from the grid is continuous across the boundary.
        assert cold.energies[8] == pytest.approx(cold.exact_energies[8], abs=5e-2)


# ---------------------------------------------------------------------------
# Checkpoints and artifacts
# ---------------------------------------------------------------------------

class TestCheckpointing:
    """Tests for the on-disk layout and the resume path."""

    def test_cooling_checkpoint_is_written(self, start, tmp_path):
        summary, _ = _cool(start, tmp_path, n_steps=3)
        reloaded = Summary.load(tmp_path / 'thermal.ckpt')

        assert reloaded.betas == pytest.approx(summary.betas)
        assert reloaded.log_z == pytest.approx(summary.log_z)
        assert reloaded.finished is True

    def test_mid_run_checkpoint_is_removed_on_success(self, start, tmp_path):
        _cool(start, tmp_path, n_steps=2)
        assert not (tmp_path / 'tgtrg.ckpt').exists()

    def test_artifacts_are_archived_per_step(self, start, tmp_path):
        _cool(start, tmp_path, n_steps=3, save_artifacts=True)
        saved = sorted(p.name for p in (tmp_path / 'artifacts').glob('step_*.ckpt'))
        assert saved == [f'step_{k:05d}.ckpt' for k in range(4)]

    def test_stride_thins_the_archive(self, start, tmp_path):
        _cool(start, tmp_path, n_steps=6, save_artifacts=True,
              save_artifacts_every=2)
        saved = sorted(p.name for p in (tmp_path / 'artifacts').glob('step_*.ckpt'))
        assert saved == [f'step_{k:05d}.ckpt' for k in (0, 2, 4, 6)]

    def test_final_step_is_always_archived(self, start, tmp_path):
        """Off-stride endpoints are archived anyway, so a run stays resumable."""
        _cool(start, tmp_path, n_steps=5, save_artifacts=True,
              save_artifacts_every=2)
        saved = sorted(p.name for p in (tmp_path / 'artifacts').glob('step_*.ckpt'))
        assert saved == [f'step_{k:05d}.ckpt' for k in (0, 2, 4, 5)]

    def test_artifacts_dir_can_be_pointed_elsewhere(self, start, tmp_path):
        elsewhere = tmp_path / 'elsewhere'
        state, mpo, _ = start
        run(state, mpo, Options(tau_0=state.beta, n_steps=2,
                                checkpoint_dir=str(tmp_path),
                                artifacts_dir=str(elsewhere),
                                save_artifacts=True))
        assert list(elsewhere.glob('step_*.ckpt'))
        assert not (tmp_path / 'artifacts').exists()

    def test_artifact_round_trips(self, start, tmp_path):
        _, _, exact_log_z = start
        _, final = _cool(start, tmp_path, n_steps=3, save_artifacts=True)
        reloaded = Artifact.load(tmp_path / 'artifacts' / 'step_00003.ckpt')

        assert reloaded.step == final.step
        assert reloaded.beta == pytest.approx(final.beta)
        log_z, _ = reloaded.rho.log_trace()
        assert log_z == pytest.approx(exact_log_z(final.beta), rel=1e-9)

    def test_resumed_run_matches_an_uninterrupted_one(self, start, tmp_path):
        """Splitting a run in two changes nothing about its history."""
        state, mpo, _ = start
        split, whole = tmp_path / 'split', tmp_path / 'whole'

        _, half = run(copy.deepcopy(state), mpo, Options(
            tau_0=state.beta, n_steps=3, checkpoint_dir=str(split),
            save_artifacts=False))
        resumed, _ = run(half, mpo, Options(
            tau_0=state.beta, n_steps=7, checkpoint_dir=str(split),
            save_artifacts=False))
        direct, _ = run(copy.deepcopy(state), mpo, Options(
            tau_0=state.beta, n_steps=7, checkpoint_dir=str(whole),
            save_artifacts=False))

        assert resumed.betas == pytest.approx(direct.betas)
        assert resumed.log_z == pytest.approx(direct.log_z, rel=1e-10)
        assert resumed.n_steps == direct.n_steps

    def test_resume_requires_the_history(self, start, tmp_path):
        state, mpo, _ = start
        state.step = 2
        state.beta = state.beta * 4
        with pytest.raises(FileNotFoundError, match='thermal.ckpt'):
            run(state, mpo, Options(tau_0=2 ** -8, n_steps=4,
                                    checkpoint_dir=str(tmp_path / 'empty')))

    def test_resume_rejects_a_mismatched_history(self, start, tmp_path):
        """The β at the state's own step has to be the state's β."""
        state, mpo, _ = start
        _, mid = _cool(start, tmp_path, n_steps=3)
        mid.beta *= 1.5
        with pytest.raises(ValueError, match='does not match'):
            run(mid, mpo, Options(grid='lin', tau=0.1, n_steps=5,
                                  checkpoint_dir=str(tmp_path)))

    def test_resume_rejects_a_history_that_stops_too_early(self, start, tmp_path):
        state, mpo, _ = start
        _, mid = _cool(start, tmp_path, n_steps=2)
        mid.step = 5
        mid.beta = state.beta * 2 ** 5
        with pytest.raises(ValueError, match='only reaches step'):
            run(mid, mpo, Options(tau_0=state.beta, n_steps=7,
                                  checkpoint_dir=str(tmp_path)))

    def test_restarting_an_earlier_step_truncates_the_history(self, start,
                                                              tmp_path, caplog):
        """Re-cooling a segment overwrites the entries past the restart point."""
        state, mpo, _ = start
        _cool(start, tmp_path, n_steps=6, save_artifacts=True)
        earlier = Artifact.load(tmp_path / 'artifacts' / 'step_00003.ckpt')

        with caplog.at_level('WARNING'):
            summary, final = run(earlier, mpo, Options(
                tau_0=state.beta, n_steps=5, checkpoint_dir=str(tmp_path),
                save_artifacts=False))

        assert 'will be recomputed' in caplog.text
        assert final.step == 5
        assert len(summary.betas) == 6

    def test_exact_energies_are_padded_when_the_history_lacks_them(
        self, start, tmp_path, caplog
    ):
        """Turning `obs_exact` on mid-run backfills NaN, and says so."""
        state, mpo, _ = start
        _, mid = _cool(start, tmp_path, n_steps=3, obs_exact=False)

        with caplog.at_level('WARNING'):
            summary, _ = run(mid, mpo, Options(
                tau_0=state.beta, n_steps=5, obs_exact=True,
                checkpoint_dir=str(tmp_path), save_artifacts=False))

        assert 'were not measured' in caplog.text
        assert len(summary.exact_energies) == 6
        assert all(math.isnan(u) for u in summary.exact_energies[:4])
        assert not any(math.isnan(u) for u in summary.exact_energies[4:])

    def test_environment_disk_cache_is_cleaned_up(self, start, tmp_path):
        """The run-specific cache subdirectory does not outlive the run."""
        cache = tmp_path / 'cache'
        _cool(start, tmp_path, n_steps=3, env_cache_dir=str(cache))
        assert list(cache.iterdir()) == []

    def test_results_are_unchanged_by_the_disk_cache(self, start, tmp_path):
        """Caching environments is an I/O decision, not a numerical one."""
        cached, _ = _cool(start, tmp_path / 'a', n_steps=4,
                          env_cache_dir=str(tmp_path / 'cache'))
        memory, _ = _cool(start, tmp_path / 'b', n_steps=4)
        assert cached.log_z == pytest.approx(memory.log_z, rel=1e-12)
