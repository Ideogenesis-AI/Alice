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


"""Tests for alice.algorithm.tdvp.tdvp: Options, Summary, Artifact, and run.

These cover the driver's contract rather than the physics: option validation,
round-trips through TOML and torch serialization, the on-disk layout, and the
property that resuming from a checkpoint continues the same trajectory rather
than a nearby one.
"""

from __future__ import annotations

import copy

import pytest

from alice.algorithm import tdvp
from alice.algorithm.tdvp import Artifact, Options, Summary, run

from .conftest import mps_overlap


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    """Tests for option defaults, aliases, and validation."""

    def test_defaults(self):
        opts = Options()
        assert opts.scheme == '2s'
        assert opts.order == 2
        assert opts.dt == 0.05

    @pytest.mark.parametrize('alias,expected', [
        ('1s', '1s'), ('2s', '2s'),
        ('1-site', '1s'), ('2-site', '2s'),
        ('one-site', '1s'), ('two-site', '2s'),
        ('1S', '1s'), ('Two-Site', '2s'),
    ])
    def test_scheme_aliases(self, alias, expected):
        assert Options(scheme=alias).scheme == expected

    def test_unknown_scheme_is_rejected(self):
        with pytest.raises(ValueError):
            Options(scheme='3s')

    def test_unimplemented_scheme_is_accepted_but_not_run(self, start):
        """`1sp`/CBE is a known name, so `Options` keeps it round-trippable.

        Rejecting it only at `run` means a TOML file naming a future scheme
        still loads; the refusal arrives when it would actually be used.
        """
        assert Options(scheme='1sp').scheme == '1sp'
        state, mpo = start
        with pytest.raises(NotImplementedError):
            run(state, mpo, Options(scheme='1sp', n_steps=1))

    @pytest.mark.parametrize('order', [0, 3, -1])
    def test_bad_order_is_rejected(self, order):
        with pytest.raises(ValueError):
            Options(order=order)

    def test_zero_dt_is_rejected(self):
        """A zero step would advance nothing while still costing a sweep."""
        with pytest.raises(ValueError):
            Options(dt=0.0)

    def test_negative_dt_is_allowed(self):
        """Backward evolution is a legitimate request."""
        assert Options(dt=-0.05).dt == -0.05

    def test_bad_krylov_dim_is_rejected(self):
        with pytest.raises(ValueError):
            Options(krylov_dim=0)

    @pytest.mark.parametrize('every', [0, -1])
    def test_non_positive_save_artifacts_every_is_rejected(self, every):
        with pytest.raises(ValueError, match='save_artifacts_every'):
            Options(save_artifacts_every=every)

    def test_off_grid_n_steps_is_accepted(self):
        """The clamp lives in `run`, so `Options` itself keeps `n_steps` as given.

        Rewriting it here would make an `Options` disagree with the TOML it
        was loaded from.
        """
        assert Options(n_steps=5, save_artifacts_every=2).n_steps == 5

    def test_toml_roundtrip(self, tmp_path):
        opts = Options(scheme='1s', dt=0.02, n_steps=8, order=1, max_bond=64,
                       krylov_dim=12, renormalize=True, save_artifacts_every=2)
        path = tmp_path / 'tdvp.toml'
        opts.to_toml(path, section='tdvp')
        assert Options.load_toml(path, section='tdvp') == opts


# ---------------------------------------------------------------------------
# Summary and Artifact
# ---------------------------------------------------------------------------

class TestSummary:
    """Tests for the summary container."""

    def test_roundtrip(self, tmp_path):
        summary = Summary(
            times=[0.05, 0.10], energies=[-1.0, -1.0], norms=[1.0, 1.0],
            max_bonds=[4, 8], discarded_weights=[0.0, 1e-14],
            krylov_errors=[1e-13, 1e-13], bond_dims=[1, 4, 8, 4, 1],
            finished=True, n_steps=2,
        )
        path = tmp_path / 'summary.ckpt'
        summary.save(path)
        assert Summary.load(path) == summary

    def test_unfinished_run_is_recorded(self, tmp_path):
        """A truncated run must not masquerade as a complete one."""
        summary = Summary(finished=False, n_steps=1)
        path = tmp_path / 'summary.ckpt'
        summary.save(path)
        assert Summary.load(path).finished is False


class TestArtifact:
    """Tests for the state artifact."""

    def test_roundtrip_preserves_the_state(self, tmp_path, heisenberg_L4):
        mps, _ = heisenberg_L4
        mps.canonical(0)
        mps.normalize()
        artifact = Artifact(state=mps, time=0.25, step=5)

        path = tmp_path / 'artifact.ckpt'
        artifact.save(path)
        loaded = Artifact.load(path)

        assert loaded.time == 0.25
        assert loaded.step == 5
        assert abs(abs(mps_overlap(mps, loaded.state)) - 1.0) < 1e-12

    def test_roundtrip_preserves_the_center(self, tmp_path, heisenberg_L4):
        mps, _ = heisenberg_L4
        mps.canonical(2)
        loaded = _save_load(Artifact(state=mps, time=0.0, step=0), tmp_path)
        assert loaded.state.center == 2


def _save_load(artifact, tmp_path):
    path = tmp_path / 'rt.ckpt'
    artifact.save(path)
    return Artifact.load(path)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

@pytest.fixture
def start(heisenberg_L4):
    """A normalized L=4 Heisenberg state at t=0 with its MPO."""
    mps, mpo = heisenberg_L4
    mps.canonical(0)
    mps.normalize()
    return Artifact(state=mps, time=0.0, step=0), mpo


class TestRun:
    """Tests for the top-level driver."""

    def test_returns_a_summary_and_an_artifact(self, start):
        state, mpo = start
        summary, final = run(state, mpo, Options(n_steps=3, save_artifacts=False))
        assert isinstance(summary, Summary)
        assert isinstance(final, Artifact)

    def test_advances_the_clock(self, start):
        state, mpo = start
        opts = Options(n_steps=4, dt=0.05, save_artifacts=False)
        summary, final = run(state, mpo, opts)
        assert abs(final.time - 0.20) < 1e-12
        assert final.step == 4
        # The history opens with the initial point, so there are n_steps + 1.
        assert summary.times == pytest.approx([0.0, 0.05, 0.10, 0.15, 0.20])

    def test_records_one_entry_per_step(self, start):
        state, mpo = start
        summary, _ = run(state, mpo, Options(n_steps=3, save_artifacts=False))
        # Observables include t=0; per-step diagnostics do not exist there.
        assert len(summary.energies) == 4
        assert len(summary.norms) == 4
        assert len(summary.max_bonds) == 4
        assert len(summary.krylov_errors) == 3
        assert len(summary.discarded_weights) == 3
        assert summary.n_steps == 3
        assert summary.finished is True

    def test_conserves_energy_and_norm(self, start):
        """Real-time evolution under `H` moves neither the energy nor the norm."""
        state, mpo = start
        from alice.network import observe
        initial = observe(state.state, mpo)

        summary, _ = run(state, mpo, Options(n_steps=6, save_artifacts=False))

        assert max(abs(e - initial) for e in summary.energies) < 1e-9
        assert max(abs(n - 1.0) for n in summary.norms) < 1e-10

    @pytest.mark.parametrize('scheme', ['1s', '2s'])
    @pytest.mark.parametrize('order', [1, 2])
    def test_runs_for_every_scheme_and_order(self, start, scheme, order):
        state, mpo = start
        summary, final = run(state, mpo, Options(
            scheme=scheme, order=order, n_steps=4, save_artifacts=False))
        assert summary.finished is True
        assert final.step == 4

    def test_default_options_are_usable(self, start):
        """`run` must work without an options object at all."""
        state, mpo = start
        summary, _ = run(state, mpo)
        assert summary.n_steps == Options().n_steps

    def test_evolves_the_input_state_in_place(self, start):
        """`run` evolves in place and hands back the same MPS object.

        Pinned deliberately: an MPS at the bond dimensions TDVP reaches is
        expensive to copy, so the driver does not, and a caller who needs the
        initial state afterwards has to deep-copy it first.
        """
        state, mpo = start
        original = copy.deepcopy(state.state)
        _, final = run(state, mpo, Options(n_steps=2, save_artifacts=False))

        assert final.state is state.state
        # The state really moved: an evolved state is not the initial one.
        assert abs(mps_overlap(original, state.state)) < 0.999

    def test_zero_steps_is_a_no_op(self, start):
        state, mpo = start
        summary, final = run(state, mpo, Options(n_steps=0, save_artifacts=False))
        assert summary.n_steps == 0
        assert final.step == 0
        assert abs(final.time) < 1e-15
        assert summary.krylov_errors == []


class TestCheckpointing:
    """Tests for the on-disk layout and the resume path."""

    def test_artifacts_are_archived_per_step(self, start, tmp_path):
        state, mpo = start
        run(state, mpo, Options(n_steps=3, artifacts_dir=str(tmp_path / 'art'),
                                save_artifacts=True))
        saved = sorted((tmp_path / 'art').glob('step_*.ckpt'))
        # One per step plus the initial state at t=0.
        assert [p.name for p in saved] == [f'step_{k:05d}.ckpt' for k in range(4)]

    def test_save_artifacts_every_thins_the_archive(self, start, tmp_path):
        state, mpo = start
        run(state, mpo, Options(n_steps=6, artifacts_dir=str(tmp_path / 'art'),
                                save_artifacts=True, save_artifacts_every=2))
        saved = sorted((tmp_path / 'art').glob('step_*.ckpt'))
        assert [p.name for p in saved] == [
            'step_00000.ckpt', 'step_00002.ckpt', 'step_00004.ckpt', 'step_00006.ckpt',
        ]

    def test_off_grid_target_is_clamped_to_the_last_grid_point(
        self, start, tmp_path, caplog,
    ):
        """A run stops on the grid, so its endpoint is always archived."""
        state, mpo = start
        with caplog.at_level('WARNING'):
            summary, final = run(state, mpo, Options(
                n_steps=5, artifacts_dir=str(tmp_path / 'art'),
                save_artifacts=True, save_artifacts_every=2))

        assert summary.n_steps == 4
        assert len(summary.times) == 5
        assert final.step == 4
        saved = sorted(p.name for p in (tmp_path / 'art').glob('step_*.ckpt'))
        assert saved == ['step_00000.ckpt', 'step_00002.ckpt', 'step_00004.ckpt']
        assert 'save_artifacts_every' in caplog.text

    def test_no_clamping_without_artifacts(self, start, tmp_path):
        """Nothing is being archived, so there is no grid to land on."""
        state, mpo = start
        summary, final = run(state, mpo, Options(
            n_steps=5, artifacts_dir=str(tmp_path / 'art'),
            save_artifacts=False, save_artifacts_every=2))
        assert summary.n_steps == 5
        assert final.step == 5

    def test_resumed_run_archives_the_same_grid(self, start, tmp_path):
        """The stride keys on the absolute step, not on where the run began.

        This is what keeps a trajectory stitched from two calls on the same
        grid as one that ran straight through.
        """
        state, mpo = start
        split = tmp_path / 'split'
        whole = tmp_path / 'whole'

        _, half = run(copy.deepcopy(state), mpo, Options(
            n_steps=4, checkpoint_dir=str(split), save_artifacts_every=2))
        run(half, mpo, Options(
            n_steps=8, checkpoint_dir=str(split), save_artifacts_every=2))
        run(copy.deepcopy(state), mpo, Options(
            n_steps=8, checkpoint_dir=str(whole), save_artifacts_every=2))

        names = sorted(p.name for p in (split / 'artifacts').glob('step_*.ckpt'))
        assert names == sorted(p.name for p in (whole / 'artifacts').glob('step_*.ckpt'))
        assert names == [f'step_{k:05d}.ckpt' for k in (0, 2, 4, 6, 8)]

    def test_off_grid_start_is_not_archived(self, start, tmp_path):
        """Only the endpoint is constrained; a run may start anywhere."""
        state, mpo = start
        _, mid = run(copy.deepcopy(state), mpo, Options(
            n_steps=3, checkpoint_dir=str(tmp_path), save_artifacts=False))
        run(mid, mpo, Options(n_steps=8, checkpoint_dir=str(tmp_path),
                              save_artifacts_every=2))

        names = sorted(p.name for p in (tmp_path / 'artifacts').glob('step_*.ckpt'))
        assert names == [f'step_{k:05d}.ckpt' for k in (4, 6, 8)]

    def test_no_artifacts_when_disabled(self, start, tmp_path):
        state, mpo = start
        art = tmp_path / 'art'
        run(state, mpo, Options(n_steps=2, artifacts_dir=str(art),
                                save_artifacts=False))
        assert not art.exists() or list(art.glob('step_*.ckpt')) == []

    def test_evolution_summary_is_written(self, start, tmp_path):
        state, mpo = start
        run(state, mpo, Options(n_steps=2, checkpoint_dir=str(tmp_path),
                                save_artifacts=False))
        assert (tmp_path / 'evolution.ckpt').is_file()

    def test_resume_checkpoint_is_removed_on_success(self, start, tmp_path):
        """A completed run must not leave a resume point behind.

        Otherwise the next run would silently continue the old trajectory
        instead of starting afresh.
        """
        state, mpo = start
        run(state, mpo, Options(n_steps=2, checkpoint_dir=str(tmp_path),
                                save_artifacts=False))
        assert not (tmp_path / 'tdvp.ckpt').exists()

    def test_no_lock_files_survive(self, start, tmp_path):
        state, mpo = start
        run(state, mpo, Options(n_steps=2, checkpoint_dir=str(tmp_path),
                                save_artifacts=False))
        assert list(tmp_path.glob('*_lock.ckpt')) == []

    @pytest.mark.parametrize('order', [1, 2])
    def test_resume_reproduces_an_uninterrupted_run(self, start, tmp_path, order):
        """Splitting a run in two gives bit-for-bit the same final state.

        This is the real test of the resume path: the orthogonality center,
        the sweep direction, and the environments all have to be reconstructed
        exactly, and for `order=1` the direction depends on where the previous
        run happened to stop.
        """
        state, mpo = start

        _, direct = run(copy.deepcopy(state), mpo, Options(
            order=order, n_steps=6, save_artifacts=False))

        _, half = run(copy.deepcopy(state), mpo, Options(
            order=order, n_steps=3, save_artifacts=False,
            checkpoint_dir=str(tmp_path)))
        _, resumed = run(half, mpo, Options(
            order=order, n_steps=6, save_artifacts=False,
            checkpoint_dir=str(tmp_path)))

        assert abs(resumed.time - direct.time) < 1e-12
        assert abs(abs(mps_overlap(direct.state, resumed.state)) - 1.0) < 1e-12

    def test_resume_continues_the_history(self, start, tmp_path):
        """The resumed summary carries the earlier steps, not just the new ones."""
        state, mpo = start
        ckpt = Options(n_steps=3, save_artifacts=False,
                       checkpoint_dir=str(tmp_path))
        _, half = run(state, mpo, ckpt)
        summary, _ = run(half, mpo, Options(n_steps=6, save_artifacts=False,
                                            checkpoint_dir=str(tmp_path)))
        assert len(summary.times) == 7
        assert summary.times == pytest.approx([0.05 * i for i in range(7)])

    def test_a_state_past_the_target_is_rejected(self, start, tmp_path):
        """`n_steps` is an absolute target, so going backwards is an error."""
        state, mpo = start
        _, half = run(state, mpo, Options(n_steps=3, save_artifacts=False,
                                          checkpoint_dir=str(tmp_path)))
        with pytest.raises(ValueError):
            run(half, mpo, Options(n_steps=1, save_artifacts=False,
                                   checkpoint_dir=str(tmp_path)))


class TestComplexPromotion:
    """`run` must accept a real state and promote it itself.

    Nicole contracts only operands of a single dtype, so a real input would
    otherwise fail at the first Krylov step — a confusing error for what is a
    perfectly reasonable starting point.
    """

    def test_a_real_state_is_promoted_and_runs(self, heisenberg_L4_real):
        mps, mpo = heisenberg_L4_real
        mps.canonical(0)
        mps.normalize()
        summary, final = run(Artifact(state=mps, time=0.0, step=0), mpo,
                             Options(n_steps=2, save_artifacts=False))
        assert summary.finished is True
        assert final.state[0].dtype.is_complex


class TestFrozenBonds:
    """A `1s` run from a product state is a silent trap, so it must warn."""

    def test_warns_on_dimension_one_bonds(self, product_state_L4, caplog):
        mps, mpo = product_state_L4
        import logging
        with caplog.at_level(logging.WARNING):
            run(Artifact(state=mps, time=0.0, step=0), mpo,
                Options(scheme='1s', n_steps=1, save_artifacts=False))
        assert any('bond' in r.message.lower() for r in caplog.records)

    def test_no_warning_for_2s(self, product_state_L4, caplog):
        mps, mpo = product_state_L4
        import logging
        with caplog.at_level(logging.WARNING):
            run(Artifact(state=mps, time=0.0, step=0), mpo,
                Options(scheme='2s', n_steps=1, save_artifacts=False))
        assert not any('frozen' in r.message.lower() for r in caplog.records)


class TestPackageExports:
    """The package must expose its public surface."""

    def test_tdvp_exports(self):
        assert set(tdvp.__all__) == {'Artifact', 'Options', 'Summary', 'run'}

    def test_alice_exports_tdvp(self):
        import alice
        assert 'tdvp' in alice.__all__
        assert alice.tdvp is tdvp
