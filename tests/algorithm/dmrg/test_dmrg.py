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


"""Tests for alice.algorithm.dmrg.dmrg (Options, Summary, dmrg())."""

from __future__ import annotations

import pytest

from alice.algorithm.dmrg import Options, Summary, run
from alice.network import MPS


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    """Tests for the Options dataclass."""

    def test_default_scheme(self):
        assert Options().scheme == '1s'

    def test_scheme_alias_1site(self):
        assert Options(scheme='1-site').scheme == '1s'

    def test_scheme_alias_oneside(self):
        assert Options(scheme='one-site').scheme == '1s'

    def test_scheme_alias_2s(self):
        assert Options(scheme='2s').scheme == '2s'

    def test_scheme_alias_2site(self):
        assert Options(scheme='2-site').scheme == '2s'

    def test_scheme_alias_twosite(self):
        assert Options(scheme='two-site').scheme == '2s'

    def test_scheme_alias_1sp(self):
        assert Options(scheme='1sp').scheme == '1sp'

    def test_scheme_alias_1siteplus(self):
        assert Options(scheme='1-site-plus').scheme == '1sp'

    def test_scheme_alias_onesiteplus(self):
        assert Options(scheme='one-site-plus').scheme == '1sp'

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="unknown DMRG scheme"):
            Options(scheme='banana')

    def test_from_toml_basic(self):
        section = {
            'scheme': '1s',
            'n_sweeps': 5,
            'max_bond': 50,
            'trunc_thresh': 1e-12,
            'davidson_tol': 1e-8,
            'e_tol': 1e-7,
        }
        opts = Options.from_toml(section)
        assert opts.scheme == '1s'
        assert opts.n_sweeps == 5
        assert opts.max_bond == 50
        assert abs(opts.trunc_thresh - 1e-12) < 1e-20
        assert abs(opts.davidson_tol - 1e-8) < 1e-20

    def test_from_toml_with_alias(self):
        """from_toml resolves scheme aliases correctly."""
        opts = Options.from_toml({'scheme': 'one-site', 'n_sweeps': 3})
        assert opts.scheme == '1s'
        assert opts.n_sweeps == 3

    def test_from_toml_ignores_unknown_keys(self):
        """Extra keys in the TOML section are silently ignored."""
        opts = Options.from_toml({'n_sweeps': 2, 'some_future_option': 'value'})
        assert opts.n_sweeps == 2

    def test_from_toml_defaults_for_missing_keys(self):
        """Missing keys in the TOML section fall back to dataclass defaults."""
        opts = Options.from_toml({})
        assert opts.scheme == '1s'
        assert opts.n_sweeps == 10

    def test_to_toml_round_trip(self, tmp_path):
        """to_toml followed by load_toml restores all fields."""
        original = Options(scheme='1s', n_sweeps=7, trunc_thresh=1e-12, e_tol=1e-9)
        path = tmp_path / 'opts.toml'
        original.to_toml(path)
        loaded = Options.load_toml(path)
        assert loaded.scheme == '1s'
        assert loaded.n_sweeps == 7
        assert abs(loaded.trunc_thresh - 1e-12) < 1e-20
        assert abs(loaded.e_tol - 1e-9) < 1e-20

    def test_to_toml_with_section(self, tmp_path):
        """Options written under a section header can be loaded back."""
        path = tmp_path / 'config.toml'
        Options(n_sweeps=3).to_toml(path, section='algorithm')
        loaded = Options.load_toml(path, section='algorithm')
        assert loaded.n_sweeps == 3

    def test_to_toml_skips_max_bond_none(self, tmp_path):
        """max_bond=None is not written to the TOML file."""
        path = tmp_path / 'opts.toml'
        Options(max_bond=None).to_toml(path)
        content = path.read_text()
        assert 'max_bond' not in content


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    """Tests for the Summary dataclass."""

    def test_summary_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(Summary)

    def test_summary_fields_exist(self):
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(Summary)}
        assert {
            'energy', 'state', 'energies', 'converged',
            'n_sweeps', 'bond_dims', 'discarded_weights',
        } <= field_names

    def test_serialize_deserialize(self, heisenberg_L2):
        """serialize/deserialize round-trips scalar fields and restores an MPS."""
        mps, mpo = heisenberg_L2
        summary = run(mps, mpo, Options(n_sweeps=2))
        data = summary.serialize()
        assert isinstance(data, dict)
        restored = Summary.deserialize(data)
        assert abs(restored.energy - summary.energy) < 1e-12
        assert restored.converged == summary.converged
        assert restored.n_sweeps == summary.n_sweeps
        assert restored.bond_dims == summary.bond_dims
        assert restored.discarded_weights == summary.discarded_weights
        assert isinstance(restored.state, MPS)

    def test_serialize_deserialize_2s_discarded_weights(self, heisenberg_L2):
        """serialize/deserialize round-trips discarded_weights for a 2-site run."""
        mps, mpo = heisenberg_L2
        summary = run(mps, mpo, Options(scheme='2s', n_sweeps=2))
        data = summary.serialize()
        restored = Summary.deserialize(data)
        assert restored.discarded_weights == summary.discarded_weights

    def test_deserialize_backward_compat_missing_discarded_weights(self, heisenberg_L2):
        """deserialize tolerates an old dict that lacks 'discarded_weights'."""
        mps, mpo = heisenberg_L2
        summary = run(mps, mpo, Options(n_sweeps=1))
        data = summary.serialize()
        del data['discarded_weights']
        restored = Summary.deserialize(data)
        assert restored.discarded_weights == []

    def test_save_load_round_trip(self, heisenberg_L2, tmp_path):
        """save then load restores all scalar fields and produces an MPS."""
        mps, mpo = heisenberg_L2
        summary = run(mps, mpo, Options(n_sweeps=2))
        path = tmp_path / 'summary.pt'
        summary.save(path)
        loaded = Summary.load(path)
        assert abs(loaded.energy - summary.energy) < 1e-12
        assert loaded.converged == summary.converged
        assert loaded.n_sweeps == summary.n_sweeps
        assert loaded.bond_dims == summary.bond_dims
        assert isinstance(loaded.state, MPS)


# ---------------------------------------------------------------------------
# dmrg() — integration tests
# ---------------------------------------------------------------------------

class TestDmrg:
    """Integration tests for the run() function (both 1-site and 2-site schemes)."""

    def test_unrecognised_scheme_raises(self, heisenberg_L2):
        with pytest.raises(ValueError, match="unknown DMRG scheme"):
            Options(scheme='unknown-scheme')

    def test_length_mismatch_raises(self, heisenberg_L2, heisenberg_L4):
        mps2, _ = heisenberg_L2
        _, mpo4 = heisenberg_L4
        with pytest.raises(ValueError, match="same length"):
            run(mps2, mpo4)

    def test_default_opts(self, heisenberg_L2):
        """run() with opts=None uses the default Options()."""
        mps, mpo = heisenberg_L2
        summary = run(mps, mpo)
        assert isinstance(summary, Summary)
        assert summary.energy < 0  # Heisenberg ground state is negative

    def test_heisenberg_L2_energy_1s(self, heisenberg_L2):
        """1-site DMRG on L=2 Heisenberg chain recovers the exact energy -0.75."""
        mps, mpo = heisenberg_L2
        opts = Options(n_sweeps=10, davidson_tol=1e-12, e_tol=1e-10)
        summary = run(mps, mpo, opts)
        assert isinstance(summary, Summary)
        assert abs(summary.energy - (-0.75)) < 1e-6, (
            f"L=2 Heisenberg energy {summary.energy} != -0.75"
        )

    def test_heisenberg_L2_converges_1s(self, heisenberg_L2):
        """1-site DMRG converges within the sweep budget for L=2."""
        mps, mpo = heisenberg_L2
        opts = Options(n_sweeps=20, e_tol=1e-10)
        summary = run(mps, mpo, opts)
        assert summary.converged

    def test_heisenberg_L2_summary_fields_1s(self, heisenberg_L2):
        """1-site Summary contains a non-empty energy history and correct bond_dims."""
        mps, mpo = heisenberg_L2
        summary = run(mps, mpo, Options(n_sweeps=4))
        assert len(summary.energies) >= 1
        assert len(summary.bond_dims) == mps.L - 1
        assert summary.n_sweeps >= 1

    def test_heisenberg_L2_discarded_weights_1s(self, heisenberg_L2):
        """1-site Summary discarded_weights has one entry per sweep, all zero."""
        mps, mpo = heisenberg_L2
        n = 3
        summary = run(mps, mpo, Options(n_sweeps=n))
        assert len(summary.discarded_weights) == summary.n_sweeps
        assert all(dw == 0.0 for dw in summary.discarded_weights)

    @pytest.mark.slow
    def test_heisenberg_L4_energy_1s(self, heisenberg_L4):
        """1-site DMRG on L=4 Heisenberg chain recovers the exact energy ≈ -1.6160254."""
        mps, mpo = heisenberg_L4
        opts = Options(n_sweeps=30, davidson_tol=1e-12, e_tol=1e-10)
        summary = run(mps, mpo, opts)
        # Exact ground-state energy for L=4 spin-1/2 Heisenberg OBC.
        E_exact = -1.6160254037844385
        assert abs(summary.energy - E_exact) < 1e-5, (
            f"L=4 Heisenberg energy {summary.energy} != {E_exact}"
        )

    def test_heisenberg_L2_energy_2s(self, heisenberg_L2):
        """2-site DMRG on L=2 Heisenberg chain recovers the exact energy -0.75."""
        mps, mpo = heisenberg_L2
        opts = Options(scheme='2s', n_sweeps=10, davidson_tol=1e-12, e_tol=1e-10)
        summary = run(mps, mpo, opts)
        assert isinstance(summary, Summary)
        assert abs(summary.energy - (-0.75)) < 1e-6, (
            f"L=2 Heisenberg 2-site energy {summary.energy} != -0.75"
        )

    def test_heisenberg_L2_converges_2s(self, heisenberg_L2):
        """2-site DMRG converges within the sweep budget for L=2."""
        mps, mpo = heisenberg_L2
        opts = Options(scheme='2s', n_sweeps=20, e_tol=1e-10)
        summary = run(mps, mpo, opts)
        assert summary.converged

    def test_heisenberg_L2_summary_fields_2s(self, heisenberg_L2):
        """2-site Summary contains a non-empty energy history and correct bond_dims."""
        mps, mpo = heisenberg_L2
        summary = run(mps, mpo, Options(scheme='2s', n_sweeps=4))
        assert len(summary.energies) >= 1
        assert len(summary.bond_dims) == mps.L - 1
        assert summary.n_sweeps >= 1

    def test_heisenberg_L2_discarded_weights_2s(self, heisenberg_L2):
        """2-site Summary discarded_weights has one non-negative entry per sweep."""
        mps, mpo = heisenberg_L2
        summary = run(mps, mpo, Options(scheme='2s', n_sweeps=4))
        assert len(summary.discarded_weights) == summary.n_sweeps
        assert all(dw >= 0.0 for dw in summary.discarded_weights)

    @pytest.mark.slow
    def test_heisenberg_L4_energy_2s(self, heisenberg_L4):
        """2-site DMRG on L=4 Heisenberg chain recovers the exact energy ≈ -1.6160254."""
        mps, mpo = heisenberg_L4
        opts = Options(scheme='2s', n_sweeps=30, davidson_tol=1e-12, e_tol=1e-10)
        summary = run(mps, mpo, opts)
        E_exact = -1.6160254037844385
        assert abs(summary.energy - E_exact) < 1e-5, (
            f"L=4 Heisenberg 2-site energy {summary.energy} != {E_exact}"
        )


# ---------------------------------------------------------------------------
# Checkpointing Functionality
# ---------------------------------------------------------------------------

class TestCheckpoint:
    """Tests for the per-sweep checkpoint writing logic."""

    def test_checkpoint_file_created(self, heisenberg_L2, tmp_path):
        """dmrg.ckpt is written to checkpoint_dir after run()."""
        mps, mpo = heisenberg_L2
        run(mps, mpo, Options(n_sweeps=2, checkpoint_dir=str(tmp_path)))
        assert (tmp_path / 'dmrg.ckpt').exists()

    def test_lock_file_not_present(self, heisenberg_L2, tmp_path):
        """dmrg_lock.ckpt is renamed away on success and must not exist afterwards."""
        mps, mpo = heisenberg_L2
        run(mps, mpo, Options(n_sweeps=2, checkpoint_dir=str(tmp_path)))
        assert not (tmp_path / 'dmrg_lock.ckpt').exists()

    def test_checkpoint_loadable(self, heisenberg_L2, tmp_path):
        """Checkpoint loads correctly and energy matches the run summary."""
        mps, mpo = heisenberg_L2
        summary = run(mps, mpo, Options(n_sweeps=2, checkpoint_dir=str(tmp_path)))
        loaded = Summary.load(tmp_path / 'dmrg.ckpt')
        assert abs(loaded.energy - summary.energy) < 1e-12
        assert loaded.n_sweeps == summary.n_sweeps

    def test_checkpoint_written_to_cwd_by_default(self, heisenberg_L2, tmp_path):
        """With checkpoint_dir=None, dmrg.ckpt is written to Path.cwd()."""
        mps, mpo = heisenberg_L2
        run(mps, mpo, Options(n_sweeps=1))
        assert (tmp_path / 'dmrg.ckpt').exists()

    def test_checkpoint_written_each_sweep(self, heisenberg_L2, tmp_path):
        """Checkpoint reflects the sweep count of the last sweep performed."""
        mps, mpo = heisenberg_L2
        summary = run(mps, mpo, Options(n_sweeps=3, checkpoint_dir=str(tmp_path)))
        loaded = Summary.load(tmp_path / 'dmrg.ckpt')
        assert loaded.n_sweeps == summary.n_sweeps

    def test_checkpoint_dir_toml_round_trip(self, tmp_path):
        """checkpoint_dir survives a to_toml / load_toml round trip."""
        original = Options(checkpoint_dir='/tmp/ckpt')
        path = tmp_path / 'opts.toml'
        original.to_toml(path)
        loaded = Options.load_toml(path)
        assert loaded.checkpoint_dir == '/tmp/ckpt'
