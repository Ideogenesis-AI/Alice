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


"""Tests for the iDMRG driver: options, summary, and the VUMPS run loop."""

from __future__ import annotations

import dataclasses
import math

import pytest

from alice.algorithm.idmrg import Options, Summary, run
from alice.network import iMPS

from .conftest import (
    HEISENBERG,
    MAJUMDAR_GHOSH,
    MAJUMDAR_GHOSH_SU2,
    dimer_imps,
    dimer_imps_su2,
    ihamiltonian,
    random_imps,
)

DIMER_ENERGY = -0.375

# Alice's spin operators reach Nicole as single-precision table entries
# promoted to double, which caps any U(1) spin energy at roughly eight
# digits. The SU(2) tensors avoid those tables.
TABLE_TOL = 1e-7

# The critical Heisenberg chain converges only algebraically in the bond
# dimension, so the small cells used here are nowhere near the exact value
# and are checked for ordering rather than accuracy.
HEISENBERG_EXACT = 0.25 - math.log(2.0)


def _options(**kwargs):
    """Run options tuned for a small test: loose environments, few iterations."""
    defaults = dict(
        n_iters=60, grad_tol=1e-8, env_tol=1e-10, env_maxiter=400,
    )
    defaults.update(kwargs)
    return Options(**defaults)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    """Alias resolution and defaults."""

    def test_default_scheme_is_vumps(self):
        assert Options().scheme == '1s'

    @pytest.mark.parametrize('alias', ['1s', '1-site', 'one-site'])
    def test_one_site_aliases(self, alias):
        assert Options(scheme=alias).scheme == '1s'

    @pytest.mark.parametrize(
        'alias, canonical',
        [('2-site', '2s'), ('two-site', '2s'), ('1-site-plus', '1sp')],
    )
    def test_other_schemes_resolve_but_are_not_implemented(self, alias, canonical):
        # Recognising the name and refusing to run it are separate steps, so
        # a config written for a planned scheme fails with a clear message.
        assert Options(scheme=alias).scheme == canonical

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match='unknown iDMRG scheme'):
            Options(scheme='no-such-scheme')


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    """Round-tripping the result through disk and back."""

    def test_fields_exist(self):
        names = {f.name for f in dataclasses.fields(Summary)}
        assert {
            'energy', 'state', 'energies', 'converged',
            'n_iters', 'bond_dims', 'gradients',
        } <= names

    def test_serialize_deserialize(self, spin_half):
        imps = dimer_imps(spin_half)
        summary = run(imps, ihamiltonian(MAJUMDAR_GHOSH, 2, n3x=True), _options(n_iters=2))
        restored = Summary.deserialize(summary.serialize())
        assert abs(restored.energy - summary.energy) < 1e-12
        assert restored.converged == summary.converged
        assert restored.n_iters == summary.n_iters
        assert restored.bond_dims == summary.bond_dims
        assert restored.gradients == summary.gradients
        assert isinstance(restored.state, iMPS)

    def test_deserialize_tolerates_a_missing_gradient_history(self, spin_half):
        imps = dimer_imps(spin_half)
        summary = run(imps, ihamiltonian(MAJUMDAR_GHOSH, 2, n3x=True), _options(n_iters=1))
        data = summary.serialize()
        del data['gradients']
        assert Summary.deserialize(data).gradients == []

    def test_save_load_round_trip(self, spin_half, tmp_path):
        imps = dimer_imps(spin_half)
        summary = run(imps, ihamiltonian(MAJUMDAR_GHOSH, 2, n3x=True), _options(n_iters=2))
        path = tmp_path / 'summary.pt'
        summary.save(path)
        loaded = Summary.load(path)
        assert abs(loaded.energy - summary.energy) < 1e-12
        assert loaded.n_iters == summary.n_iters
        assert isinstance(loaded.state, iMPS)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestRunValidation:
    """The driver must reject what it cannot handle, with a usable message."""

    def test_finite_state_is_rejected(self):
        from nicole import load_space

        from alice import build_hamiltonian, build_interaction
        from alice.network.automps import init_mps

        interactions, spc, geo = build_interaction({
            'geometry': {'lattice': 'chain', 'lx': 4, 'bcx': 'OBC', 'n2x': True},
            'model': HEISENBERG,
        })
        _, op = load_space('Spin', 'U1', {'J': 0.5})
        mps = init_mps(geo.L, spc, op, bond_dim=4)
        with pytest.raises(TypeError, match='use dmrg.run'):
            run(mps, build_hamiltonian(interactions, geo.L, spc))

    def test_mismatched_unit_cells_are_rejected(self, spin_half):
        imps = random_imps(spin_half, 2)
        with pytest.raises(ValueError, match='same unit cell'):
            run(imps, ihamiltonian(HEISENBERG, 4))

    def test_unimplemented_scheme_is_rejected(self, spin_half):
        imps = random_imps(spin_half, 2)
        with pytest.raises(NotImplementedError, match='not yet implemented'):
            run(imps, ihamiltonian(HEISENBERG, 2), Options(scheme='2s'))


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestRun:
    """End-to-end runs against states whose energies are known."""

    @pytest.mark.parametrize('N', [2, 4])
    def test_majumdar_ghosh_is_found_from_a_random_start(self, spin_half, N):
        # Gapped and exactly representable at a small bond dimension, so
        # this is the tight accuracy check.
        summary = run(
            random_imps(spin_half, N),
            ihamiltonian(MAJUMDAR_GHOSH, N, n3x=True),
            _options(),
        )
        assert summary.converged
        assert summary.energy == pytest.approx(DIMER_ENERGY, abs=TABLE_TOL)

    def test_majumdar_ghosh_under_su2(self, spin_half_su2):
        # The SU(2) tensors are pure Clebsch-Gordan and avoid the spin
        # tables, so the same physical check tightens by five orders. The
        # fixture starts from random entries on bond indices that leave no
        # freedom, so this is still a run and not a lookup.
        summary = run(
            dimer_imps_su2(spin_half_su2),
            ihamiltonian(MAJUMDAR_GHOSH_SU2, 2, n3x=True),
            _options(),
        )
        assert summary.converged
        assert summary.energy == pytest.approx(DIMER_ENERGY, abs=1e-11)

    def test_result_is_independent_of_the_unit_cell(self, spin_half):
        two = run(
            random_imps(spin_half, 2),
            ihamiltonian(MAJUMDAR_GHOSH, 2, n3x=True),
            _options(),
        )
        four = run(
            random_imps(spin_half, 4),
            ihamiltonian(MAJUMDAR_GHOSH, 4, n3x=True),
            _options(),
        )
        assert four.energy == pytest.approx(two.energy, abs=TABLE_TOL)

    def test_an_exact_ground_state_converges_immediately(self, spin_half):
        summary = run(
            dimer_imps(spin_half),
            ihamiltonian(MAJUMDAR_GHOSH, 2, n3x=True),
            _options(),
        )
        assert summary.converged
        assert summary.n_iters == 1

    def test_energy_and_gradient_histories_are_recorded(self, spin_half):
        summary = run(
            random_imps(spin_half, 2),
            ihamiltonian(MAJUMDAR_GHOSH, 2, n3x=True),
            _options(),
        )
        assert len(summary.energies) == summary.n_iters
        assert len(summary.gradients) == summary.n_iters
        assert summary.energy == summary.energies[-1]
        # The gradient is what the loop stops on, so it must have fallen.
        assert summary.gradients[-1] < summary.gradients[0]

    def test_bond_dimension_is_left_alone(self, spin_half):
        imps = random_imps(spin_half, 2)
        before = list(imps.bond_dims)
        summary = run(imps, ihamiltonian(MAJUMDAR_GHOSH, 2, n3x=True), _options())
        assert summary.bond_dims == before

    def test_heisenberg_improves_with_bond_dimension(self, spin_half):
        # The chain is critical, so the energy approaches the exact value
        # only algebraically. What must hold is that it approaches from
        # above and that a wider cell does better.
        narrow = run(
            random_imps(spin_half, 2, width=2, dim=1),
            ihamiltonian(HEISENBERG, 2),
            _options(n_iters=120),
        )
        wide = run(
            random_imps(spin_half, 2, width=2, dim=2),
            ihamiltonian(HEISENBERG, 2),
            _options(n_iters=120),
        )
        assert narrow.energy > HEISENBERG_EXACT
        assert wide.energy > HEISENBERG_EXACT
        assert wide.energy < narrow.energy

    def test_artifact_is_archived(self, spin_half, tmp_path):
        run(
            dimer_imps(spin_half),
            ihamiltonian(MAJUMDAR_GHOSH, 2, n3x=True),
            _options(checkpoint_dir=str(tmp_path)),
        )
        assert (tmp_path / 'artifacts' / 'state.ckpt').exists()
        # The per-iteration checkpoint is redundant once the run finishes.
        assert not (tmp_path / 'idmrg.ckpt').exists()
