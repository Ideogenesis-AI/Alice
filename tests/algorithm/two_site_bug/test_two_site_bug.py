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


"""Tests for the gate-based two-site BUG integrator (Options, Summary, run)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from alice import init_mps
from alice.algorithm import two_site_bug
from alice.algorithm.two_site_bug.gate import build_bond_generators
from alice.network.interaction import Interaction2Site

from .conftest import (
    dense_heisenberg,
    dense_total_sz,
    exact_evolve,
    heisenberg_chain,
    mps_to_vector,
    product_vector,
)


def _domain_wall(length, spin_space):
    """Return `(mps, interactions, charges, config)` for a Heisenberg domain wall."""
    spc_index, operators = spin_space
    charges = [sector.charge for sector in spc_index.sectors]
    interactions, spc, _ = heisenberg_chain(length)
    config = [0] * (length // 2) + [1] * (length - length // 2)
    target = sum(charges[c] for c in config)
    mps = init_mps(length, spc, operators, config=config, target_qn=target)
    return mps, interactions, charges, config


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    """Tests for the Options dataclass."""

    def test_default_order(self):
        assert two_site_bug.Options().order == 'strang'

    @pytest.mark.parametrize('alias,canonical', [
        ('strang', 'strang'), ('second', 'strang'), ('2', 'strang'),
        ('lie', 'lie'), ('first', 'lie'), ('1', 'lie'),
    ])
    def test_order_aliases(self, alias, canonical):
        assert two_site_bug.Options(order=alias).order == canonical

    def test_unknown_order_raises(self):
        with pytest.raises(ValueError, match='unknown Trotter order'):
            two_site_bug.Options(order='leapfrog')

    def test_from_toml(self):
        opts = two_site_bug.Options.from_toml(
            {'dt': 0.02, 'n_steps': 50, 'order': 'second', 'max_bond': 32}
        )
        assert opts.dt == 0.02
        assert opts.n_steps == 50
        assert opts.order == 'strang'
        assert opts.max_bond == 32

    def test_to_toml_round_trip(self, tmp_path):
        original = two_site_bug.Options(dt=0.01, n_steps=7, order='lie', max_bond=16)
        path = tmp_path / 'opts.toml'
        original.to_toml(path)
        loaded = two_site_bug.Options.load_toml(path)
        assert loaded.dt == 0.01
        assert loaded.n_steps == 7
        assert loaded.order == 'lie'
        assert loaded.max_bond == 16


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    """Tests for the Summary dataclass."""

    def test_serialize_round_trip(self, spin_space):
        mps, interactions, _, _ = _domain_wall(6, spin_space)
        summary = two_site_bug.run(
            mps, interactions, two_site_bug.Options(dt=0.05, n_steps=3, max_bond=16)
        )
        restored = two_site_bug.Summary.deserialize(summary.serialize())
        assert restored.n_steps == summary.n_steps
        assert restored.bond_dims == summary.bond_dims
        assert restored.times == pytest.approx(summary.times)
        assert restored.state.L == summary.state.L


# ---------------------------------------------------------------------------
# Generators / error handling
# ---------------------------------------------------------------------------

class TestGenerators:
    """Tests for bond-generator extraction from the interaction list."""

    def test_all_bonds_populated_for_heisenberg(self):
        interactions, _, geo = heisenberg_chain(5)
        generators = build_bond_generators(interactions, geo.L)
        assert len(generators) == geo.L - 1
        assert all(g is not None for g in generators)

    def test_long_range_term_raises(self):
        # A synthetic non-nearest-neighbour term must be rejected.
        interactions, _, geo = heisenberg_chain(4)
        far = dataclasses.replace(
            next(i for i in interactions if isinstance(i, Interaction2Site)),
            leading_site=0, terminal_site=2,
        )
        with pytest.raises(NotImplementedError, match='nearest-neighbour'):
            build_bond_generators([far], geo.L)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

class TestDynamics:
    """Physical correctness of the time evolution."""

    def test_norm_conserved_real_time(self, spin_space):
        mps, interactions, _, _ = _domain_wall(6, spin_space)
        summary = two_site_bug.run(
            mps, interactions,
            two_site_bug.Options(dt=0.05, n_steps=10, max_bond=64, normalize=False),
        )
        for norm in summary.norms:
            assert abs(norm - 1.0) < 1e-10

    def test_total_sz_conserved(self, spin_space):
        mps, interactions, charges, config = _domain_wall(6, spin_space)
        sz_total = dense_total_sz(6, charges)
        psi0 = product_vector(config, charges)
        sz_before = (psi0.conj() @ sz_total @ psi0).real.item()
        summary = two_site_bug.run(
            mps, interactions, two_site_bug.Options(dt=0.05, n_steps=10, max_bond=64)
        )
        vec = mps_to_vector(summary.state)
        sz_after = (vec.conj() @ sz_total @ vec).real.item() / vec.norm().item() ** 2
        assert abs(sz_after - sz_before) < 1e-10

    def test_fidelity_matches_exact_diagonalization(self, spin_space):
        length = 6
        mps, interactions, charges, config = _domain_wall(length, spin_space)
        ham = dense_heisenberg(length, charges)
        psi0 = product_vector(config, charges)
        dt, n_steps = 0.05, 20
        summary = two_site_bug.run(
            mps, interactions,
            two_site_bug.Options(dt=dt, n_steps=n_steps, max_bond=64, normalize=False),
        )
        evolved = mps_to_vector(summary.state)
        evolved = evolved / evolved.norm()
        exact = exact_evolve(ham, psi0, dt * n_steps)
        exact = exact / exact.norm()
        fidelity = abs(torch.vdot(exact, evolved)).item()
        assert 1.0 - fidelity < 1e-6

    def test_strang_converges_second_order(self, spin_space):
        length = 6
        _, interactions, charges, config = _domain_wall(length, spin_space)
        ham = dense_heisenberg(length, charges)
        psi0 = product_vector(config, charges)

        def infidelity(dt, n_steps):
            mps, _, _, _ = _domain_wall(length, spin_space)
            summary = two_site_bug.run(
                mps, interactions,
                two_site_bug.Options(dt=dt, n_steps=n_steps, max_bond=64, normalize=False),
            )
            evolved = mps_to_vector(summary.state)
            evolved = evolved / evolved.norm()
            exact = exact_evolve(ham, psi0, dt * n_steps)
            exact = exact / exact.norm()
            return 1.0 - abs(torch.vdot(exact, evolved)).item()

        coarse = infidelity(0.10, 10)
        fine = infidelity(0.05, 20)
        # Strang state error is O(dt^2), so the infidelity is O(dt^4): halving dt
        # cuts it by ~16. Allow a generous band around the asymptotic ratio.
        assert coarse / fine > 8.0

    def test_strang_beats_lie(self, spin_space):
        length = 6
        ham = dense_heisenberg(length, [s.charge for s in spin_space[0].sectors])

        def infidelity(order):
            mps, interactions, charges, config = _domain_wall(length, spin_space)
            psi0 = product_vector(config, charges)
            summary = two_site_bug.run(
                mps, interactions,
                two_site_bug.Options(dt=0.1, n_steps=10, order=order, max_bond=64, normalize=False),
            )
            evolved = mps_to_vector(summary.state)
            evolved = evolved / evolved.norm()
            exact = exact_evolve(ham, psi0, 1.0)
            exact = exact / exact.norm()
            return 1.0 - abs(torch.vdot(exact, evolved)).item()

        assert infidelity('strang') < infidelity('lie')

    def test_imaginary_time_lowers_energy(self, spin_space):
        length = 6
        mps, interactions, charges, config = _domain_wall(length, spin_space)
        ham = dense_heisenberg(length, charges)
        ground = torch.linalg.eigvalsh(ham)[0].item()
        psi0 = product_vector(config, charges)
        energy_before = (psi0.conj() @ ham @ psi0).real.item()
        summary = two_site_bug.run(
            mps, interactions,
            two_site_bug.Options(dt=0.05, n_steps=40, imaginary_time=True, max_bond=64),
        )
        vec = mps_to_vector(summary.state)
        vec = vec / vec.norm()
        energy_after = (vec.conj() @ ham @ vec).real.item()
        assert energy_after < energy_before
        assert energy_after > ground - 1e-9
