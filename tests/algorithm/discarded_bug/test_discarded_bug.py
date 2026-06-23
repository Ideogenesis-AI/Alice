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
# Author of code: Madhav Menon.


"""Tests for the discarded-projector BUG integrator (Options, Summary, run).

The discarded-projector BUG (see :mod:`alice.algorithm.discarded_bug`) is a
rank-adaptive two-site Basis-Update & Galerkin integrator that differs from the
faithful Ceruti–Kusch–Lubich scheme only in the local bond update: the discarded
projector is applied to the K/L generator *before* the exponential, and the basis
is grown by a direct sum with no augmented overlap matrices. These tests check, on
the symmetric (isotropic) Heisenberg chain — which conserves total Sz and whose
small-chain dynamics are available by exact diagonalization — that the integrator:

* grows the bond dimension as a domain wall melts (rank adaptivity),
* converges to the ANALYTICAL solution (exact diagonalization of the chain) at the
  expected Strang order O(dt^2) — this is the primary correctness criterion,
* conserves the state norm (real time) and total Sz,
* cools toward the ground state in imaginary time.

Correctness is judged against the analytical (exact-diagonalization) solution, NOT
against the faithful two-site BUG. A single informational check records that the
two schemes happen to agree (they span the same augmented subspaces), but the
binding assertions are all against exact diagonalization.

A note on comparison baselines: Alice's 2-site TDVP is not yet implemented. The
reference Julia implementation of this scheme was measured head-to-head against
2-site TDVP on the same domain-wall quench; the discarded-BUG infidelity stayed
within a bounded ~6x factor of TDVP's (same O(dt^2) error class), and 2-site TDVP's
own error is flat in dt (a fixed-rank manifold error, not convergent to zero).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from nicole import Index, Tensor

from alice import init_mps
from alice.algorithm import discarded_bug, two_site_bug
from alice.algorithm.two_site_bug.bond import build_bond_generators
from alice.network.interaction import Interaction2Site

from .conftest import (
    dense_hamiltonian,
    dense_total_sz,
    exact_evolve,
    heisenberg_chain,
    mps_to_vector,
)


def _domain_wall(length, spin_space):
    """Return `(mps, interactions, charges, psi0)` for a full-phys Heisenberg domain wall.

    The state is the Sz=0 domain wall `|↓…↓↑…↑⟩`. Each physical leg is inflated to
    the full spin-1/2 index so spins can flip and the state densifies to `2**L`.
    `psi0` is the dense initial vector. (Identical construction to the faithful
    two-site BUG tests so the two integrators see the same initial condition.)
    """
    _, operators = spin_space
    interactions, spc, _ = heisenberg_chain(length)
    charges = [sector.charge for sector in spc.sectors]
    config = [0] * (length // 2) + [1] * (length - length // 2)
    target = sum(charges[c] for c in config)
    mps = init_mps(length, spc, operators, config=config, target_qn=target)
    for i in range(mps.L):
        core = mps[i]
        full_phys = Index(core.indices[2].direction, core.indices[2].group, spc.sectors)
        mps[i] = Tensor(
            indices=(core.indices[0], core.indices[1], full_phys),
            itags=core.itags,
            data={key: block.clone() for key, block in core.data.items()},
            dtype=core.dtype,
        )
    psi0 = mps_to_vector(mps, charges)
    return mps, interactions, charges, psi0


# ---------------------------------------------------------------------------
# Options / Summary
# ---------------------------------------------------------------------------

class TestOptions:
    """The discarded-projector BUG reuses the two-site BUG Options/Summary."""

    def test_options_is_two_site_bug_options(self):
        assert discarded_bug.Options is two_site_bug.Options

    def test_default_order(self):
        assert discarded_bug.Options().order == 'strang'

    @pytest.mark.parametrize('alias,canonical', [
        ('strang', 'strang'), ('second', 'strang'), ('2', 'strang'),
        ('lie', 'lie'), ('first', 'lie'), ('1', 'lie'),
    ])
    def test_order_aliases(self, alias, canonical):
        assert discarded_bug.Options(order=alias).order == canonical

    def test_serialize_round_trip(self, spin_space):
        mps, interactions, _, _ = _domain_wall(6, spin_space)
        summary = discarded_bug.run(
            mps, interactions, discarded_bug.Options(dt=0.05, n_steps=3, max_bond=16)
        )
        restored = discarded_bug.Summary.deserialize(summary.serialize())
        assert restored.n_steps == summary.n_steps
        assert restored.bond_dims == summary.bond_dims
        assert restored.times == pytest.approx(summary.times)


# ---------------------------------------------------------------------------
# Generators / error handling
# ---------------------------------------------------------------------------

class TestGenerators:
    """Bond-generator handling is shared with the faithful kernel."""

    def test_long_range_term_raises(self):
        interactions, _, geo = heisenberg_chain(4)
        far = dataclasses.replace(
            next(i for i in interactions if isinstance(i, Interaction2Site)),
            leading_site=0, terminal_site=2,
        )
        with pytest.raises(NotImplementedError, match='nearest-neighbour'):
            build_bond_generators([far], geo.L)

    def test_two_site_chain_runs(self, spin_space):
        """L == 2 is the minimal valid chain (a single bond)."""
        mps, interactions, _, _ = _domain_wall(2, spin_space)
        summary = discarded_bug.run(
            mps, interactions, discarded_bug.Options(dt=0.05, n_steps=2, max_bond=8)
        )
        assert summary.n_steps == 2
        assert len(summary.bond_dims) == 1


# ---------------------------------------------------------------------------
# Rank adaptivity
# ---------------------------------------------------------------------------

class TestRankAdaptivity:
    """The bond dimension must grow as the domain wall melts."""

    def test_bond_dimension_grows(self, spin_space):
        length = 6
        mps, interactions, _, _ = _domain_wall(length, spin_space)
        # The wall starts as a product state (every bond chi=1).
        assert max(mps.bond_dims) == 1
        summary = discarded_bug.run(
            mps, interactions,
            discarded_bug.Options(dt=0.05, n_steps=10, max_bond=64),
        )
        # It melts and the bond dimension climbs well past 1.
        assert max(summary.bond_dims) > 1
        assert max(summary.max_bond_dims) >= 4
        # The proposed augmented rank reaches at least the kept rank every step.
        assert all(a >= 1 for a in summary.aug_dims)

    def test_max_bond_cap_respected(self, spin_space):
        length = 6
        mps, interactions, _, _ = _domain_wall(length, spin_space)
        cap = 4
        summary = discarded_bug.run(
            mps, interactions,
            discarded_bug.Options(dt=0.05, n_steps=10, max_bond=cap),
        )
        assert max(summary.bond_dims) <= cap


# ---------------------------------------------------------------------------
# Accuracy vs exact diagonalization and vs the faithful scheme
# ---------------------------------------------------------------------------

class TestAccuracy:
    """Physical correctness of the time evolution on the symmetric Heisenberg chain."""

    def test_fidelity_matches_exact_diagonalization(self, spin_space):
        length = 6
        mps, interactions, charges, psi0 = _domain_wall(length, spin_space)
        ham = dense_hamiltonian(interactions, length, charges)
        psi0 = psi0 / psi0.norm()
        dt, n_steps = 0.05, 20
        summary = discarded_bug.run(
            mps, interactions,
            discarded_bug.Options(dt=dt, n_steps=n_steps, max_bond=64, normalize=False),
        )
        evolved = mps_to_vector(summary.state, charges)
        evolved = evolved / evolved.norm()
        exact = exact_evolve(ham, psi0, dt * n_steps)
        exact = exact / exact.norm()
        fidelity = abs(torch.vdot(exact, evolved)).item()
        assert 1.0 - fidelity < 1e-6

    def test_strang_converges_second_order(self, spin_space):
        length = 6
        _, interactions, charges, psi0 = _domain_wall(length, spin_space)
        ham = dense_hamiltonian(interactions, length, charges)
        psi0 = psi0 / psi0.norm()

        def infidelity(dt, n_steps):
            mps, _, _, _ = _domain_wall(length, spin_space)
            summary = discarded_bug.run(
                mps, interactions,
                discarded_bug.Options(dt=dt, n_steps=n_steps, max_bond=64, normalize=False),
            )
            evolved = mps_to_vector(summary.state, charges)
            evolved = evolved / evolved.norm()
            exact = exact_evolve(ham, psi0, dt * n_steps)
            exact = exact / exact.norm()
            return 1.0 - abs(torch.vdot(exact, evolved)).item()

        coarse = infidelity(0.10, 10)
        fine = infidelity(0.05, 20)
        # Strang state error is O(dt^2) ⇒ infidelity O(dt^4): halving dt cuts it ~16x.
        assert coarse / fine > 8.0

    def test_agreement_with_faithful_is_informational(self, spin_space):
        """INFORMATIONAL (not the correctness criterion): the discarded and faithful
        schemes span the same augmented subspaces, so the symmetric sweep happens to
        agree. The binding accuracy test is `test_fidelity_matches_exact_diagonalization`
        (vs the analytical solution); this only records the incidental agreement."""
        length = 6
        mps_d, interactions, charges, _ = _domain_wall(length, spin_space)
        mps_f, _, _, _ = _domain_wall(length, spin_space)
        opts = dict(dt=0.05, n_steps=15, max_bond=64, normalize=False)
        sd = discarded_bug.run(mps_d, interactions, discarded_bug.Options(**opts))
        sf = two_site_bug.run(mps_f, interactions, two_site_bug.Options(**opts))
        vd = mps_to_vector(sd.state, charges)
        vd = vd / vd.norm()
        vf = mps_to_vector(sf.state, charges)
        vf = vf / vf.norm()
        infidelity = 1.0 - abs(torch.vdot(vf, vd)).item()
        # Loose bound — this is a sanity note, not the accuracy gate.
        assert infidelity < 1e-6

    def test_strang_beats_lie(self, spin_space):
        length = 6
        _, interactions, charges, _ = _domain_wall(length, spin_space)
        ham = dense_hamiltonian(interactions, length, charges)

        def infidelity(order):
            mps, interactions_l, charges_l, psi0 = _domain_wall(length, spin_space)
            psi0 = psi0 / psi0.norm()
            summary = discarded_bug.run(
                mps, interactions_l,
                discarded_bug.Options(dt=0.1, n_steps=10, order=order, max_bond=64, normalize=False),
            )
            evolved = mps_to_vector(summary.state, charges_l)
            evolved = evolved / evolved.norm()
            exact = exact_evolve(ham, psi0, 1.0)
            exact = exact / exact.norm()
            return 1.0 - abs(torch.vdot(exact, evolved)).item()

        assert infidelity('strang') < infidelity('lie')


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------

class TestConservation:
    """Norm (real time), total Sz, and imaginary-time energy descent."""

    def test_norm_conserved_real_time(self, spin_space):
        mps, interactions, _, _ = _domain_wall(6, spin_space)
        summary = discarded_bug.run(
            mps, interactions,
            discarded_bug.Options(dt=0.05, n_steps=10, max_bond=64, normalize=False),
        )
        for norm in summary.norms:
            assert abs(norm - 1.0) < 1e-10

    def test_total_sz_conserved(self, spin_space):
        mps, interactions, charges, psi0 = _domain_wall(6, spin_space)
        sz_total = dense_total_sz(6, charges)
        sz_before = (psi0.conj() @ sz_total @ psi0).real.item() / psi0.norm().item() ** 2
        summary = discarded_bug.run(
            mps, interactions, discarded_bug.Options(dt=0.05, n_steps=10, max_bond=64)
        )
        vec = mps_to_vector(summary.state, charges)
        sz_after = (vec.conj() @ sz_total @ vec).real.item() / vec.norm().item() ** 2
        assert abs(sz_after - sz_before) < 1e-10

    def test_imaginary_time_lowers_energy(self, spin_space):
        length = 6
        mps, interactions, charges, psi0 = _domain_wall(length, spin_space)
        ham = dense_hamiltonian(interactions, length, charges)
        ground = torch.linalg.eigvalsh(ham)[0].item()
        psi0 = psi0 / psi0.norm()
        energy_before = (psi0.conj() @ ham @ psi0).real.item()
        summary = discarded_bug.run(
            mps, interactions,
            discarded_bug.Options(dt=0.05, n_steps=40, imaginary_time=True, max_bond=64),
        )
        vec = mps_to_vector(summary.state, charges)
        vec = vec / vec.norm()
        energy_after = (vec.conj() @ ham @ vec).real.item()
        assert energy_after < energy_before
        assert energy_after > ground - 1e-9


# ---------------------------------------------------------------------------
# Project-before behaviour: seeded melt vs pure-product bootstrap
# ---------------------------------------------------------------------------

class TestProjectBefore:
    """The defining project-before behaviour and its one documented limitation."""

    def test_seeded_wall_melts_with_project_before(self, spin_space):
        """Once the wall carries chi>=2 (seeded by a couple of faithful steps), the
        project-before discarded update grows the rank further and tracks the exact
        dynamics — i.e. project-before is fine away from a pure product state."""
        length = 6
        mps, interactions, charges, psi0 = _domain_wall(length, spin_space)
        ham = dense_hamiltonian(interactions, length, charges)
        psi0n = psi0 / psi0.norm()

        # Seed off the product state with two faithful steps.
        two_site_bug.run(mps, interactions,
                         two_site_bug.Options(dt=0.025, n_steps=2, max_bond=64, normalize=False))
        seeded_chi = max(mps.bond_dims)
        assert seeded_chi >= 2

        # Continue with the discarded (project-before) scheme.
        summary = discarded_bug.run(
            mps, interactions,
            discarded_bug.Options(dt=0.025, n_steps=18, max_bond=64, normalize=False),
        )
        assert max(summary.bond_dims) >= seeded_chi  # rank kept growing / held
        evolved = mps_to_vector(summary.state, charges)
        evolved = evolved / evolved.norm()
        exact = exact_evolve(ham, psi0n, 0.025 * 20)
        exact = exact / exact.norm()
        assert 1.0 - abs(torch.vdot(exact, evolved)).item() < 1e-5

    def test_pure_product_without_augmentation_stays_rank_one(self, spin_space):
        """With augmentation DISABLED, a pure product wall cannot grow rank: the
        one-sided K/L generators see the two-spin flip only through augmentation, so
        the bond dimension stays chi=1. This is the explicit no-bootstrap baseline
        (with augmentation ON, the symmetry sector-completion does grow the rank)."""
        length = 6
        mps, interactions, _, _ = _domain_wall(length, spin_space)
        assert max(mps.bond_dims) == 1
        summary = discarded_bug.run(
            mps, interactions,
            discarded_bug.Options(dt=0.05, n_steps=5, max_bond=64, augment=False, normalize=True),
        )
        assert max(summary.bond_dims) == 1
        assert all(a == 1 for a in summary.aug_dims)
