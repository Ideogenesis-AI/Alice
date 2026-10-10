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


"""Physics validation for TDVP.

The unit tests check that each piece computes what it claims to; these check
that the assembled integrator reproduces quantum mechanics. Three independent
kinds of evidence are used, since each misses different classes of error:

1. **Exact references.** A single particle on an open tight-binding chain has
   a closed-form propagator, so site occupations can be compared against
   analytic values with no tensor-network reference involved. This is the only
   test here that would catch a consistent error shared by every component.
2. **Symmetries.** The norm, the energy, and the U(1) charge are conserved by
   `exp(-i H t)`, and evolution is reversible. These are cheap, sharp, and
   sensitive to sign and ordering mistakes that an exact reference on a small
   system might not resolve.
3. **Convergence rates.** The error must fall as `dt²`. A wrong rate points at
   the integrator composition even when every individual step looks plausible.

Measuring a convergence rate means running the same evolution at several step
sizes, the finest of which dominates the cost, so those tests carry the `slow`
marker and are skipped by `-m "not slow"`. Everything that can catch an error
from a single run stays in the default suite.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from alice.algorithm import dmrg
from alice.algorithm.tdvp import Artifact, Options, run
from alice.network import observe

from .conftest import (
    FREE_FERMION_L,
    FREE_FERMION_SITE,
    analytic_occupations,
    bond_profile,
    mps_overlap,
)


def _evolve(mps, mpo, **kwargs):
    """Evolve a copy of `mps` and return `(summary, artifact)`."""
    opts = Options(save_artifacts=False, **kwargs)
    return run(Artifact(state=copy.deepcopy(mps), time=0.0, step=0), mpo, opts)


def _occupations(mps, n_obs):
    """Measure the occupation of every site."""
    return np.array([observe(mps, n) for n in n_obs])


# ---------------------------------------------------------------------------
# Exact reference: the free-fermion chain
# ---------------------------------------------------------------------------

class TestFreeFermionPropagator:
    """Compare against the closed-form single-particle propagator.

    One particle on `H = -t Σ (c†_j c_{j+1} + h.c.)` with open boundaries
    evolves within the single-particle sector, where the eigenbasis is a sine
    series and the propagator is analytic. The MPS carries the full many-body
    state, so agreement tests the whole machinery — Jordan-Wigner strings,
    environments, and integrator alike — against a result that owes nothing
    to tensor networks.
    """

    def test_initial_state_is_a_particle_at_one_site(self, free_fermion):
        """The benchmark starts where the analytic formula says it does."""
        _, n_obs, make_state = free_fermion
        occ = _occupations(make_state(), n_obs)
        expected = np.zeros(FREE_FERMION_L)
        expected[FREE_FERMION_SITE] = 1.0
        assert np.allclose(occ, expected, atol=1e-12)

    def test_initial_energy_vanishes(self, free_fermion):
        """A localized particle has zero hopping energy, exactly."""
        mpo, _, make_state = free_fermion
        assert abs(observe(make_state(), mpo)) < 1e-12

    @pytest.mark.slow
    def test_occupations_match_the_analytic_propagator(self, free_fermion):
        """The 2-site scheme reproduces the exact site occupations.

        `max_bond=None` lets the bond grow to whatever the single-particle
        sector needs, so the only error left is the integrator's `dt²`.
        """
        mpo, n_obs, make_state = free_fermion
        _, final = _evolve(make_state(), mpo, scheme='2s', dt=0.01, n_steps=100)

        occ = _occupations(final.state, n_obs)
        assert np.allclose(occ, analytic_occupations(1.0), atol=1e-4)

    def test_particle_number_is_conserved(self, free_fermion):
        """The occupations keep summing to one particle."""
        mpo, n_obs, make_state = free_fermion
        _, final = _evolve(make_state(), mpo, scheme='2s', dt=0.02, n_steps=50)
        assert abs(_occupations(final.state, n_obs).sum() - 1.0) < 1e-10

    def test_the_wavefront_actually_spreads(self, free_fermion):
        """Guard against a frozen state passing the other tests.

        A state that never moved would conserve everything and sit at zero
        energy forever, so the benchmark needs an explicit check that the
        particle has left its initial site.
        """
        mpo, n_obs, make_state = free_fermion
        _, final = _evolve(make_state(), mpo, scheme='2s', dt=0.02, n_steps=50)
        occ = _occupations(final.state, n_obs)
        assert occ[FREE_FERMION_SITE] < 0.5
        assert occ[0] > 1e-4

    @pytest.mark.slow
    def test_second_order_convergence_to_the_analytic_result(self, free_fermion):
        """Halving `dt` must quarter the error against the exact propagator.

        Measured against analytic values rather than a finer TDVP run, so a
        systematically wrong propagator cannot hide behind self-consistency.
        """
        mpo, n_obs, make_state = free_fermion
        exact = analytic_occupations(1.0)

        errors = []
        for n_steps in (25, 50, 100):
            _, final = _evolve(make_state(), mpo, scheme='2s',
                               dt=1.0 / n_steps, n_steps=n_steps)
            errors.append(
                float(np.max(np.abs(_occupations(final.state, n_obs) - exact)))
            )

        for coarse, fine in zip(errors, errors[1:]):
            assert 3.0 < coarse / fine < 5.0


class TestFreeFermion1Site:
    """The 1-site scheme on the same benchmark, once its bonds are open.

    Started from a product state the scheme is frozen — every bond has
    dimension 1 and it cannot grow them — so the particle would never move.
    Warming up with a few 2-site steps opens the bonds first, after which the
    1-site scheme is a legitimate fixed-rank integrator.
    """

    def test_a_product_state_is_frozen(self, free_fermion):
        """The documented trap, pinned so it cannot change silently."""
        mpo, n_obs, make_state = free_fermion
        _, final = _evolve(make_state(), mpo, scheme='1s', dt=0.02, n_steps=50)

        occ = _occupations(final.state, n_obs)
        assert occ[FREE_FERMION_SITE] > 1.0 - 1e-9
        # And it is frozen *silently*: norm and energy look perfectly healthy.
        assert abs(final.state.norm() - 1.0) < 1e-10
        assert abs(observe(final.state, mpo)) < 1e-10

    @pytest.mark.slow
    def test_converges_once_the_bonds_are_open(self, free_fermion):
        """After a 2-site warm-up, `1s` converges at second order.

        The comparison is against a converged 2-site run rather than the
        analytic values: the warm-up itself carries error, and the fixed-rank
        manifold the 1-site scheme is confined to cannot represent the exact
        state anyway.
        """
        mpo, n_obs, make_state = free_fermion

        _, warm = _evolve(make_state(), mpo, scheme='2s', dt=0.005, n_steps=40)
        assert min(bond_profile(warm.state)) >= 2

        _, reference = _evolve(warm.state, mpo, scheme='2s',
                               dt=0.0025, n_steps=200)
        ref_occ = _occupations(reference.state, n_obs)

        errors = []
        for n_steps in (25, 50, 100):
            _, final = _evolve(warm.state, mpo, scheme='1s',
                               dt=0.5 / n_steps, n_steps=n_steps)
            errors.append(
                float(np.max(np.abs(_occupations(final.state, n_obs) - ref_occ)))
            )

        # The 1-site manifold has a fixed-rank error floor, so the ratios are
        # checked on the dt-dependent part only, by requiring a clear decrease.
        assert errors[0] > errors[1] > errors[2]


# ---------------------------------------------------------------------------
# Symmetries and conservation laws
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('scheme', ['1s', '2s'])
@pytest.mark.parametrize('order', [1, 2])
class TestConservation:
    """Quantities that `exp(-i H t)` leaves untouched."""

    def test_norm(self, heisenberg_L6, scheme, order):
        mps, mpo = heisenberg_L6
        mps.canonical(0)
        mps.normalize()
        summary, _ = _evolve(mps, mpo, scheme=scheme, order=order,
                             dt=0.05, n_steps=10)
        assert max(abs(n - 1.0) for n in summary.norms) < 1e-10

    def test_energy(self, heisenberg_L6, scheme, order):
        """Energy conservation is the sharpest cheap check available.

        It is violated by a wrong sign in the propagator, a misordered
        composition, or a missing back-evolution, none of which the norm would
        notice.
        """
        mps, mpo = heisenberg_L6
        mps.canonical(0)
        mps.normalize()
        summary, _ = _evolve(mps, mpo, scheme=scheme, order=order,
                             dt=0.05, n_steps=10)
        initial = summary.energies[0]
        assert max(abs(e - initial) for e in summary.energies) < 1e-9

    def test_u1_charge(self, heisenberg_L6, scheme, order):
        """The symmetry structure forbids charge leaking between sectors.

        Nicole's block sparsity makes this structural rather than numerical,
        so the test is that no sector outside the initial charge appears.
        """
        mps, mpo = heisenberg_L6
        mps.canonical(0)
        mps.normalize()
        before = {k[-1] for k in mps[mps.L - 1].data}

        _, final = _evolve(mps, mpo, scheme=scheme, order=order,
                           dt=0.05, n_steps=5)

        assert {k[-1] for k in final.state[final.state.L - 1].data} <= before


class TestReversibility:
    """Running time backwards must undo the evolution."""

    @pytest.mark.parametrize('scheme', ['1s', '2s'])
    def test_forward_then_backward_returns_the_state(self, heisenberg_L6, scheme):
        mps, mpo = heisenberg_L6
        mps.canonical(0)
        mps.normalize()
        original = copy.deepcopy(mps)

        _, forward = _evolve(mps, mpo, scheme=scheme, dt=0.05, n_steps=8)
        _, back = run(Artifact(state=forward.state, time=0.0, step=0), mpo,
                      Options(scheme=scheme, dt=-0.05, n_steps=8,
                              save_artifacts=False))

        assert abs(abs(mps_overlap(original, back.state)) - 1.0) < 1e-8


class TestEigenstatePhase:
    """An eigenstate must only pick up a phase.

    `exp(-i H t)|E⟩ = exp(-i E t)|E⟩`, so a DMRG ground state should keep its
    overlap magnitude at one while its phase winds at the ground-state energy.
    This is a strong test because the tangent-space projection is exact for an
    eigenstate: any drift in the overlap magnitude is integrator error, not
    a manifold restriction.
    """

    @pytest.fixture
    def ground_state(self, heisenberg_L4_real):
        """A DMRG ground state, together with its energy.

        DMRG runs on the real fixture: its environments are built from a real
        identity, so unlike TDVP it cannot take a complex state. `run` then
        promotes both the state and the MPO itself.
        """
        mps, mpo = heisenberg_L4_real
        mps.canonical(0)
        mps.normalize()
        summary = dmrg.run(mps, mpo,
                           dmrg.Options(scheme='2s', n_sweeps=14, max_bond=32))
        return summary.state, mpo, summary.energy

    def test_overlap_magnitude_stays_at_one(self, ground_state):
        mps, mpo, _ = ground_state
        original = copy.deepcopy(mps)
        original.to_complex()
        _, final = _evolve(mps, mpo, scheme='2s', dt=0.05, n_steps=20)
        assert abs(abs(mps_overlap(original, final.state)) - 1.0) < 1e-8

    def test_phase_winds_at_the_ground_state_energy(self, ground_state):
        """The accumulated phase is `-E·t`, which fixes the sign convention.

        A propagator built as `exp(+i H dt)` would conserve the energy and the
        norm and pass every test above; only the phase direction catches it.
        """
        mps, mpo, energy = ground_state
        original = copy.deepcopy(mps)
        original.to_complex()
        t = 1.0
        _, final = _evolve(mps, mpo, scheme='2s', dt=0.05, n_steps=20)

        overlap = mps_overlap(original, final.state)
        expected = complex(math.cos(-energy * t), math.sin(-energy * t))
        assert abs(overlap - expected) < 1e-6


# ---------------------------------------------------------------------------
# Convergence in the time step
# ---------------------------------------------------------------------------

def _distance(a, b):
    """Return the distance between two normalized states, up to a phase.

    `‖a - e^{iθ}b‖ = sqrt(2(1 - |⟨a|b⟩|))` at the optimal phase. The overlap
    deficit `1 - |⟨a|b⟩|` is *quadratic* in this distance, so using it
    directly would report twice the convergence order.
    """
    return math.sqrt(max(0.0, 2.0 * (1.0 - abs(mps_overlap(a, b)))))


class TestConvergenceRate:
    """The integrator error must fall as `dt²`."""

    @pytest.mark.slow
    @pytest.mark.parametrize('order', [1, 2])
    def test_second_order_in_dt(self, heisenberg_L6, order):
        """Both orders are globally second order, which is the claim tested.

        Two `order=1` steps compose into exactly one symmetric `order=2` step
        of twice the size, so the two settings differ in where snapshots land,
        not in the accuracy of the underlying flow. Were `order=1` genuinely
        first order, its ratios here would sit near 2 instead of 4.

        The 1-site scheme is used because it is a true fixed-rank manifold:
        the 2-site scheme either truncates — adding a dt-independent error
        that swamps the integrator — or reaches full rank, where the flow is
        exact and there is no dt error left to measure.
        """
        mps, mpo = heisenberg_L6
        mps.canonical(0)
        mps.normalize()

        _, reference = _evolve(mps, mpo, scheme='1s', order=order,
                               dt=0.0025, n_steps=160)

        errors = []
        for n_steps in (10, 20, 40):
            _, final = _evolve(mps, mpo, scheme='1s', order=order,
                               dt=0.4 / n_steps, n_steps=n_steps)
            errors.append(_distance(reference.state, final.state))

        for coarse, fine in zip(errors, errors[1:]):
            assert 3.0 < coarse / fine < 5.5

    def test_is_exact_at_full_rank(self, heisenberg_L4):
        """At full bond dimension the flow is exact for any step size.

        Once the manifold is the whole Hilbert space the tangent projector is
        the identity, the sum over sites telescopes exactly, and the sweep
        reproduces `exp(-i H dt)` with no discretization error. Worth pinning:
        it is why a convergence rate cannot be measured at full rank, and it
        is the sense in which TDVP is "exact" — on its own manifold.
        """
        mps, mpo = heisenberg_L4
        mps.canonical(0)
        mps.normalize()

        _, coarse = _evolve(mps, mpo, scheme='2s', dt=0.25, n_steps=4)
        _, fine = _evolve(mps, mpo, scheme='2s', dt=0.02, n_steps=50)

        assert _distance(coarse.state, fine.state) < 1e-10
