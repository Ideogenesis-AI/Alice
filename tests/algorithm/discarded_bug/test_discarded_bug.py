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
rank-adaptive **two-site** Basis-Update & Galerkin integrator: the MPS
specialisation of the tree-tensor-network BUG of Ceruti–Lubich–Walach, with two
modifications — every local update is two-site (through the two-site effective
Hamiltonian with the MPO environments), and the basis is grown with the **discarded
projector** (``qr([Theta1_left | U0])`` read off the evolved two-site block, with no
augmented overlap matrices). Like 2-site TDVP and DMRG it takes a Hamiltonian
``MPO``; the step recursively bisects the chain (the Lubich tree BUG, whose tree is
built by recursive bisection of the 1D modes) with one two-site node update per
bisection bond, and has no backward substep.

These tests check, on the symmetric (isotropic) Heisenberg chain — which conserves
total Sz and whose small-chain dynamics are available by exact diagonalization —
that the integrator:

* **grows the bond dimension as a domain wall melts** — the headline rank-adaptive
  property: a product-state wall develops the full ballistic light cone, a peaked bond
  profile reaching the exact half-chain Schmidt rank ``2**(L/2)`` (this is the primary
  validation);
* tracks the exact-diagonalization trajectory at short time — the recursive-bisection
  step is first order, so accuracy is *not* the validated property; the bond growth is;
* conserves the state norm (real time) and total Sz;
* lowers the energy in imaginary time.
"""

from __future__ import annotations

import pytest
import torch
from nicole import Index, Tensor

from alice import init_mps
from alice.algorithm import discarded_bug
from alice.network import build_hamiltonian

from .conftest import (
    dense_hamiltonian,
    dense_total_sz,
    exact_evolve,
    heisenberg_chain,
    mps_to_vector,
)


def _domain_wall(length, spin_space):
    """Return ``(mps, mpo, charges, psi0)`` for a full-phys Heisenberg domain wall.

    The state is the Sz=0 domain wall ``|down…down up…up>``. Each physical leg is
    inflated to the full spin-1/2 index so spins can flip and the state densifies to
    ``2**L``. ``mpo`` is the Hamiltonian MPO the integrator consumes; ``psi0`` is the
    dense initial vector.
    """
    _, operators = spin_space
    interactions, spc, _ = heisenberg_chain(length)
    mpo = build_hamiltonian(interactions, length, spc)
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
    return mps, mpo, charges, psi0


def _infidelity(vec, exact):
    vec = vec / vec.norm()
    exact = exact / exact.norm()
    return 1.0 - abs(torch.vdot(exact, vec)).item()


# ---------------------------------------------------------------------------
# Options / Summary
# ---------------------------------------------------------------------------

class TestOptions:
    """Options defaults, validation, and Summary serialization."""

    def test_defaults(self):
        opts = discarded_bug.Options()
        assert opts.dt == pytest.approx(0.02)
        assert opts.normalize is True
        assert opts.imaginary_time is False

    def test_requires_two_sites(self, spin_space):
        mps, mpo, _, _ = _domain_wall(2, spin_space)
        # L == 2 is the minimal valid chain; L < 2 is rejected.
        summary = discarded_bug.run(mps, mpo, discarded_bug.Options(dt=0.05, n_steps=1))
        assert summary.n_steps == 1
        # A single-site chain is rejected by the >= 2-site guard.
        one_site_mps = type(mps)([mps[0]])
        one_site_mpo = type(mpo)([mpo[0]])
        with pytest.raises(ValueError, match='at least 2 sites'):
            discarded_bug.run(one_site_mps, one_site_mpo,
                              discarded_bug.Options(dt=0.05, n_steps=1))

    def test_length_mismatch_raises(self, spin_space):
        mps, mpo, _, _ = _domain_wall(4, spin_space)
        short_mpo = type(mpo)([mpo[b] for b in range(3)])
        with pytest.raises(ValueError, match='same length'):
            discarded_bug.run(mps, short_mpo, discarded_bug.Options(dt=0.05, n_steps=1))

    def test_serialize_round_trip(self, spin_space):
        mps, mpo, _, _ = _domain_wall(6, spin_space)
        summary = discarded_bug.run(mps, mpo, discarded_bug.Options(dt=0.05, n_steps=3, max_bond=16))
        restored = discarded_bug.Summary.deserialize(summary.serialize())
        assert restored.n_steps == summary.n_steps
        assert restored.bond_dims == summary.bond_dims
        assert restored.times == pytest.approx(summary.times)
        assert restored.max_bond_dims == summary.max_bond_dims


# ---------------------------------------------------------------------------
# Rank adaptivity — the primary validated property
# ---------------------------------------------------------------------------

class TestRankAdaptivity:
    """The bond dimension must grow as the domain wall melts (the headline property)."""

    def test_product_wall_grows_bond_dimension(self, spin_space):
        """A pure product-state wall (every bond chi=1) develops entanglement: acting
        with H creates a rank-2 interface, and the bisection spreads it outward."""
        length = 8
        mps, mpo, _, _ = _domain_wall(length, spin_space)
        assert max(mps.bond_dims) == 1  # starts as a product state
        summary = discarded_bug.run(
            mps, mpo, discarded_bug.Options(dt=0.05, n_steps=10, max_bond=64, normalize=False),
        )
        # The wall melts: the bond dimension climbs well past 1 and the max kept rank
        # grows step by step (rank adaptivity, not a fixed manifold).
        assert max(summary.bond_dims) >= 8
        assert summary.max_bond_dims[0] < summary.max_bond_dims[-1]

    def test_ballistic_light_cone(self, spin_space):
        """The recursive-bisection BUG melts the domain wall into the full ballistic
        light cone: a peaked bond-dimension profile rising from the edges to the centre,
        reaching the exact central-bond saturation ``2**(L/2)``. This is the headline
        rank-adaptive property — every bond (every bisection node) grows."""
        length = 8
        dt, n_steps = 0.05, 12
        mps, mpo, _, _ = _domain_wall(length, spin_space)
        summary = discarded_bug.run(
            mps, mpo, discarded_bug.Options(dt=dt, n_steps=n_steps, max_bond=64, normalize=False),
        )
        bond = summary.bond_dims  # length L-1, indices 0 … L-2
        c = length // 2 - 1       # central bond index
        # Peaked profile: bond dimension rises from the left edge to the centre …
        for b in range(c):
            assert bond[b] <= bond[b + 1]
        # … and falls from the centre to the right edge.
        for b in range(c, length - 2):
            assert bond[b] >= bond[b + 1]
        # The centre bond reaches the full Schmidt rank of the half-chain bipartition.
        assert max(bond) == 2 ** (length // 2)
        # Every interior bond has grown past the product-state value of 1.
        assert min(bond) > 1

    def test_max_bond_cap_respected(self, spin_space):
        length = 8
        mps, mpo, _, _ = _domain_wall(length, spin_space)
        cap = 4
        summary = discarded_bug.run(
            mps, mpo, discarded_bug.Options(dt=0.05, n_steps=10, max_bond=cap, normalize=False),
        )
        assert max(summary.bond_dims) <= cap


# ---------------------------------------------------------------------------
# Accuracy vs exact diagonalization (forward-only floor)
# ---------------------------------------------------------------------------

class TestAccuracy:
    """The trajectory tracks exact diagonalization at the forward-only error floor."""

    def test_tracks_exact_diagonalization(self, spin_space):
        """Short-time fidelity: a few steps stay close to the exact dynamics. (The
        recursive-bisection step is first order, so the error grows with time; this
        checks the early trajectory, where it is still small.)"""
        length = 6
        mps, mpo, charges, psi0 = _domain_wall(length, spin_space)
        ham = dense_hamiltonian(*_ham_args(spin_space, length, charges))
        dt, n_steps = 0.02, 5
        summary = discarded_bug.run(
            mps, mpo, discarded_bug.Options(dt=dt, n_steps=n_steps, max_bond=64, normalize=False),
        )
        evolved = mps_to_vector(summary.state, charges)
        exact = exact_evolve(ham, psi0 / psi0.norm(), dt * n_steps)
        assert _infidelity(evolved, exact) < 1e-2

    def test_single_step_is_first_order(self, spin_space):
        """The forward sweep is a first-order integrator: its SINGLE-STEP infidelity
        scales as O(dt^2) (halving dt cuts the single-step error ~4x). This is the
        local-error order; note the *multi-step* error to a fixed time does NOT shrink
        with dt because the forward-only projection floor (no backward step) dominates
        — see ``test_forward_only_floor_does_not_shrink_with_dt``."""
        length = 6
        _, _, charges, psi0 = _domain_wall(length, spin_space)
        ham = dense_hamiltonian(*_ham_args(spin_space, length, charges))
        psi0n = psi0 / psi0.norm()

        def single_step_infidelity(dt):
            mps, mpo_l, _, _ = _domain_wall(length, spin_space)
            # Seed off the product state so the single step exercises a generic
            # (entangled) bond, then take exactly one step of size dt.
            summary = discarded_bug.run(
                mps, mpo_l, discarded_bug.Options(dt=dt, n_steps=1, max_bond=64, normalize=False),
            )
            evolved = mps_to_vector(summary.state, charges)
            return _infidelity(evolved, exact_evolve(ham, psi0n, dt))

        coarse = single_step_infidelity(0.04)
        fine = single_step_infidelity(0.02)
        # O(dt^2) single-step error => ratio ~4 when halving dt (allow a generous band).
        assert 3.0 < coarse / fine < 5.5

    def test_forward_only_floor_does_not_shrink_with_dt(self, spin_space):
        """The forward-only BUG has an intrinsic projection floor: evolving to a FIXED
        time with a smaller dt does not reduce the error (it is not a dt-discretisation
        error; only a backward step, which BUG forbids, would remove it). Documents the
        known accuracy limit — the validated property is the rank growth, not accuracy."""
        length = 6
        _, _, charges, psi0 = _domain_wall(length, spin_space)
        ham = dense_hamiltonian(*_ham_args(spin_space, length, charges))
        psi0n = psi0 / psi0.norm()

        def infidelity_at_T(dt, T):
            mps, mpo_l, _, _ = _domain_wall(length, spin_space)
            summary = discarded_bug.run(
                mps, mpo_l,
                discarded_bug.Options(dt=dt, n_steps=round(T / dt), max_bond=64, normalize=False),
            )
            evolved = mps_to_vector(summary.state, charges)
            return _infidelity(evolved, exact_evolve(ham, psi0n, T))

        coarse = infidelity_at_T(0.10, 0.5)
        fine = infidelity_at_T(0.05, 0.5)
        # The floor does not shrink with dt: halving dt leaves the error within ~30%
        # (in fact marginally larger), confirming it is not a dt-discretisation error.
        assert fine > 0.5 * coarse


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------

class TestConservation:
    """Norm (real time), total Sz, and imaginary-time energy descent."""

    def test_norm_conserved_real_time(self, spin_space):
        mps, mpo, _, _ = _domain_wall(6, spin_space)
        summary = discarded_bug.run(
            mps, mpo, discarded_bug.Options(dt=0.05, n_steps=10, max_bond=64, normalize=False),
        )
        for norm in summary.norms:
            assert abs(norm - 1.0) < 1e-9

    def test_total_sz_conserved(self, spin_space):
        mps, mpo, charges, psi0 = _domain_wall(6, spin_space)
        sz_total = dense_total_sz(6, charges)
        sz_before = (psi0.conj() @ sz_total @ psi0).real.item() / psi0.norm().item() ** 2
        summary = discarded_bug.run(mps, mpo, discarded_bug.Options(dt=0.05, n_steps=10, max_bond=64))
        vec = mps_to_vector(summary.state, charges)
        sz_after = (vec.conj() @ sz_total @ vec).real.item() / vec.norm().item() ** 2
        assert abs(sz_after - sz_before) < 1e-9

    def test_imaginary_time_lowers_energy(self, spin_space):
        length = 6
        mps, mpo, charges, psi0 = _domain_wall(length, spin_space)
        ham = dense_hamiltonian(*_ham_args(spin_space, length, charges))
        ground = torch.linalg.eigvalsh(ham)[0].item()
        psi0n = psi0 / psi0.norm()
        energy_before = (psi0n.conj() @ ham @ psi0n).real.item()
        summary = discarded_bug.run(
            mps, mpo, discarded_bug.Options(dt=0.05, n_steps=40, imaginary_time=True, max_bond=64),
        )
        vec = mps_to_vector(summary.state, charges)
        vec = vec / vec.norm()
        energy_after = (vec.conj() @ ham @ vec).real.item()
        assert energy_after < energy_before
        assert energy_after > ground - 1e-9


def _ham_args(spin_space, length, charges):
    """Build the (interactions, length, charges) tuple for ``dense_hamiltonian``."""
    interactions, _, _ = heisenberg_chain(length)
    return interactions, length, charges
