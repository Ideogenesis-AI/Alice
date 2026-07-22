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


"""Tests for the `bond_update_bug` local K/L/S bond update.

The update (see :mod:`alice.algorithm.bond_update_bug._kernel.kls.candidate`)
projects the K/L generators by the discarded (orthogonal-complement) projector
*before* the exponential and acts the augmented **isometries directly** in the
S-step (``Ŝ0 = Û† Θ0 V̂†``), forming no overlap matrices. These tests check it
against exact diagonalization, the conservation laws (norm and total Sz), the
second-order Strang convergence, and imaginary-time cooling to the ground state.
"""

from __future__ import annotations

import pytest
import torch
from nicole import Index, Tensor

from alice import init_mps
from alice.algorithm import bond_update_bug

from .conftest import (
    dense_hamiltonian,
    dense_sz_profile,
    dense_total_sz,
    exact_evolve,
    heisenberg_chain,
    mps_to_vector,
)


def _neel(length, spin_space):
    """Return ``(mps, interactions, charges, psi0)`` for a full-phys Néel state.

    The Néel state ``|↑↓↑↓…⟩`` (alternating ``config=[0,1,0,1,…]``) lives in the
    Sz=0 sector for even ``length``. Each physical leg is inflated to the full
    spin-1/2 index so spins can flip and the state densifies to ``2**length``;
    ``psi0`` is the dense initial vector in the ED helpers' basis order.
    """
    _, operators = spin_space
    interactions, spc, _ = heisenberg_chain(length)
    charges = [sector.charge for sector in spc.sectors]
    config = [0, 1] * (length // 2)
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


def _opts(**kwargs):
    """`bond_update_bug` options with sensible test defaults."""
    return bond_update_bug.Options(**kwargs)


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

class TestConservation:
    """Norm (real time) and total Sz are conserved by the discarded S-step."""

    def test_norm_conserved_real_time(self, spin_space):
        mps, interactions, _, _ = _neel(6, spin_space)
        summary = bond_update_bug.run(
            mps, interactions, _opts(dt=0.05, n_steps=10, max_bond=64, normalize=False)
        )
        for norm in summary.norms:
            assert abs(norm - 1.0) < 1e-10

    def test_total_sz_conserved(self, spin_space):
        mps, interactions, charges, psi0 = _neel(6, spin_space)
        sz_total = dense_total_sz(6, charges)
        sz_before = (psi0.conj() @ sz_total @ psi0).real.item() / psi0.norm().item() ** 2
        summary = bond_update_bug.run(
            mps, interactions, _opts(dt=0.05, n_steps=10, max_bond=64)
        )
        vec = mps_to_vector(summary.state, charges)
        sz_after = (vec.conj() @ sz_total @ vec).real.item() / vec.norm().item() ** 2
        assert abs(sz_after - sz_before) < 1e-10


# ---------------------------------------------------------------------------
# Accuracy: the discarded-projector + augmented-isometry S-step is correct
# ---------------------------------------------------------------------------

class TestAccuracy:
    """Real-time accuracy of the discarded S-step against ED and the kernel."""

    def test_fidelity_matches_exact_diagonalization(self, spin_space):
        length = 6
        mps, interactions, charges, psi0 = _neel(length, spin_space)
        ham = dense_hamiltonian(interactions, length, charges)
        psi0 = psi0 / psi0.norm()
        dt, n_steps = 0.05, 20
        summary = bond_update_bug.run(
            mps, interactions, _opts(dt=dt, n_steps=n_steps, max_bond=64, normalize=False)
        )
        evolved = mps_to_vector(summary.state, charges)
        evolved = evolved / evolved.norm()
        exact = exact_evolve(ham, psi0, dt * n_steps)
        exact = exact / exact.norm()
        fidelity = abs(torch.vdot(exact, evolved)).item()
        # At full bond dimension the only error is the Strang splitting (O(dt^2)).
        assert 1.0 - fidelity < 1e-6

    def test_strang_converges_second_order(self, spin_space):
        """Strang state error is O(dt^2) -> infidelity O(dt^4): halving dt cuts ~16x.

        Measured in the ASYMPTOTIC regime. At dt=0.1 the higher-order Trotter terms
        are still large enough to contaminate the ratio (it reads ~6 at every system
        size, L=4/6/8 alike), which measures how far dt is from asymptotia rather
        than the method's order. Halving into dt=0.05/0.025 recovers the expected
        behaviour. Verified against a dt/L scan: the ratio rises monotonically
        towards 16 as dt shrinks (L=6: 6.33 -> 10.48 -> 14.63 at T=1.0/0.5/0.2),
        which is the signature of a genuine second-order method; a rank-projection
        floor would push the ratio DOWN as the Trotter error vanished, not up.
        """
        length = 6
        _, interactions, charges, psi0 = _neel(length, spin_space)
        ham = dense_hamiltonian(interactions, length, charges)
        psi0 = psi0 / psi0.norm()

        def infidelity(dt, n_steps):
            mps, _, _, _ = _neel(length, spin_space)
            summary = bond_update_bug.run(
                mps, interactions, _opts(dt=dt, n_steps=n_steps, max_bond=64, normalize=False)
            )
            evolved = mps_to_vector(summary.state, charges)
            evolved = evolved / evolved.norm()
            exact = exact_evolve(ham, psi0, dt * n_steps)
            exact = exact / exact.norm()
            return 1.0 - abs(torch.vdot(exact, evolved)).item()

        coarse = infidelity(0.05, 10)
        fine = infidelity(0.025, 20)
        assert coarse / fine > 8.0


# ---------------------------------------------------------------------------
# Imaginary time
# ---------------------------------------------------------------------------

class TestImaginaryTime:
    """Imaginary-time cooling toward the exact ground state."""

    def test_imaginary_time_reaches_ground_state(self, spin_space):
        length = 6
        mps, interactions, charges, psi0 = _neel(length, spin_space)
        ham = dense_hamiltonian(interactions, length, charges)
        evals, evecs = torch.linalg.eigh(ham)
        ground_energy = evals[0].item()
        ground_vec = evecs[:, 0]
        psi0 = psi0 / psi0.norm()
        err_before = 1.0 - abs(torch.vdot(ground_vec, psi0)).item()

        summary = bond_update_bug.run(
            mps, interactions,
            _opts(dt=0.05, n_steps=160, imaginary_time=True, max_bond=64),
        )
        vec = mps_to_vector(summary.state, charges)
        vec = vec / vec.norm()
        energy_after = (vec.conj() @ ham @ vec).real.item()
        err_after = 1.0 - abs(torch.vdot(ground_vec, vec)).item()

        # Variational lower bound, substantial cooling, and tight final overlap.
        assert energy_after > ground_energy - 1e-9
        assert energy_after - ground_energy < 1e-2
        assert err_after < err_before
        assert err_after < 1e-2


