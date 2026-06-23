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


"""Tests for the 2-site TDVP integrator (Options, Summary, run).

The 2-site TDVP integrator (see :mod:`alice.algorithm.tdvp2`) evolves an `MPS`
under a Hamiltonian `MPO` by symmetric forward/reverse half-sweeps of effective-
Hamiltonian exponentials, with an inverse-free one-site backward correction. These
tests check, on the symmetric (isotropic) Heisenberg chain — which conserves total
Sz and is available by exact diagonalization — that the integrator:

* grows the bond dimension as a domain wall melts (rank adaptivity),
* tracks the analytical (exact-diagonalization) solution to high accuracy,
* conserves the state norm to machine precision (real time; TDVP is unitary),
* conserves total Sz,
* cools toward the ground state in imaginary time.

A note on convergence: at fixed/adaptive bond dimension, 2-site TDVP's error is a
projection (manifold) error that does **not** vanish as ``dt → 0`` — it plateaus,
unlike the Strang state error of a full-rank propagator. The accuracy tests
therefore assert a bounded, non-increasing error rather than a strict O(dt^2) ratio.
"""

from __future__ import annotations

import pytest
import torch
from nicole import Index, Tensor

from alice import build_hamiltonian, init_mps
from alice.algorithm import tdvp2

from .conftest import (
    dense_heisenberg,
    dense_total_sz,
    exact_evolve,
    heisenberg_chain,
    mps_to_vector,
)


def _domain_wall(length, spin_space):
    """Return `(mps, mpo, charges, psi0)` for a full-phys Heisenberg domain wall."""
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
    mpo = build_hamiltonian(interactions, length, spc)
    psi0 = mps_to_vector(mps, charges)
    return mps, mpo, charges, psi0


# ---------------------------------------------------------------------------
# Options / Summary
# ---------------------------------------------------------------------------

class TestOptions:
    """Options/Summary basics."""

    def test_defaults(self):
        opts = tdvp2.Options()
        assert opts.dt == 0.05
        assert opts.max_bond is None
        assert opts.normalize is True

    def test_serialize_round_trip(self, spin_space):
        mps, mpo, _, _ = _domain_wall(6, spin_space)
        summary = tdvp2.run(mps, mpo, tdvp2.Options(dt=0.05, n_steps=3, max_bond=16))
        restored = tdvp2.Summary.deserialize(summary.serialize())
        assert restored.n_steps == summary.n_steps
        assert restored.bond_dims == summary.bond_dims
        assert restored.times == pytest.approx(summary.times)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_length_mismatch_raises(self, spin_space):
        mps6, mpo6, _, _ = _domain_wall(6, spin_space)
        _, mpo4, _, _ = _domain_wall(4, spin_space)
        with pytest.raises(ValueError, match='same length'):
            tdvp2.run(mps6, mpo4, tdvp2.Options(dt=0.05, n_steps=1))


# ---------------------------------------------------------------------------
# Rank adaptivity
# ---------------------------------------------------------------------------

class TestRankAdaptivity:
    def test_bond_dimension_grows(self, spin_space):
        mps, mpo, _, _ = _domain_wall(6, spin_space)
        assert max(mps.bond_dims) == 1
        summary = tdvp2.run(mps, mpo, tdvp2.Options(dt=0.05, n_steps=10, max_bond=64))
        assert max(summary.bond_dims) > 1
        assert max(summary.max_bond_dims) >= 4

    def test_max_bond_cap_respected(self, spin_space):
        mps, mpo, _, _ = _domain_wall(6, spin_space)
        cap = 4
        summary = tdvp2.run(mps, mpo, tdvp2.Options(dt=0.05, n_steps=10, max_bond=cap))
        assert max(summary.bond_dims) <= cap


# ---------------------------------------------------------------------------
# Accuracy vs exact diagonalization
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_fidelity_matches_exact_diagonalization(self, spin_space):
        length = 6
        mps, mpo, charges, psi0 = _domain_wall(length, spin_space)
        ham = dense_heisenberg(length, charges)
        psi0 = psi0 / psi0.norm()
        dt, n_steps = 0.02, 25
        summary = tdvp2.run(
            mps, mpo, tdvp2.Options(dt=dt, n_steps=n_steps, max_bond=64, normalize=False)
        )
        evolved = mps_to_vector(summary.state, charges)
        evolved = evolved / evolved.norm()
        exact = exact_evolve(ham, psi0, dt * n_steps)
        exact = exact / exact.norm()
        assert 1.0 - abs(torch.vdot(exact, evolved)).item() < 1e-5

    def test_error_is_bounded_and_non_increasing(self, spin_space):
        """2-site TDVP's error is a manifold-projection error: it does NOT vanish as
        dt→0 (it plateaus), but it must stay small and not GROW as dt shrinks."""
        length = 6
        mps0, mpo, charges, psi0 = _domain_wall(length, spin_space)
        ham = dense_heisenberg(length, charges)
        psi0 = psi0 / psi0.norm()

        def infidelity(dt, n):
            mps, _, _, _ = _domain_wall(length, spin_space)
            tdvp2.run(mps, mpo, tdvp2.Options(dt=dt, n_steps=n, max_bond=64, normalize=False))
            v = mps_to_vector(mps, charges)
            v = v / v.norm()
            ex = exact_evolve(ham, psi0, dt * n)
            ex = ex / ex.norm()
            return 1.0 - abs(torch.vdot(ex, v)).item()

        coarse = infidelity(0.1, 5)
        fine = infidelity(0.05, 10)
        assert coarse < 1e-4 and fine < 1e-4          # both small
        assert fine <= coarse * 1.5                   # does not grow as dt shrinks


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------

class TestConservation:
    def test_norm_conserved_real_time(self, spin_space):
        mps, mpo, _, _ = _domain_wall(6, spin_space)
        summary = tdvp2.run(mps, mpo, tdvp2.Options(dt=0.05, n_steps=10, max_bond=64, normalize=False))
        for norm in summary.norms:
            assert abs(norm - 1.0) < 1e-9

    def test_total_sz_conserved(self, spin_space):
        mps, mpo, charges, psi0 = _domain_wall(6, spin_space)
        sz_total = dense_total_sz(6, charges)
        sz_before = (psi0.conj() @ sz_total @ psi0).real.item() / psi0.norm().item() ** 2
        summary = tdvp2.run(mps, mpo, tdvp2.Options(dt=0.05, n_steps=10, max_bond=64))
        vec = mps_to_vector(summary.state, charges)
        sz_after = (vec.conj() @ sz_total @ vec).real.item() / vec.norm().item() ** 2
        assert abs(sz_after - sz_before) < 1e-9

    def test_imaginary_time_lowers_energy(self, spin_space):
        length = 6
        mps, mpo, charges, psi0 = _domain_wall(length, spin_space)
        ham = dense_heisenberg(length, charges)
        ground = torch.linalg.eigvalsh(ham)[0].item()
        psi0 = psi0 / psi0.norm()
        energy_before = (psi0.conj() @ ham @ psi0).real.item()
        summary = tdvp2.run(
            mps, mpo, tdvp2.Options(dt=0.05, n_steps=40, imaginary_time=True, max_bond=64)
        )
        vec = mps_to_vector(summary.state, charges)
        vec = vec / vec.norm()
        energy_after = (vec.conj() @ ham @ vec).real.item()
        assert energy_after < energy_before
        assert energy_after > ground - 1e-9
