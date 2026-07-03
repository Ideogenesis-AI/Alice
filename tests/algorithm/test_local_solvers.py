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


"""Pluggable local-solver tests for the (imaginary-time) discarded-projector BUGs.

In imaginary time the local update ``y = exp(tau A) x`` is the exact flow of a
linear ODE, so it may be computed by any stable integrator instead of the exact
Krylov exponential. Both the two-site BUG (``variant='discarded'``) and the global
``discarded_bug`` expose ``solver`` / ``solver_substeps`` for this. These tests
check, end-to-end through the real symmetry-blocked tensor machinery, that:

* the substepped integrators (``midpoint``/``rk4``/``trapezoid``) reproduce the exact
  ``krylov`` evolution as ``solver_substeps`` grows (the screenshot's "increase n"),
  validating the explicit RK actions *and* the implicit Crank–Nicolson GMRES solve;
* every solver still cools a Néel state toward the exact ground state; and
* bad solver names are rejected at ``Options`` construction.
"""

from __future__ import annotations

import pytest
import torch
from nicole import Index, Tensor, load_space

from alice import build_hamiltonian, build_interaction, init_mps
from alice.algorithm import discarded_bug, two_site_bug
from alice.algorithm.two_site_bug._kernel.local_solvers import LOCAL_SOLVERS

from tests.algorithm.two_site_bug.conftest import (
    dense_hamiltonian,
    heisenberg_chain,
    mps_to_vector,
)

_LENGTH = 6


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope='module')
def spin_space():
    return load_space('Spin', 'U1', {'J': 0.5})


def _neel(length, spin_space):
    _, operators = spin_space
    interactions, spc, _ = heisenberg_chain(length)
    mpo = build_hamiltonian(interactions, length, spc)
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
    return mps, interactions, mpo, charges


def _two_site_state(spin_space, *, solver, substeps, n_steps=4, dt=0.05):
    mps, interactions, _, charges = _neel(_LENGTH, spin_space)
    state = two_site_bug.run(
        mps, interactions,
        two_site_bug.Options(variant='discarded', solver=solver, solver_substeps=substeps,
                             dt=dt, n_steps=n_steps, imaginary_time=True, max_bond=64),
    ).state
    vec = mps_to_vector(state, charges)
    return vec / vec.norm()


def _global_state(spin_space, *, solver, substeps, n_steps=4, dt=0.05):
    mps, _, mpo, charges = _neel(_LENGTH, spin_space)
    state = discarded_bug.run(
        mps, mpo,
        discarded_bug.Options(solver=solver, solver_substeps=substeps,
                              dt=dt, n_steps=n_steps, imaginary_time=True, max_bond=64),
    ).state
    vec = mps_to_vector(state, charges)
    return vec / vec.norm()


def _overlap_err(a, b):
    # Clamp at 0: when two states agree to machine precision, |<a|b>| can round to
    # just above 1 and give a tiny negative "error".
    return max(0.0, 1.0 - abs(torch.vdot(a, b)).item())


# ---------------------------------------------------------------------------
# Option validation
# ---------------------------------------------------------------------------

class TestSolverOptions:

    def test_known_solvers(self):
        assert set(LOCAL_SOLVERS) == {'krylov', 'midpoint', 'rk4', 'trapezoid'}

    def test_default_is_krylov(self):
        assert two_site_bug.Options().solver == 'krylov'
        assert discarded_bug.Options().solver == 'krylov'

    @pytest.mark.parametrize('factory', [two_site_bug.Options, discarded_bug.Options])
    def test_unknown_solver_raises(self, factory):
        with pytest.raises(ValueError, match='unknown local solver'):
            factory(solver='euler')


# ---------------------------------------------------------------------------
# Two-site BUG (variant='discarded'): K/L/S solves
# ---------------------------------------------------------------------------

class TestTwoSiteSolvers:
    """Substepped solvers reproduce the exact Krylov evolution as n grows."""

    @pytest.mark.parametrize('solver', ['midpoint', 'rk4', 'trapezoid'])
    def test_converges_to_krylov_with_substeps(self, solver, spin_space):
        ref = _two_site_state(spin_space, solver='krylov', substeps=1, n_steps=3)
        coarse = _two_site_state(spin_space, solver=solver, substeps=2, n_steps=3)
        fine = _two_site_state(spin_space, solver=solver, substeps=10, n_steps=3)
        err_coarse = _overlap_err(ref, coarse)
        err_fine = _overlap_err(ref, fine)
        # More substeps -> at least as close to the exact Krylov action (rk4 already
        # hits machine precision at n=2, so allow equality at the FP floor), and tight.
        assert err_fine <= err_coarse + 1e-12
        assert err_fine < 1e-4, f"{solver}: err_fine {err_fine:.2e}"


# ---------------------------------------------------------------------------
# Global discarded_bug: central Galerkin core solve
# ---------------------------------------------------------------------------

class TestGlobalSolvers:

    @pytest.mark.parametrize('solver', ['midpoint', 'rk4', 'trapezoid'])
    def test_converges_to_krylov_with_substeps(self, solver, spin_space):
        ref = _global_state(spin_space, solver='krylov', substeps=1, n_steps=3)
        coarse = _global_state(spin_space, solver=solver, substeps=2, n_steps=3)
        fine = _global_state(spin_space, solver=solver, substeps=10, n_steps=3)
        err_coarse = _overlap_err(ref, coarse)
        err_fine = _overlap_err(ref, fine)
        assert err_fine <= err_coarse + 1e-12
        assert err_fine < 1e-4, f"{solver}: not tight at n=10"


# ---------------------------------------------------------------------------
# Every solver cools toward the ground state
# ---------------------------------------------------------------------------

class TestCoolsWithEverySolver:

    @pytest.mark.slow
    @pytest.mark.parametrize('solver', ['krylov', 'midpoint', 'rk4', 'trapezoid'])
    def test_two_site_discarded_cools(self, solver, spin_space):
        mps, interactions, _, charges = _neel(_LENGTH, spin_space)
        ham = dense_hamiltonian(interactions, _LENGTH, charges)
        evals, evecs = torch.linalg.eigh(ham)
        ground_vec = evecs[:, 0]
        psi0 = mps_to_vector(mps, charges)
        psi0 = psi0 / psi0.norm()
        err_before = _overlap_err(ground_vec, psi0)
        vec = _two_site_state(spin_space, solver=solver, substeps=8, n_steps=100, dt=0.05)
        err_after = _overlap_err(ground_vec, vec)
        assert err_after < err_before
        assert err_after < 5e-2, f"{solver}: final overlap error {err_after:.2e}"
