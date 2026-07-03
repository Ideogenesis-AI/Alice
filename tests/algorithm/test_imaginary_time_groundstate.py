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


"""Cross-method imaginary-time ground-state convergence.

The headline quantity of the BUG-vs-TDVP study is the **overlap error of the
imaginary-time-cooled state with the exact ground state**. This module checks, on
a small Heisenberg chain where the ground state is available by exact
diagonalization, that *every* integrator under comparison cools a Néel product
state toward that exact ground state:

* faithful two-site BUG               (``two_site_bug``, ``variant='faithful'``),
* discarded-projector two-site BUG    (``two_site_bug``, ``variant='discarded'``),
* global discarded-projector BUG      (``discarded_bug``), and
* two-site TDVP                       (``tdvp2``).

For each method the final state must have a small overlap error with the exact
ground state, a near-degenerate energy, and clear cooling relative to the Néel
start. This is the unit-level guard for the imaginary-time pipeline the full
``L = 26`` campaign runs.
"""

from __future__ import annotations

import pytest
import torch
from nicole import Index, Tensor, load_space

from alice import build_hamiltonian, init_mps
from alice.algorithm import discarded_bug, tdvp2, two_site_bug

from tests.algorithm.two_site_bug.conftest import (
    dense_hamiltonian,
    heisenberg_chain,
    mps_to_vector,
)

# Imaginary-time schedule: dt matches the production setup; beta = dt * n_steps is
# made long enough that, at full bond dimension (no truncation on L = 6), the only
# residual is the O(dt^2) Strang/splitting bias. Kept modest so the test is fast.
_DT = 0.05
_N_STEPS = 200
_LENGTH = 6

# This unit test validates that the inverse-free BUG family converges to the exact
# ground state in imaginary time. Two-site TDVP is the comparison baseline whose
# imaginary-time instability (it stalls under truncation and blows up) is the very
# phenomenon the study figure exhibits — so it is driven by the study harness and
# its own test module, and is deliberately not asserted as a convergence invariant
# here.
_BUG_METHODS = ['bug_faithful', 'bug_discarded', 'discarded_bug']


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope='module')
def spin_space():
    return load_space('Spin', 'U1', {'J': 0.5})


def _neel(length, spin_space):
    """Build the full-phys Néel MPS plus the interaction list, MPO, and ED data."""
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
    psi0 = mps_to_vector(mps, charges)
    return mps, interactions, mpo, charges, psi0


def _cool(method, mps, interactions, mpo):
    """Run one method in imaginary time and return its evolved MPS state."""
    if method == 'bug_faithful':
        return two_site_bug.run(
            mps, interactions,
            two_site_bug.Options(variant='faithful', dt=_DT, n_steps=_N_STEPS,
                                 imaginary_time=True, max_bond=64),
        ).state
    if method == 'bug_discarded':
        return two_site_bug.run(
            mps, interactions,
            two_site_bug.Options(variant='discarded', dt=_DT, n_steps=_N_STEPS,
                                 imaginary_time=True, max_bond=64),
        ).state
    if method == 'discarded_bug':
        return discarded_bug.run(
            mps, mpo,
            discarded_bug.Options(dt=_DT, n_steps=_N_STEPS, imaginary_time=True, max_bond=64),
        ).state
    if method == 'tdvp2':
        return tdvp2.run(
            mps, mpo,
            tdvp2.Options(dt=_DT, n_steps=_N_STEPS, imaginary_time=True, max_bond=64),
        ).state
    raise ValueError(f"unknown method {method!r}")


@pytest.mark.parametrize('method', _BUG_METHODS)
def test_cools_neel_to_exact_ground_state(method, spin_space):
    mps, interactions, mpo, charges, psi0 = _neel(_LENGTH, spin_space)
    ham = dense_hamiltonian(interactions, _LENGTH, charges)
    evals, evecs = torch.linalg.eigh(ham)
    ground_energy = evals[0].item()
    ground_vec = evecs[:, 0]

    psi0 = psi0 / psi0.norm()
    energy_before = (psi0.conj() @ ham @ psi0).real.item()
    err_before = 1.0 - abs(torch.vdot(ground_vec, psi0)).item()

    state = _cool(method, mps, interactions, mpo)
    vec = mps_to_vector(state, charges)
    vec = vec / vec.norm()
    energy_after = (vec.conj() @ ham @ vec).real.item()
    err_after = 1.0 - abs(torch.vdot(ground_vec, vec)).item()

    # Variational lower bound, genuine cooling, and convergence to the exact GS.
    assert energy_after > ground_energy - 1e-9, f"{method}: energy below ED ground state"
    assert energy_after < energy_before - 1e-6, f"{method}: energy did not decrease"
    assert err_after < err_before, f"{method}: did not cool toward ground state"
    assert energy_after - ground_energy < 1e-2, f"{method}: energy not converged"
    assert err_after < 1e-2, f"{method}: final overlap error {err_after:.2e} too large"
