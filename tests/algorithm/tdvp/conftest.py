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


"""Pytest fixtures for TDVP algorithm tests.

Every fixture hands back complex128 tensors, since real-time evolution cannot
be represented in real arithmetic and `run()` is the only place that performs
the promotion automatically.

Chains are kept short (L = 4 … 8). A TDVP step is a full sweep of Krylov
exponentials with no inner convergence loop to cut short, so test cost grows
quickly with both `L` and the number of steps.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from nicole import Direction, Tensor, load_space
from nicole.index import Index, Sector

from nicole import einsum

from alice.algorithm.tdvp.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
)
from alice.network import MPS, build_hamiltonian, build_interaction
from alice.network.interaction import Interaction1Site
from alice.physics.system import build_fermionic


# ---------------------------------------------------------------------------
# Working-directory isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Run every test with cwd set to a fresh tmp_path.

    This prevents checkpoint files (and any other cwd-relative artefacts)
    from accumulating in the project root during the test session.
    """
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Physical spaces
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def spin_space():
    """Spin-1/2 U(1) physical space and operators (shared across session)."""
    return load_space('Spin', 'U1', {'J': 0.5})


@pytest.fixture(scope='session')
def fermion_space():
    """Spinless-fermion U(1) physical space and MPO operator templates."""
    return build_fermionic('U1')


# ---------------------------------------------------------------------------
# Helper: build a random complex MPS
# ---------------------------------------------------------------------------

def _random_mps(spin_space, L: int, bond_sectors, bond_dim: int, seed_offset: int = 0):
    """Build a random complex MPS with the given bond sector structure.

    Parameters
    ----------
    spin_space:
        `(Spc, Op)` from `load_space`.
    L:
        Number of sites.
    bond_sectors:
        Tuple of integer charges for the bulk bond index.
    bond_dim:
        Number of states per sector.
    seed_offset:
        Offset added to the per-site seed for reproducibility.

    Returns
    -------
    MPS
        Normalized complex128 MPS with `center == 0`.
    """
    Spc, Op = spin_space
    vac = Op['vac']
    bulk = Index(
        direction=Direction.IN,
        group=Spc.group,
        sectors=tuple(Sector(charge=q, dim=bond_dim) for q in bond_sectors),
    )
    tensors = []
    for i in range(L):
        l_idx = vac if i == 0 else bulk
        r_idx = (vac if i == L - 1 else bulk).flip()
        tensors.append(Tensor.random(
            [l_idx, r_idx, Spc],
            seed=seed_offset + i,
            dtype=torch.complex128,
            itags=[f'A{i:02d}', f'A{i + 1:02d}', f's{i:02d}'],
        ))
    mps = MPS(tensors, center=None)
    mps.canonical(0)
    mps.normalize()
    return mps


def _heisenberg_mpo(L: int):
    """Build the Heisenberg spin-1/2 OBC chain MPO, promoted to complex."""
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': L, 'bcx': 'OBC', 'n2x': True},
        'model': {
            'category': 'bosonic', 'label': 'Heisenberg',
            'symmetry': 'U1', 'spin': 0.5, 'J': 1.0,
        },
    }
    interactions, spc, geo = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, geo.L, spc)
    mpo.to(dtype=torch.complex128)
    return mpo


# ---------------------------------------------------------------------------
# Heisenberg chains
# ---------------------------------------------------------------------------

@pytest.fixture
def heisenberg_L2(spin_space):
    """Heisenberg spin-1/2 chain, L=2, OBC, U(1), J=1.

    Returns `(mps, mpo)` with a complex normalized `mps` at `center == 0`.
    The smallest system on which an effective Hamiltonian exists, used for
    the cheap algebraic unit tests.
    """
    mpo = _heisenberg_mpo(2)
    mps = _random_mps(spin_space, 2, bond_sectors=(-1, 1), bond_dim=1, seed_offset=200)
    return mps, mpo


@pytest.fixture
def heisenberg_L4(spin_space):
    """Heisenberg spin-1/2 chain, L=4, OBC, U(1), J=1.

    Returns `(mps, mpo)` with a complex normalized `mps` at `center == 0`.
    Long enough to have interior environment blocks, so it is the fixture the
    disk-cache and sliding-window tests run on.
    """
    mpo = _heisenberg_mpo(4)
    mps = _random_mps(spin_space, 4, bond_sectors=tuple(range(-4, 5)), bond_dim=2,
                      seed_offset=100)
    return mps, mpo


@pytest.fixture
def heisenberg_L6(spin_space):
    """Heisenberg spin-1/2 chain, L=6, OBC, U(1), J=1.

    Returns `(mps, mpo)` with a complex normalized `mps` at `center == 0` and
    bond dimensions well below the exact rank, so the variational manifold is
    genuinely curved and the integrator order is measurable.
    """
    mpo = _heisenberg_mpo(6)
    mps = _random_mps(spin_space, 6, bond_sectors=tuple(range(-3, 4)), bond_dim=1,
                      seed_offset=300)
    return mps, mpo


@pytest.fixture
def heisenberg_L4_real(spin_space):
    """Heisenberg L=4 with a float64 state and a float64 MPO.

    The one fixture that is deliberately real, so the driver's promotion to
    complex is exercised on an input that could not evolve otherwise.
    """
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': 4, 'bcx': 'OBC', 'n2x': True},
        'model': {
            'category': 'bosonic', 'label': 'Heisenberg',
            'symmetry': 'U1', 'spin': 0.5, 'J': 1.0,
        },
    }
    interactions, spc, geo = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, geo.L, spc)

    Spc, Op = spin_space
    vac = Op['vac']
    bulk = Index(
        direction=Direction.IN,
        group=Spc.group,
        sectors=tuple(Sector(charge=q, dim=2) for q in range(-4, 5)),
    )
    tensors = [
        Tensor.random(
            [vac if i == 0 else bulk, (vac if i == 3 else bulk).flip(), Spc],
            seed=700 + i,
            itags=[f'A{i:02d}', f'A{i + 1:02d}', f's{i:02d}'],
        )
        for i in range(4)
    ]
    mps = MPS(tensors, center=None)
    mps.canonical(0)
    mps.normalize()
    return mps, mpo


@pytest.fixture
def product_state_L4(spin_space):
    """Heisenberg L=4 with an antiferromagnetic product state.

    Every bond has dimension 1, which is the configuration the 1-site scheme
    cannot escape, so this is what the frozen-bond warning is tested against.
    """
    from alice.network import init_mps

    Spc, Op = spin_space
    mpo = _heisenberg_mpo(4)
    mps = init_mps(4, Spc, Op, bond_dim=1, config=[1, 0, 1, 0],
                   dtype=torch.complex128)
    mps.canonical(0)
    mps.normalize()
    return mps, mpo


# ---------------------------------------------------------------------------
# Free-fermion chain — the analytically solvable benchmark
# ---------------------------------------------------------------------------

FREE_FERMION_L = 8
FREE_FERMION_SITE = 4
FREE_FERMION_T = 1.0


@pytest.fixture(scope='session')
def free_fermion(fermion_space):
    """Single particle on an open tight-binding chain.

    `H = -t Σ_j (c†_j c_{j+1} + h.c.)` with `t = 1`, built through
    `build_free_fermion` so the Jordan-Wigner strings are exercised. In the
    one-particle sector this reduces to the `L x L` hopping matrix, whose OBC
    eigenbasis is closed-form, making site occupations an analytic reference.

    Returns
    -------
    tuple
        `(mpo, n_obs, make_state)` where `n_obs[j]` is an MPO measuring the
        occupation of site `j` and `make_state()` returns a fresh product
        state with the particle localized at `FREE_FERMION_SITE`.
    """
    _, ops = fermion_space
    L = FREE_FERMION_L
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': L, 'bcx': 'OBC', 'n2x': True},
        'model': {
            'category': 'fermionic', 'label': 'FreeFermion',
            'symmetry': 'U1', 't': FREE_FERMION_T,
        },
    }
    interactions, spc, geo = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, geo.L, spc)

    n_obs = [
        build_hamiltonian(
            [Interaction1Site(cpl=1.0, label=['N', 'onsite'], site=j,
                              tnsr=ops['N4'].clone())],
            L, spc,
        )
        for j in range(L)
    ]

    def make_state():
        """Fresh complex product state with one particle at the center."""
        from alice.network import init_mps

        config = [0] * L
        config[FREE_FERMION_SITE] = 1
        return init_mps(L, spc, ops, bond_dim=1, config=config,
                        dtype=torch.complex128)

    return mpo, n_obs, make_state


def analytic_occupations(t: float) -> np.ndarray:
    """Exact site occupations of the one-particle tight-binding chain at `t`.

    The open-chain single-particle eigenbasis is
    `phi_k(j) = sqrt(2/(L+1)) sin(pi k (j+1)/(L+1))` with energies
    `eps_k = -2 t cos(pi k/(L+1))`, so the propagator needs no diagonalization.

    Parameters
    ----------
    t:
        Evolution time.

    Returns
    -------
    numpy.ndarray
        Occupation of each site, length `FREE_FERMION_L`.
    """
    L = FREE_FERMION_L
    k = np.arange(1, L + 1)[:, None]
    j = np.arange(L)[None, :]
    phi = np.sqrt(2.0 / (L + 1)) * np.sin(np.pi * k * (j + 1) / (L + 1))
    eps = -2.0 * FREE_FERMION_T * np.cos(np.pi * k[:, 0] / (L + 1))
    amp = phi.T @ (np.exp(-1j * eps * t) * phi[:, FREE_FERMION_SITE])
    return np.abs(amp) ** 2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def mps_overlap(a: MPS, b: MPS) -> complex:
    """Compute `⟨a|b⟩` by a left-to-right transfer-matrix sweep.

    Parameters
    ----------
    a:
        Bra MPS (conjugated here).
    b:
        Ket MPS, with the same index structure as `a`.

    Returns
    -------
    complex
        The overlap, not normalized by either state's norm.
    """
    E = None
    for i in range(a.L):
        E = (einsum('abr,acr->bc', a[i].conj(), b[i]) if E is None
             else einsum('bc,bdr,cer->de', E, a[i].conj(), b[i]))
    if not E.data:
        return 0.0 + 0.0j
    key, val = next(iter(E.data.items()))
    weight = 1.0 if E.intw is None else complex(E.intw[key].weights[0, 0].item())
    return complex(val.item()) * weight


def bond_profile(mps: MPS) -> list:
    """Return the dimension of every internal bond, left to right."""
    return [mps[i].indices[1].dim for i in range(mps.L - 1)]


def fresh_envs(mps: MPS, mpo, **kwargs):
    """Build a `(env_left, env_right)` pair seeded for a forward sweep.

    Parameters
    ----------
    mps:
        State with `center == 0`.
    mpo:
        Hamiltonian MPO.
    **kwargs:
        Forwarded to both `Environment` constructors.

    Returns
    -------
    tuple
        `(env_left, env_right)` with `env_left[0]` and every right block set.
    """
    env_left = Environment(mps.L, **kwargs)
    env_right = Environment(mps.L, **kwargs)
    env_left[0] = left_env_boundary(mps, mpo)
    build_right_envs(mps, mpo, env_right)
    return env_left, env_right
