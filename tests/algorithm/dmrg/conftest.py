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


"""Pytest fixtures for DMRG algorithm tests."""

from __future__ import annotations

import pytest
from nicole import Direction, Tensor, load_space
from nicole.index import Index, Sector

from alice.network import MPS, MPO, build_hamiltonian, build_interaction


# ---------------------------------------------------------------------------
# Physical space
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def spin_space():
    """Spin-1/2 U(1) physical space and operators (shared across session)."""
    return load_space("Spin", "U1", {"J": 0.5})


# ---------------------------------------------------------------------------
# Helper: build an MPS from a physical space for a given chain length
# ---------------------------------------------------------------------------

def _random_mps(spin_space, L: int, bond_sectors, bond_dim: int, seed_offset: int = 0):
    """Build a random MPS with the given bond sector structure.

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
    """
    Spc, Op = spin_space
    vac = Op["vac"]
    bulk = Index(
        direction=Direction.IN,
        group=Spc.group,
        sectors=tuple(Sector(charge=q, dim=bond_dim) for q in bond_sectors),
    )
    tensors = []
    for i in range(L):
        l_idx = vac if i == 0 else bulk
        r_idx = (vac if i == L - 1 else bulk).flip()
        T = Tensor.random(
            [l_idx, r_idx, Spc],
            seed=seed_offset + i,
            itags=[f'A{i:02d}', f'A{i + 1:02d}', f's{i:02d}'],
        )
        tensors.append(T)
    mps = MPS(tensors, center=None)
    mps.canonical(0)
    return mps


# ---------------------------------------------------------------------------
# L=4 Heisenberg chain (main end-to-end test system)
# ---------------------------------------------------------------------------

@pytest.fixture
def heisenberg_L4(spin_space):
    """Heisenberg spin-1/2 chain, L=4, OBC, U(1), J=1.

    Returns `(mps, mpo)` where `mps.center == 0`. The known exact
    ground-state energy is approximately -1.6160254.
    """
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': 4, 'bcx': 'OBC', 'n2x': True},
        'model': {
            'category': 'bosonic', 'label': 'Heisenberg',
            'symmetry': 'U1', 'spin': 0.5, 'J': 1.0,
        },
    }
    interactions, spc, L = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, L, spc)
    # bond_dim=2 per sector gives the middle bond dim=4, enough for the exact
    # ground state (which needs sector-0 dim=2 at the middle bond).
    mps = _random_mps(spin_space, L, bond_sectors=tuple(range(-4, 5)), bond_dim=2, seed_offset=100)
    return mps, mpo


# ---------------------------------------------------------------------------
# L=2 Heisenberg chain (cheap unit tests for sweep / environ)
# ---------------------------------------------------------------------------

@pytest.fixture
def heisenberg_L2(spin_space):
    """Heisenberg spin-1/2 chain, L=2, OBC, U(1), J=1.

    Returns `(mps, mpo)` where `mps.center == 0`. The exact ground-state
    energy is -0.75 (two-site singlet).
    """
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': 2, 'bcx': 'OBC', 'n2x': True},
        'model': {
            'category': 'bosonic', 'label': 'Heisenberg',
            'symmetry': 'U1', 'spin': 0.5, 'J': 1.0,
        },
    }
    interactions, spc, L = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, L, spc)
    # For L=2 the mid-bond sectors can be ±1 only (one spin).
    mps = _random_mps(spin_space, L, bond_sectors=(-1, 1), bond_dim=1, seed_offset=200)
    return mps, mpo
