# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Alice library.
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


"""Pytest configuration and fixtures for alice tests."""

from __future__ import annotations

import logging

import pytest
from nicole import Direction, Tensor, load_space
from nicole.index import Index, Sector


@pytest.fixture(autouse=True)
def configure_logging():
    """Configure logging to suppress info messages during tests."""
    logging.getLogger('alice.physics.geometry').setLevel(logging.WARNING)


# ------------------------------------------------------------------
# Network / MPS / MPO fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope='session')
def spin_space():
    """Spin-1/2 U(1) physical space and operators (shared across session)."""
    return load_space("Spin", "U1", {"J": 0.5})


_L = 10   # chain length for both fixtures
# Bulk bond: sectors {-3, …, 3} each with dim 2 → bond dimension 14.
_BULK_SECTORS = tuple(range(-3, 4))
_BULK_DIM_PER_SECTOR = 2


@pytest.fixture
def mps_tensors(spin_space):
    """Fresh L=10 random MPS tensors for each test.

    Site *i* has axes `(left_bond, right_bond, physical)` with physical itag
    `s{i:02d}`. Bond itags follow the MPS convention: the left bond of site
    *i* carries itag `A{i:02d}` and the right bond carries `A{i+1:02d}`.
    Interior bonds carry U(1) sectors {-3, …, 3} each with dimension 2 (total
    bond dim 14). Boundary bonds use the vacuum index from `spin_space` (dim
    1). A fixed seed per site makes tests reproducible.
    """
    Spc, Op = spin_space
    bulk = Index(direction=Direction.IN, group=Spc.group,
                 sectors=tuple(Sector(charge=q, dim=_BULK_DIM_PER_SECTOR)
                               for q in _BULK_SECTORS))
    vac = Op["vac"]
    tensors = []
    for i in range(_L):
        l = vac  if i == 0      else bulk
        r = (vac if i == _L - 1 else bulk).flip()
        T = Tensor.random([l, r, Spc], seed=i,
                          itags=[f'A{i:02d}', f'A{i + 1:02d}', f's{i:02d}'])
        tensors.append(T)
    return tensors


@pytest.fixture
def mpo_tensors(spin_space):
    """Fresh L=10 random MPO tensors for each test.

    Site *i* has axes `(left_bond, right_bond, phys_in, phys_out)` with both
    physical axes carrying itag `s{i:02d}` and opposite directions (IN/OUT).
    Bond itags follow the MPO convention: the left bond of site *i* carries
    itag `W{i:02d}` and the right bond carries `W{i+1:02d}`. Same bulk bond
    structure as `mps_tensors` (bond dim 14).
    """
    Spc, Op = spin_space
    bulk = Index(direction=Direction.IN, group=Spc.group,
                 sectors=tuple(Sector(charge=q, dim=_BULK_DIM_PER_SECTOR)
                               for q in _BULK_SECTORS))
    vac = Op["vac"]
    tensors = []
    for i in range(_L):
        l = vac  if i == 0      else bulk
        r = (vac if i == _L - 1 else bulk).flip()
        W = Tensor.random([l, r, Spc, Spc.flip()], seed=100 + i,
                          itags=[f'W{i:02d}', f'W{i + 1:02d}', f's{i:02d}', f's{i:02d}'])
        tensors.append(W)
    return tensors


# ------------------------------------------------------------------
# SU(2) Network / MPS / MPO fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope='session')
def spin_space_su2():
    """Spin-1/2 SU(2) physical space and operators (shared across session)."""
    return load_space("Spin", "SU2", {"J": 0.5})


_SU2_BULK_SECTORS       = (0, 1, 2)  # j = 0, 1/2, 1 in 2j notation
_SU2_BULK_DIM_PER_SECTOR = 2         # 2 multiplets per irrep → total dim = 6


@pytest.fixture
def mps_tensors_su2(spin_space_su2):
    """Fresh L=10 random SU(2) MPS tensors for each test.

    Site *i* has axes `(left_bond, right_bond, physical)` with physical itag
    `s{i:02d}`. Bond itags follow the MPS convention: the left bond of site
    *i* carries itag `A{i:02d}` and the right bond carries `A{i+1:02d}`.
    Interior bonds carry SU(2) sectors j=0, 1/2, 1 (charges 0, 1, 2 in the
    2j convention) each with 2 multiplets. Boundary bonds use the vacuum
    index from `spin_space_su2` (j=0 singlet). A fixed seed per site makes
    tests reproducible.
    """
    Spc, Op = spin_space_su2
    bulk = Index(direction=Direction.IN, group=Spc.group,
                 sectors=tuple(Sector(charge=q, dim=_SU2_BULK_DIM_PER_SECTOR)
                               for q in _SU2_BULK_SECTORS))
    vac = Op["vac"]
    tensors = []
    for i in range(_L):
        l = vac if i == 0 else bulk
        r = (vac if i == _L - 1 else bulk).flip()
        T = Tensor.random([l, r, Spc], seed=200 + i,
                          itags=[f'A{i:02d}', f'A{i + 1:02d}', f's{i:02d}'])
        tensors.append(T)
    return tensors


@pytest.fixture
def mpo_tensors_su2(spin_space_su2):
    """Fresh L=10 random SU(2) MPO tensors for each test.

    Site *i* has axes `(left_bond, right_bond, phys_in, phys_out)` with both
    physical axes carrying itag `s{i:02d}` and opposite directions (IN/OUT).
    Bond itags follow the MPO convention: the left bond of site *i* carries
    itag `W{i:02d}` and the right bond carries `W{i+1:02d}`. Same bulk bond
    structure as `mps_tensors_su2`.
    """
    Spc, Op = spin_space_su2
    bulk = Index(direction=Direction.IN, group=Spc.group,
                 sectors=tuple(Sector(charge=q, dim=_SU2_BULK_DIM_PER_SECTOR)
                               for q in _SU2_BULK_SECTORS))
    vac = Op["vac"]
    tensors = []
    for i in range(_L):
        l = vac if i == 0 else bulk
        r = (vac if i == _L - 1 else bulk).flip()
        W = Tensor.random([l, r, Spc, Spc.flip()], seed=300 + i,
                          itags=[f'W{i:02d}', f'W{i + 1:02d}', f's{i:02d}', f's{i:02d}'])
        tensors.append(W)
    return tensors


# ------------------------------------------------------------------
# Lattice geometry fixtures (symmetry-agnostic)
# ------------------------------------------------------------------

@pytest.fixture
def basic_1d_config():
    """Basic 1D chain configuration (50 sites)."""
    return {
        'lx': 50,
        'ly': 1,
        'bcx': 'OBC',
        'bcy': 'OBC',
        'label': 'SpinSqLatt',
        'cpl': 1.0,
        'cplp': [0.0, 0.0]
    }


@pytest.fixture
def basic_2d_config():
    """Basic 2D lattice configuration (6x4 = 24 sites)."""
    return {
        'lx': 6,
        'ly': 4,
        'bcx': 'OBC',
        'bcy': 'OBC',
        'label': 'SpinSqLatt',
        'cpl': 1.0,
        'cplp': [0.0, 0.0]
    }


@pytest.fixture
def pbc_cylinder_config():
    """Cylinder (PBC in Y) configuration (8x4 = 32 sites)."""
    return {
        'lx': 8,
        'ly': 4,
        'bcx': 'OBC',
        'bcy': 'PBC',
        'label': 'SpinSqLatt',
        'cpl': 1.0,
        'cplp': [0.0, 0.0]
    }


@pytest.fixture
def j2_config():
    """Configuration with J2 (NNN) interactions (6x6 = 36 sites)."""
    return {
        'lx': 6,
        'ly': 6,
        'bcx': 'OBC',
        'bcy': 'OBC',
        'label': 'SpinSqLatt',
        'cpl': 1.0,
        'cplp': [0.5, 0.3]
    }


@pytest.fixture
def torus_config():
    """Torus (PBC in both directions) configuration (8x6 = 48 sites)."""
    return {
        'lx': 8,
        'ly': 6,
        'bcx': 'PBC',
        'bcy': 'PBC',
        'label': 'SpinSqLatt',
        'cpl': 1.0,
        'cplp': [0.2, 0.1]
    }
