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


"""Ground-state MPS and MPO fixtures for standard lattice models."""

from __future__ import annotations

from typing import Tuple

import pytest

from alice.network import MPS, MPO

from .bosonic   import iter_diag_spin
from .fermionic import iter_diag_ferm
from .conductor import iter_diag_band
from .system    import build_heisenberg, build_freefermion, build_conductor

# Chain length used for all iterative diagonalization fixtures. N=20 gives
# a good ground-state approximation while keeping the test runtime under
# a few seconds per fixture.
_N     = 20
_NKEEP = 300


@pytest.fixture(scope='session')
def spin_chain() -> Tuple[MPS, MPO, float]:
    """Spin-1/2 Heisenberg ground state and Hamiltonian MPO (U1 symmetry).

    Returns
    -------
    tuple
        `(mps, mpo, E_gs)` where `mps` is the ground-state `MPS`, `mpo` is
        the Heisenberg `MPO`, and `E_gs` is the reference ground-state energy
        from iterative diagonalization.
    """
    Eg, _, mps_tensors = iter_diag_spin(N=_N, Nkeep=_NKEEP, verbose=False)
    mpo_tensors = build_heisenberg(N=_N)
    # The iter_diag MPS is in left-canonical form; the last site holds the
    # single ground-state vector, so center=N-1.
    return MPS(mps_tensors, center=_N - 1), MPO(mpo_tensors), float(Eg[-1])


@pytest.fixture(scope='session')
def ferm_chain() -> Tuple[MPS, MPO, float]:
    """Free-fermion tight-binding ground state and Hamiltonian MPO (U1 symmetry).

    Returns
    -------
    tuple
        `(mps, mpo, E_gs)` where `mps` is the ground-state `MPS`, `mpo` is
        the free-fermion `MPO`, and `E_gs` is the reference ground-state energy
        from iterative diagonalization.
    """
    Eg, _, mps_tensors = iter_diag_ferm(N=_N, Nkeep=_NKEEP, verbose=False)
    mpo_tensors = build_freefermion(N=_N)
    return MPS(mps_tensors, center=_N - 1), MPO(mpo_tensors), float(Eg[-1])


@pytest.fixture(scope='session')
def spin_chain_su2() -> Tuple[MPS, MPO, float]:
    """Spin-1/2 Heisenberg ground state and Hamiltonian MPO (SU2 symmetry).

    Returns
    -------
    tuple
        `(mps, mpo, E_gs)` where `mps` is the ground-state `MPS`, `mpo` is
        the Heisenberg `MPO`, and `E_gs` is the reference ground-state energy
        from iterative diagonalization.
    """
    Eg, _, mps_tensors = iter_diag_spin(N=_N, Nkeep=_NKEEP, symmetry='SU2', verbose=False)
    mpo_tensors = build_heisenberg(N=_N, symmetry='SU2')
    return MPS(mps_tensors, center=_N - 1), MPO(mpo_tensors), float(Eg[-1])


@pytest.fixture(scope='session')
def band_chain() -> Tuple[MPS, MPO, float]:
    """Spinful tight-binding (band conductor) ground state and MPO (U1,U1 symmetry).

    Returns
    -------
    tuple
        `(mps, mpo, E_gs)` where `mps` is the ground-state `MPS`, `mpo` is
        the conductor `MPO`, and `E_gs` is the reference ground-state energy
        from iterative diagonalization.
    """
    Eg, _, mps_tensors = iter_diag_band(N=_N, Nkeep=_NKEEP, symmetry='U1,U1', verbose=False)
    mpo_tensors = build_conductor(N=_N, symmetry='U1,U1')
    return MPS(mps_tensors, center=_N - 1), MPO(mpo_tensors), float(Eg[-1])


@pytest.fixture(scope='session')
def band_chain_su2() -> Tuple[MPS, MPO, float]:
    """Spinful tight-binding (band conductor) ground state and MPO (U1,SU2 symmetry).

    Returns
    -------
    tuple
        `(mps, mpo, E_gs)` where `mps` is the ground-state `MPS`, `mpo` is
        the conductor `MPO`, and `E_gs` is the reference ground-state energy
        from iterative diagonalization.
    """
    Eg, _, mps_tensors = iter_diag_band(N=_N, Nkeep=_NKEEP, symmetry='U1,SU2', verbose=False)
    mpo_tensors = build_conductor(N=_N, symmetry='U1,SU2')
    return MPS(mps_tensors, center=_N - 1), MPO(mpo_tensors), float(Eg[-1])
