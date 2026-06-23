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


"""Pytest fixtures and exact-diagonalization helpers for 2-site TDVP tests.

These helpers build a dense Heisenberg Hamiltonian, a dense total-Sz operator,
and a dense state vector from an MPS — all in the same physical basis ordering as
Nicole's spin-1/2 U(1) space — so the integrator can be checked against exact
diagonalization (the analytical reference for the small chains tested here).
"""

from __future__ import annotations

import functools
from typing import Dict, List, Tuple

import pytest
import torch
from nicole import Index, Tensor, load_space

from alice.network import MPS, build_interaction


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Run every test in a fresh working directory.

    Parameters
    ----------
    tmp_path:
        Pytest per-test temporary directory.
    monkeypatch:
        Pytest fixture used to change the working directory for the test.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope='session')
def spin_space() -> Tuple[Index, Dict[str, Tensor]]:
    """Spin-1/2 U(1) physical space and operators (shared across the session).

    Returns
    -------
    tuple
        The `(spc, operators)` pair from `load_space('Spin', 'U1', {'J': 0.5})`.
    """
    return load_space('Spin', 'U1', {'J': 0.5})


def heisenberg_chain(length: int, coupling: float = 1.0):
    """Build the Heisenberg interaction list, physical index, and geometry.

    Parameters
    ----------
    length:
        Number of sites.
    coupling:
        Isotropic exchange coupling `J`.

    Returns
    -------
    tuple
        `(interactions, spc, geo)` from `build_interaction`.
    """
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': length, 'bcx': 'OBC', 'n2x': True},
        'model': {
            'category': 'bosonic', 'label': 'Heisenberg',
            'symmetry': 'U1', 'spin': 0.5, 'J': coupling,
        },
    }
    return build_interaction(cfg)


def _spin_matrices(charges: List[int]):
    """Return dense `(Sz, Sp, Sm)` spin-1/2 operators in the given sector order.

    Parameters
    ----------
    charges:
        Sector charges of the physical index in dense order, fixing the
        single-site basis ordering.

    Returns
    -------
    tuple
        Dense `(Sz, Sp, Sm)` matrices.
    """
    sz = torch.diag(torch.tensor([c / 2.0 for c in charges], dtype=torch.complex128))
    up = 0 if charges[0] > charges[1] else 1
    sp = torch.zeros((2, 2), dtype=torch.complex128)
    sp[up, 1 - up] = 1.0
    return sz, sp, sp.conj().T.contiguous()


def _embed(op: torch.Tensor, site: int, length: int) -> torch.Tensor:
    """Embed a single-site operator into the full `2**length` Hilbert space.

    Parameters
    ----------
    op:
        Single-site `2 x 2` operator.
    site:
        Site index the operator acts on.
    length:
        Number of sites.

    Returns
    -------
    torch.Tensor
        The operator embedded as a `2**length x 2**length` dense matrix.
    """
    eye = torch.eye(2, dtype=torch.complex128)
    factors = [op if k == site else eye for k in range(length)]
    return functools.reduce(lambda a, b: torch.kron(a.contiguous(), b.contiguous()), factors)


def dense_heisenberg(length: int, charges: List[int], coupling: float = 1.0) -> torch.Tensor:
    """Build the dense Heisenberg Hamiltonian matching Alice's spin basis.

    This is the same nearest-neighbour `J (Sz Sz + ½(S+ S- + S- S+))` chain that
    `build_hamiltonian` assembles as an MPO from the Heisenberg interaction list,
    written directly in the dense `2**length` basis so it can serve as the
    exact-diagonalization reference.

    Parameters
    ----------
    length:
        Number of sites.
    charges:
        Sector charges of the physical index, in dense order, fixing the basis.
    coupling:
        Isotropic exchange coupling `J`.

    Returns
    -------
    torch.Tensor
        Dense `(2**length, 2**length)` Hamiltonian.
    """
    sz, sp, sm = _spin_matrices(charges)
    dim = 2 ** length
    ham = torch.zeros((dim, dim), dtype=torch.complex128)
    for i in range(length - 1):
        ham = ham + coupling * (
            _embed(sz, i, length) @ _embed(sz, i + 1, length)
            + 0.5 * (_embed(sp, i, length) @ _embed(sm, i + 1, length))
            + 0.5 * (_embed(sm, i, length) @ _embed(sp, i + 1, length))
        )
    return ham


def dense_total_sz(length: int, charges: List[int]) -> torch.Tensor:
    """Build the dense total-`S_z` operator matching Alice's spin basis.

    Parameters
    ----------
    length:
        Number of sites.
    charges:
        Sector charges of the physical index, in dense order, fixing the basis.

    Returns
    -------
    torch.Tensor
        Dense `(2**length, 2**length)` total-`S_z` operator.
    """
    sz, _, _ = _spin_matrices(charges)
    return sum(_embed(sz, i, length) for i in range(length))


def exact_evolve(ham: torch.Tensor, psi0: torch.Tensor, t: float) -> torch.Tensor:
    """Return `exp(-i t H) |psi0>` via dense eigendecomposition.

    Parameters
    ----------
    ham:
        Dense Hermitian Hamiltonian.
    psi0:
        Dense initial state vector.
    t:
        Evolution time.

    Returns
    -------
    torch.Tensor
        The exactly evolved dense state vector.
    """
    evals, evecs = torch.linalg.eigh(ham)
    return evecs @ (torch.exp(-1j * t * evals) * (evecs.conj().T @ psi0))


def _core_dense(core: Tensor, charges: List[int]) -> torch.Tensor:
    """Densify a 3-index MPS core `(left, right, phys)` to a dense torch tensor.

    The physical axis is embedded into the full local basis of size
    `len(charges)`: a symmetric core only stores the physical sectors its charge
    structure allows, so each present physical charge `q` is placed at its global
    basis index `charges.index(q)` and the rest is zero.

    Parameters
    ----------
    core:
        MPS site tensor with axes `(left_bond, right_bond, physical)`.
    charges:
        Charges of the full physical space, in dense order.

    Returns
    -------
    torch.Tensor
        Dense `(left, right, len(charges))` tensor.
    """
    phys_table = {q: (charges.index(q), 1) for q in charges}
    offsets = []
    for axis, index in enumerate(core.indices):
        if axis == 2:
            offsets.append((phys_table, len(charges)))
            continue
        table = {}
        cursor = 0
        for sector in index.sectors:
            table[sector.charge] = (cursor, sector.dim)
            cursor += sector.dim
        offsets.append((table, cursor))
    dense = torch.zeros([total for _, total in offsets], dtype=torch.complex128)
    for key, block in core.data.items():
        slices = tuple(
            slice(offsets[axis][0][key[axis]][0],
                  offsets[axis][0][key[axis]][0] + offsets[axis][0][key[axis]][1])
            for axis in range(3)
        )
        dense[slices] = block.to(torch.complex128)
    return dense


def mps_to_vector(mps: MPS, charges: List[int]) -> torch.Tensor:
    """Contract an OBC MPS into a dense state vector in the full physical basis.

    Parameters
    ----------
    mps:
        MPS with trivial (dimension-1) boundary bonds.
    charges:
        Charges of the full physical space, in dense order. Each site's physical
        index is embedded into this `len(charges)`-dimensional basis.

    Returns
    -------
    torch.Tensor
        Dense state vector of length `len(charges)**L`.
    """
    psi = _core_dense(mps[0], charges)[0]
    for site in range(1, mps.L):
        psi = torch.tensordot(psi, _core_dense(mps[site], charges), dims=([0], [0]))
        psi = psi.movedim(-2, 0)
    return psi[0].reshape(-1)
