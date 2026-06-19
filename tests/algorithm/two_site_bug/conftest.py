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


"""Pytest fixtures and exact-diagonalization helpers for two-site BUG tests.

The helpers build a dense Heisenberg Hamiltonian, dense product states, and a
dense vector from an MPS — all in the same physical basis ordering as Nicole's
spin-1/2 U(1) space — so the integrator can be checked against exact
diagonalization.
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
    """Run every test in a fresh working directory."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope='session')
def spin_space() -> Tuple[Index, Dict[str, Tensor]]:
    """Spin-1/2 U(1) physical space and operators (shared across the session)."""
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
    """Return dense `(Sz, Sp, Sm)` in the sector order given by `charges`."""
    sz = torch.diag(torch.tensor([c / 2.0 for c in charges], dtype=torch.complex128))
    up = 0 if charges[0] > charges[1] else 1
    sp = torch.zeros((2, 2), dtype=torch.complex128)
    sp[up, 1 - up] = 1.0
    return sz, sp, sp.conj().T.contiguous()


def _embed(op: torch.Tensor, site: int, length: int) -> torch.Tensor:
    """Embed a single-site operator into the full `2**length` Hilbert space."""
    eye = torch.eye(2, dtype=torch.complex128)
    factors = [op if k == site else eye for k in range(length)]
    return functools.reduce(lambda a, b: torch.kron(a.contiguous(), b.contiguous()), factors)


def dense_heisenberg(length: int, charges: List[int], coupling: float = 1.0) -> torch.Tensor:
    """Build the dense Heisenberg Hamiltonian matching Alice's spin basis.

    Parameters
    ----------
    length:
        Number of sites.
    charges:
        Sector charges of the physical index, in dense order (from
        `Spc.sectors`), used to fix the single-site basis ordering.
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
    """Build the dense total-`S_z` operator matching Alice's spin basis."""
    sz, _, _ = _spin_matrices(charges)
    return sum(_embed(sz, i, length) for i in range(length))


def dense_hamiltonian(interactions, length: int, charges: List[int]) -> torch.Tensor:
    """Assemble the full `d**L` dense Hamiltonian from Alice's own bond terms.

    Densifies each nearest-neighbour `Interaction2Site` bond Hamiltonian exactly as
    the integrator consumes it (`build_bond_generators`) and lifts it to the full
    Hilbert space. This is convention-exact — the dense operator is, by
    construction, the same Hamiltonian the MPS evolves under — so it avoids any
    basis/normalisation mismatch a hand-written model matrix could introduce.

    Parameters
    ----------
    interactions:
        Interaction list from `build_interaction`.
    length:
        Number of sites `L`.
    charges:
        Charges of the physical space in dense order (fixes the local basis).

    Returns
    -------
    torch.Tensor
        Dense `(d**L, d**L)` Hamiltonian, `d = len(charges)`.
    """
    from alice.algorithm.two_site_bug._kernel import to_dense
    from alice.algorithm.two_site_bug.bond import build_bond_generators

    generators = build_bond_generators(interactions, length)
    d = len(charges)
    dim = d ** length
    ham = torch.zeros((dim, dim), dtype=torch.complex128)
    eye = torch.eye(d, dtype=torch.complex128)
    for bond, h in enumerate(generators):
        if h is None:
            continue
        # h axes: (bra_i, ket_i, bra_j, ket_j). Densify, then reorder to the
        # operator matrix [(bra_i, bra_j), (ket_i, ket_j)].
        dense = to_dense(h, [h.itags[0], h.itags[1], h.itags[2], h.itags[3]]).to(torch.complex128)
        local = dense.permute(0, 2, 1, 3).reshape(d * d, d * d)
        factors: List[torch.Tensor] = []
        site = 0
        while site < length:
            if site == bond:
                factors.append(local)
                site += 2
            else:
                factors.append(eye)
                site += 1
        lifted = factors[0]
        for factor in factors[1:]:
            lifted = torch.kron(lifted.contiguous(), factor.contiguous())
        ham = ham + lifted
    return ham


def product_vector(config: List[int], charges: List[int]) -> torch.Tensor:
    """Build the dense product-state vector for a sector-index configuration.

    Parameters
    ----------
    config:
        Per-site sector index (0 or 1) — the same `config` passed to `init_mps`.
    charges:
        Sector charges in dense order (unused beyond fixing length-2 basis).

    Returns
    -------
    torch.Tensor
        Dense state vector of length `2**len(config)`.
    """
    basis = [
        torch.tensor([1.0, 0.0], dtype=torch.complex128),
        torch.tensor([0.0, 1.0], dtype=torch.complex128),
    ]
    return functools.reduce(
        lambda a, b: torch.kron(a.contiguous(), b.contiguous()),
        [basis[c] for c in config],
    )


def exact_evolve(ham: torch.Tensor, psi0: torch.Tensor, t: float) -> torch.Tensor:
    """Return `exp(-i t H) |psi0>` via dense eigendecomposition."""
    evals, evecs = torch.linalg.eigh(ham)
    return evecs @ (torch.exp(-1j * t * evals) * (evecs.conj().T @ psi0))


def _core_dense(core: Tensor, charges: List[int]) -> torch.Tensor:
    """Densify a 3-index MPS core `(left, right, phys)` to a dense torch tensor.

    The bonds are densified to their own (symmetry-restricted) dimensions, but the
    physical axis is *embedded into the full local basis* of size `len(charges)`:
    a symmetric core only stores the physical sectors its charge structure allows
    (e.g. a boundary site pinned to one charge has a dim-1 physical leg), so each
    present physical charge `q` is placed at its global basis index
    `charges.index(q)` and the rest is zero. This makes the contracted state live
    in the full `len(charges)**L` space the dense ED helpers use.
    """
    phys_table = {q: (charges.index(q), 1) for q in charges}
    offsets = []
    for axis, index in enumerate(core.indices):
        if axis == 2:  # physical leg → full local basis
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
        Charges of the full physical space, in dense order (from `Spc.sectors`).
        Each site's physical leg is embedded into this `len(charges)`-dimensional
        basis (see `_core_dense`), so the result has length `len(charges)**L`
        regardless of which charges each site's symmetric core actually carries.

    Returns
    -------
    torch.Tensor
        Dense state vector of length `len(charges)**L`.
    """
    psi = _core_dense(mps[0], charges)[0]  # drop trivial left bond -> (right, phys_0)
    for site in range(1, mps.L):
        psi = torch.tensordot(psi, _core_dense(mps[site], charges), dims=([0], [0]))
        psi = psi.movedim(-2, 0)  # keep the open right bond at the front
    return psi[0].reshape(-1)  # drop trivial right bond
