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


"""Infinite states and Hamiltonians shared by the iDMRG tests."""

from __future__ import annotations

import math

import pytest
from nicole import Direction, Tensor, load_space
from nicole.index import Index, Sector

from alice import build_interaction
from alice.network import iMPS
from alice.network.autoimpo import build_ihamiltonian

# Model configurations, keyed by the name used in the tests. The third entry
# says whether the model needs next-nearest-neighbor bonds emitted.
HEISENBERG = dict(
    category='bosonic', label='Heisenberg', symmetry='U1', spin=0.5, J=1.0,
)
MAJUMDAR_GHOSH = dict(HEISENBERG, Jp=0.5)
HEISENBERG_SU2 = dict(
    category='bosonic', label='Heisenberg', symmetry='SU2', spin=0.5, J=1.0,
)
MAJUMDAR_GHOSH_SU2 = dict(HEISENBERG_SU2, Jp=0.5)


def ihamiltonian(model, N, *, n3x=False):
    """Build a Hamiltonian iMPO for `model` over a unit cell of `N` sites."""
    interactions, spc, geo = build_interaction({
        'geometry': {
            'lattice': 'chain', 'lx': N, 'bcx': 'IBC', 'n2x': True, 'n3x': n3x,
        },
        'model': model,
    })
    return build_ihamiltonian(interactions, geo.L, spc)


def _bond(group, charges, dim=1):
    """Build a bond index carrying `dim` states in each of `charges`."""
    return Index(
        direction=Direction.IN,
        group=group,
        sectors=tuple(Sector(charge=q, dim=dim) for q in charges),
    )


@pytest.fixture(scope='session')
def spin_half():
    """Spin-1/2 physical space under U(1)."""
    Spc, _ = load_space('Spin', 'U1', {'J': 0.5})
    return Spc


@pytest.fixture(scope='session')
def spin_half_su2():
    """Spin-1/2 physical space under SU(2)."""
    Spc, _ = load_space('Spin', 'SU2', {'J': 0.5})
    return Spc


def neel_imps(Spc, N):
    """Build the Neel product state as a gauge-fixed iMPS, `N` even.

    Every bond is one-dimensional, so the energy of this state against the
    Heisenberg Hamiltonian is the classical Ising value and nothing else.
    """
    bonds = [_bond(Spc.group, (k % 2,)) for k in range(N + 1)]
    tensors = []
    for i in range(N):
        t = Tensor.zeros(
            [bonds[i], bonds[i + 1].flip(), Spc],
            itags=[f'A{i % N:02d}', f'A{(i + 1) % N:02d}', f's{i:02d}'],
        )
        t.data[(i % 2, (i + 1) % 2, 1 if i % 2 == 0 else -1)][:] = 1.0
        tensors.append(t)
    imps = iMPS(tensors)
    imps.canonical()
    return imps


def dimer_imps(Spc):
    """Build the Majumdar-Ghosh dimer state as a gauge-fixed 2-site iMPS.

    Sites 0 and 1 share a singlet and the pattern repeats. This is the exact
    ground state of the Majumdar-Ghosh point, so its energy density is known
    in closed form.
    """
    trivial = _bond(Spc.group, (0,))
    half = _bond(Spc.group, (-1, 1))

    opener = Tensor.zeros(
        [trivial, half.flip(), Spc], itags=['A00', 'A01', 's00'],
    )
    opener.data[(0, 1, 1)][:] = 1.0
    opener.data[(0, -1, -1)][:] = 1.0

    closer = Tensor.zeros(
        [half, trivial.flip(), Spc], itags=['A01', 'A00', 's01'],
    )
    closer.data[(1, 0, -1)][:] = 1.0 / math.sqrt(2.0)
    closer.data[(-1, 0, 1)][:] = -1.0 / math.sqrt(2.0)

    imps = iMPS([opener, closer])
    imps.canonical()
    return imps


def dimer_imps_su2(Spc):
    """Build the dimer state under SU(2), where it is the only state there is.

    A bond alternating between one singlet and one spin-1/2 multiplet, each
    of multiplicity one, leaves no freedom: Clebsch-Gordan fixes every entry,
    so a random tensor on those indices is the dimer state after gauge
    fixing. That makes this the sharpest available check, since it avoids the
    spin tables whose finite precision limits the U(1) version.
    """
    singlet = _bond(Spc.group, (0,))
    doublet = _bond(Spc.group, (1,))
    imps = iMPS([
        Tensor.random(
            [singlet, doublet.flip(), Spc], seed=1,
            itags=['A00', 'A01', 's00'],
        ),
        Tensor.random(
            [doublet, singlet.flip(), Spc], seed=2,
            itags=['A01', 'A00', 's01'],
        ),
    ])
    imps.canonical()
    return imps


def random_imps(Spc, N, *, period=None, width=2, dim=2, seed=1):
    """Build a random gauge-fixable iMPS of `N` sites.

    Bond `k` carries only charges of parity `k`, which is what a physical
    spin-1/2 state looks like and what keeps the transfer operator out of
    the bipartite trap described in `iMPS.canonical`. When `period` is given
    the cell repeats with that period, so the same physical state can be
    described by unit cells of different sizes.
    """
    period = N if period is None else period

    def bond(k):
        charges = tuple(
            q for q in range(-2 * width, 2 * width + 1) if (q - k) % 2 == 0
        )
        return _bond(Spc.group, charges, dim=dim)

    tensors = []
    for i in range(N):
        j = i % period
        tensors.append(Tensor.random(
            [bond(j), bond(j + 1).flip(), Spc],
            seed=seed * 10 + j,
            itags=[f'A{i % N:02d}', f'A{(i + 1) % N:02d}', f's{i:02d}'],
        ))
    imps = iMPS(tensors)
    imps.canonical()
    return imps
