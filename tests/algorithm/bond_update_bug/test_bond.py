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


"""Tests for nearest-neighbour bond Hamiltonian extraction and gate relabelling."""

from __future__ import annotations

from alice.algorithm.bond_update_bug.bond import bond_hamiltonian, build_bond_generators, kernel_gate

from .conftest import heisenberg_chain


def _first_generator(length=4):
    """Return the first active two-site bond Hamiltonian of a Heisenberg chain."""
    interactions, _, geo = heisenberg_chain(length)
    generators = build_bond_generators(interactions, geo.L)
    bond = next(i for i, g in enumerate(generators) if g is not None)
    return generators[bond]


def test_bond_hamiltonian_is_four_index():
    """A nearest-neighbour bond term has axes (bra_i, ket_i, bra_{i+1}, ket_{i+1})."""
    h = _first_generator()
    assert len(h.indices) == 4


def test_kernel_gate_itags_and_axes():
    """`kernel_gate` relabels the term into the kernel's (ket_i, ket_j, bra_i*, bra_j*) order."""
    h = _first_generator()
    gate = kernel_gate(h, 's00', 's01')
    # Axes are (ket_i, ket_j, bra_i, bra_j) with the bra (output) legs starred.
    assert list(gate.itags) == ['s00', 's01', 's00*', 's01*']
    assert len(gate.indices) == 4


def test_kernel_gate_is_complex():
    """The gate must be complex so the local exponentials share the backend dtype."""
    import torch

    h = _first_generator()
    gate = kernel_gate(h, 's00', 's01')
    assert all(block.dtype == torch.complex128 for block in gate.data.values())
