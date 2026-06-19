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


"""Tests for two-site bond Hamiltonian extraction and gate construction."""

from __future__ import annotations

from alice.algorithm.two_site_bug.gate import (
    apply_bond_gate,
    build_bond_generators,
    exp_bond_gate,
    retag_gate_for_bond,
    to_complex,
)
from alice.algorithm.two_site_bug.scheme import _build_theta

from .conftest import heisenberg_chain


def _first_bond_block(spin_space, length=4):
    """Return `(generator, theta, phys_itags)` for the first active bond."""
    from alice import init_mps

    spc_index, operators = spin_space
    charges = [sector.charge for sector in spc_index.sectors]
    interactions, spc, geo = heisenberg_chain(length)
    config = [0, 1] * (length // 2)
    mps = init_mps(length, spc, operators, config=config,
                   target_qn=sum(charges[c] for c in config))
    generators = build_bond_generators(interactions, geo.L)
    bond = next(i for i, g in enumerate(generators) if g is not None)
    theta = _build_theta(to_complex(mps[bond]), to_complex(mps[bond + 1]))
    phys_itags = (mps[bond].itags[2], mps[bond + 1].itags[2])
    return generators[bond], theta, phys_itags


def test_zero_coefficient_gate_is_identity(spin_space):
    """`exp_bond_gate(h, 0)` must leave a two-site block unchanged."""
    generator, theta, phys_itags = _first_bond_block(spin_space)
    gate = retag_gate_for_bond(exp_bond_gate(generator, 0.0), phys_itags)
    updated = apply_bond_gate(theta, gate)
    assert (updated - theta).norm() < 1e-12


def test_gate_is_unitary(spin_space):
    """A real-time gate must preserve the norm of a two-site block."""
    generator, theta, phys_itags = _first_bond_block(spin_space)
    gate = retag_gate_for_bond(exp_bond_gate(generator, -1j * 0.37), phys_itags)
    updated = apply_bond_gate(theta, gate)
    assert abs(updated.norm() - theta.norm()) < 1e-12
