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


"""Nearest-neighbour bond Hamiltonians and two-site gates for the BUG integrator.

The gate-based BUG integrator evolves an `MPS` with the time-evolution operator
of a nearest-neighbour Hamiltonian, split into commuting odd/even bond groups
(a Trotter split). The bare two-site bond Hamiltonian for bond *(i, i+1)* is
reused directly from the AutoMPO interaction list (`build_interaction`): the
leading and terminal MPO tensors of an `Interaction2Site` term are contracted
over their shared operator channel, exactly as `build_hamiltonian` would, so no
new operator algebra is introduced.

Index conventions follow the rest of Alice:

- A bond Hamiltonian `h` is a 4-index tensor with axes
  `(bra_i, ket_i, bra_{i+1}, ket_{i+1})`; physical (`bra`/`ket`) directions match
  the MPS physical index and its dual.
- A two-site gate `G = exp(coeff * h)` is a 4-index tensor with axes
  `(ket_i, ket_{i+1}, bra_i, bra_{i+1})`: the `ket` axes contract a two-site MPS
  block, the `bra` axes become the updated physical indices.

The matrix exponential runs block-wise on the PyTorch backend (via Nicole), so
it inherits device, dtype, and autograd support and preserves the symmetry block
structure exactly.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from nicole import Tensor, contract, einsum, merge_axes

from alice.network.interaction import Interaction, Interaction1Site, Interaction2Site


def to_complex(tensor: Tensor) -> Tensor:
    """Return a copy of `tensor` with every block cast to `complex128`.

    Parameters
    ----------
    tensor:
        Nicole tensor with real or complex blocks.

    Returns
    -------
    Tensor
        Tensor with identical indices and itags but `complex128` block data.
    """
    return Tensor(
        indices=tensor.indices,
        itags=tensor.itags,
        data={key: block.to(torch.complex128) for key, block in tensor.data.items()},
        dtype=torch.complex128,
    )


def bond_hamiltonian(intr: Interaction2Site) -> Tensor:
    """Build the bare two-site bond Hamiltonian of one nearest-neighbour term.

    Contracts the leading and terminal MPO tensors of `intr` over their shared
    operator channel and drops the two trivial boundary bonds, returning the
    physical two-site operator scaled by the coupling `intr.cpl`. This mirrors
    the contraction `build_hamiltonian` performs, so the bond Hamiltonian is
    exactly the term that enters the AutoMPO Hamiltonian.

    Parameters
    ----------
    intr:
        Nearest-neighbour two-site interaction with populated `leading_tnsr`
        and `terminal_tnsr` and `terminal_site == leading_site + 1`.

    Returns
    -------
    Tensor
        4-index bond Hamiltonian with axes
        `(bra_i, ket_i, bra_{i+1}, ket_{i+1})`.

    Raises
    ------
    ValueError
        If `intr` is not nearest-neighbour, or its tensors are not populated.
    """
    if intr.terminal_site != intr.leading_site + 1:
        raise ValueError(
            "bond_hamiltonian requires a nearest-neighbour term "
            f"(terminal_site == leading_site + 1), got leading_site="
            f"{intr.leading_site}, terminal_site={intr.terminal_site}"
        )
    if intr.leading_tnsr is None or intr.terminal_tnsr is None:
        raise ValueError(
            "bond_hamiltonian requires populated leading_tnsr and terminal_tnsr; "
            "build the interaction list with build_interaction first"
        )

    # leading_tnsr: (L_trivial_IN, op_OUT, bra_i, ket_i)
    # terminal_tnsr: (op_IN, R_trivial_OUT, bra_{i+1}, ket_{i+1})
    # Contract the shared operator channel (leading axis 1, terminal axis 0).
    h = contract(intr.leading_tnsr, intr.terminal_tnsr, axes=(1, 0))
    # h axes: (L_trivial, bra_i, ket_i, R_trivial, bra_{i+1}, ket_{i+1}).
    h.squeeze(0)  # drop L_trivial -> (bra_i, ket_i, R_trivial, bra_{i+1}, ket_{i+1})
    h.squeeze(2)  # drop R_trivial -> (bra_i, ket_i, bra_{i+1}, ket_{i+1})
    return h * intr.cpl


def build_bond_generators(interactions: List[Interaction], length: int) -> List[Optional[Tensor]]:
    """Accumulate per-bond Hamiltonians from an AutoMPO interaction list.

    Sums every nearest-neighbour `Interaction2Site` term onto its bond. Bonds
    with no term are left as `None`. This yields the bond decomposition
    `H = Σ_b h_b` used by the Trotter split.

    Parameters
    ----------
    interactions:
        Interaction list from `build_interaction`. Every active term must be a
        nearest-neighbour `Interaction2Site`.
    length:
        Number of sites `L`; there are `L - 1` bonds.

    Returns
    -------
    list of (Tensor or None)
        Length `L - 1`. Entry *b* is the bond Hamiltonian for bond
        *(b, b+1)*, or `None` if no term acts on that bond.

    Raises
    ------
    NotImplementedError
        If a non-nearest-neighbour two-site term or a one-site term with a
        non-zero coupling is present (the gate-based BUG integrator targets
        nearest-neighbour Hamiltonians).
    """
    generators: List[Optional[Tensor]] = [None] * (length - 1)
    for intr in interactions:
        if isinstance(intr, Interaction1Site):
            if intr.cpl != 0.0:
                raise NotImplementedError(
                    "gate-based BUG currently supports nearest-neighbour two-site "
                    f"Hamiltonians only; found a one-site term on site {intr.site}"
                )
            continue
        if isinstance(intr, Interaction2Site):
            if intr.cpl == 0.0:
                continue
            if intr.terminal_site != intr.leading_site + 1:
                raise NotImplementedError(
                    "gate-based BUG supports nearest-neighbour terms only; found a "
                    f"term coupling sites {intr.leading_site} and {intr.terminal_site}"
                )
            bond = intr.leading_site
            term = bond_hamiltonian(intr)
            generators[bond] = term if generators[bond] is None else generators[bond] + term
    return generators


def exp_bond_gate(h: Tensor, coeff: complex) -> Tensor:
    """Exponentiate a bond Hamiltonian into a two-site gate `exp(coeff * h)`.

    Merges the two `bra` axes and the two `ket` axes of `h` into a single
    matrix per symmetry sector, applies `torch.linalg.matrix_exp` block-wise on
    the PyTorch backend, then unmerges back to a 4-index gate. Because the merge
    groups states by total charge, the block-wise exponential equals the full
    matrix exponential while preserving the symmetry structure exactly.

    Parameters
    ----------
    h:
        4-index bond Hamiltonian with axes `(bra_i, ket_i, bra_{i+1}, ket_{i+1})`.
    coeff:
        Scalar multiplying `h` before exponentiation. For real-time evolution by
        a step `dt` use `coeff = -1j * dt`.

    Returns
    -------
    Tensor
        4-index gate with axes `(ket_i, ket_{i+1}, bra_i, bra_{i+1})`.
    """
    # Merge bra_i, bra_{i+1} -> B and ket_i, ket_{i+1} -> K, leaving a (K, B)
    # operator matrix in each total-charge sector.
    merged_bra, split_bra = merge_axes(h, [0, 2], merged_tag='_bug_bra_')
    merged, split_ket = merge_axes(merged_bra, [1, 2], merged_tag='_bug_ket_')

    exp_data = {
        key: torch.linalg.matrix_exp(coeff * block.to(torch.complex128))
        for key, block in merged.data.items()
    }
    gate_matrix = Tensor(
        indices=merged.indices,
        itags=merged.itags,
        data=exp_data,
        dtype=torch.complex128,
    )

    # Unmerge: (K, B) -> (B, ket_i, ket_{i+1}) -> (ket_i, ket_{i+1}, bra_i, bra_{i+1}).
    gate = contract(gate_matrix, to_complex(split_ket), axes=(0, 2))
    gate = contract(gate, to_complex(split_bra), axes=(0, 2))
    return gate


def retag_gate_for_bond(gate: Tensor, phys_itags: Tuple[str, str]) -> Tensor:
    """Relabel a gate's physical axes with the itags of a specific bond.

    :func:`exp_bond_gate` returns a gate with generic physical itags. Before the
    gate can contract a two-site block, its `ket` and `bra` axes must carry the
    physical itags of the two sites it acts on (Nicole contracts by matching
    itag and opposite direction). `ket` and `bra` axes share an itag but have
    opposite directions, exactly as an MPO's two physical axes do.

    Parameters
    ----------
    gate:
        Gate with axes `(ket_i, ket_{i+1}, bra_i, bra_{i+1})`.
    phys_itags:
        Physical itags `('s{i:02d}', 's{i+1:02d}')` of the two sites.

    Returns
    -------
    Tensor
        A copy of `gate` whose four axes carry the bond's physical itags.
    """
    si, sj = phys_itags
    out = gate.clone()
    out.retag([0, 1, 2, 3], [si, sj, si, sj])
    return out


def apply_bond_gate(theta: Tensor, gate: Tensor) -> Tensor:
    """Apply a two-site gate to a two-site MPS block.

    Contracts the gate `ket` axes with the physical axes of `theta`; the gate
    `bra` axes become the updated physical axes. The gate must already carry the
    bond's physical itags (see :func:`retag_gate_for_bond`).

    Parameters
    ----------
    theta:
        Two-site block with axes `(left, right, phys_i, phys_{i+1})`.
    gate:
        Gate with axes `(ket_i, ket_{i+1}, bra_i, bra_{i+1})` already relabelled
        for this bond.

    Returns
    -------
    Tensor
        Updated two-site block with axes `(left, right, phys_i, phys_{i+1})`.
    """
    # theta (a=left, c=right, r=phys_i, s=phys_{i+1}); gate (r=ket_i, s=ket_{i+1},
    # k=bra_i, u=bra_{i+1}). Contract physical/ket axes -> (a, c, k, u).
    return einsum('acrs,rsku->acku', theta, gate)
