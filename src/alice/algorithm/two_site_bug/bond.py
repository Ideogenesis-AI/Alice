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


"""Nearest-neighbour bond Hamiltonians for the two-site BUG integrator.

The faithful Basis-Update & Galerkin (BUG) integrator (Ceruti, Kusch & Lubich,
*BIT* 2022; arXiv:2304.05660) evolves an `MPS` under a nearest-neighbour
Hamiltonian split into commuting odd/even bond groups. Each bond carries the
*bare* two-site Hamiltonian term `h_{i,i+1}` — not a pre-exponentiated gate. The
KLS local update (see :mod:`alice.algorithm.two_site_bug.kls`) exponentiates the
*projected* effective Hamiltonian internally; this module only supplies the bond
terms.

The bond Hamiltonian for bond *(i, i+1)* is reused directly from the AutoMPO
interaction list (`build_interaction`): the leading and terminal MPO tensors of
an `Interaction2Site` term are contracted over their shared operator channel,
exactly as `build_hamiltonian` would, so no new operator algebra is introduced.

Index convention (shared with the rest of Alice): a bond Hamiltonian `h` is a
4-index tensor with axes `(bra_i, ket_i, bra_{i+1}, ket_{i+1})`, the physical
`bra`/`ket` directions matching the MPS physical index and its dual.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from nicole import Tensor, contract, permute

from alice.network.interaction import Interaction, Interaction1Site, Interaction2Site


def to_complex(tensor: Tensor) -> Tensor:
    """Return a copy of `tensor` with every block cast to `complex128`.

    The KLS update exponentiates Hamiltonian terms, so the state must share the
    `complex128` dtype of the PyTorch backend.

    Parameters
    ----------
    tensor:
        Nicole tensor with real or complex blocks.

    Returns
    -------
    Tensor
        Tensor with identical indices and itags but `complex128` block data.
    """
    new_intw = None
    if tensor.intw is not None:
        new_intw = {
            key: bridge.to(tensor.device, dtype=torch.complex128)
            for key, bridge in tensor.intw.items()
        }
    return Tensor(
        indices=tensor.indices,
        itags=tensor.itags,
        data={key: block.to(torch.complex128) for key, block in tensor.data.items()},
        intw=new_intw,
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
    `H = Σ_b h_b` used by the odd/even Trotter split.

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
        non-zero coupling is present (the two-site BUG integrator targets
        nearest-neighbour Hamiltonians).
    """
    generators: List[Optional[Tensor]] = [None] * (length - 1)
    for intr in interactions:
        if isinstance(intr, Interaction1Site):
            if intr.cpl != 0.0:
                raise NotImplementedError(
                    "two-site BUG currently supports nearest-neighbour two-site "
                    f"Hamiltonians only; found a one-site term on site {intr.site}"
                )
            continue
        if isinstance(intr, Interaction2Site):
            if intr.cpl == 0.0:
                continue
            if intr.terminal_site != intr.leading_site + 1:
                raise NotImplementedError(
                    "two-site BUG supports nearest-neighbour terms only; found a "
                    f"term coupling sites {intr.leading_site} and {intr.terminal_site}"
                )
            bond = intr.leading_site
            term = bond_hamiltonian(intr)
            generators[bond] = term if generators[bond] is None else generators[bond] + term
    return generators


def kernel_gate(h: Tensor, site_l_itag: str, site_r_itag: str) -> Tensor:
    """Relabel a bond Hamiltonian into the local-KLS kernel's gate convention.

    The faithful-KLS kernel applies a bare two-site term `g` to a two-site block
    `theta` with `einsum('LRlr,aLRb->alrb', g, theta)`, then strips the trailing
    ``*`` from the output physical itags. It therefore expects `g` with axes
    `(ket_i, ket_j, bra_i, bra_j)`: the *ket* legs (`L`, `R`) carry the two site
    itags and contract `theta`'s physical legs, while the *bra* legs (`l`, `r`)
    carry the starred itags `('{si}*', '{sj}*')` and become the updated legs.

    `bond_hamiltonian` returns the term with axes
    `(bra_i, ket_i, bra_j, ket_j)`; this permutes to `(ket_i, ket_j, bra_i,
    bra_j)` and retags the four legs with the two sites' physical itags so the
    gate contracts the actual MPS physical indices.

    Parameters
    ----------
    h:
        Bond Hamiltonian with axes `(bra_i, ket_i, bra_j, ket_j)` (from
        :func:`bond_hamiltonian`).
    site_l_itag:
        Physical itag of the left site `i` in the MPS.
    site_r_itag:
        Physical itag of the right site `i+1` in the MPS.

    Returns
    -------
    Tensor
        Complex gate with axes `(ket_i, ket_j, bra_i, bra_j)` and itags
        `(site_l, site_r, '{site_l}*', '{site_r}*')`.
    """
    gate = permute(to_complex(h), [1, 3, 0, 2])
    gate.retag(
        [0, 1, 2, 3],
        [site_l_itag, site_r_itag, f'{site_l_itag}*', f'{site_r_itag}*'],
    )
    return gate
