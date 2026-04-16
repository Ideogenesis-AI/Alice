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


"""Interaction dataclasses for AutoMPO Hamiltonian construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from nicole import Tensor


@dataclass
class Interaction:
    """Base class for a single Hamiltonian interaction term.

    Attributes
    ----------
    cpl:
        Coupling constant. Stored as metadata; the coupling must already be
        baked into the tensor fields before calling `build_hamiltonian`.
    label:
        List of string labels describing the interaction (e.g. `['NN']`).
    """

    cpl:   float
    label: List[str]


@dataclass
class Interaction1Site(Interaction):
    """On-site (1-site) interaction term.

    Attributes
    ----------
    site:
        Site index (0-based).
    tnsr:
        4-index MPO tensor in format `(L_trivial_IN, R_trivial_OUT, bra_OUT,
        ket_IN)`. The coupling `cpl` must be baked in. Set before calling
        `build_hamiltonian`.
    """

    site: int
    tnsr: Optional[Tensor] = None


@dataclass
class Interaction2Site(Interaction):
    """Two-site interaction term.

    Attributes
    ----------
    leading_site:
        Index of the leading (left) site (0-based).
    terminal_site:
        Index of the terminal (right) site (0-based). Must satisfy
        `terminal_site > leading_site`.
    leading_tnsr:
        4-index MPO tensor for the leading site, in format
        `(L_trivial_IN, op_OUT, bra_OUT, ket_IN)`. The op axis carries the
        operator channel contracted with the terminal site. Set before calling
        `build_hamiltonian`.
    terminal_tnsr:
        4-index MPO tensor for the terminal site, in format
        `(op_IN, R_trivial_OUT, bra_OUT, ket_IN)`. The coupling `cpl` must
        be baked in. Set before calling `build_hamiltonian`.
    intermid_tnsr:
        4-index MPO tensor for intermediate sites (between `leading_site` and
        `terminal_site`), in format `(left_op_IN, right_op_OUT, bra_OUT,
        ket_IN)`. Required when `terminal_site > leading_site + 1`. For
        bosonic systems this is typically the physical identity dressed with
        op-sector bonds; for fermionic systems it is the Jordan-Wigner string.
        Set before calling `build_hamiltonian`.
    """

    leading_site:  int
    terminal_site: int
    leading_tnsr:  Optional[Tensor] = None
    terminal_tnsr: Optional[Tensor] = None
    intermid_tnsr: Optional[Tensor] = None
