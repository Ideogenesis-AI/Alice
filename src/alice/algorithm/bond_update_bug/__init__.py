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


"""bond_update_bug algorithm package.

Implements the bond_update_bug (Basis-Update & Galerkin) time integrator
of Ceruti, Kusch & Lubich (arXiv:2304.05660): a nearest-neighbour Hamiltonian is
evolved by odd/even Trotter sweeps of local K/L/S bond updates. Each update
augments the left/right frames from the evolved K/L factors, evolves the small
core in the augmented bases (Galerkin), and truncates with an SVD — exact at
full rank, rank-adaptive otherwise. The public API includes:

- `Options` — run options (loadable from TOML).
- `Summary` — output dataclass.
- `run`     — top-level entry point.

The KLS local kernel lives in the vendored, Nicole-native `_kernel`
subpackage; this package wires it to Alice's `MPS` and AutoMPO bond terms.
"""

from .bond_update_bug import Options, Summary
from .bond_update_bug import run

__all__ = [
    'Options',
    'Summary',
    'run',
]
