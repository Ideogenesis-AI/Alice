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


"""BUG algorithm package.

Implements the gate-based two-site BUG (Basis-Update & Galerkin) time
integrator: a nearest-neighbour Hamiltonian is evolved by even/odd Trotter
sweeps of two-site bond gates, each block split with a truncated SVD that adapts
the bond dimension. The public API includes:

- `Options` — run options (loadable from TOML).
- `Summary` — output dataclass.
- `run`     — top-level entry point.
"""

from .two_site_bug import Options, Summary
from .two_site_bug import run

__all__ = [
    'Options',
    'Summary',
    'run',
]
