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


"""Discarded-projector BUG algorithm package.

A rank-adaptive two-site Basis-Update & Galerkin (BUG) time integrator derived
from the faithful Ceruti–Kusch–Lubich scheme (arXiv:2304.05660), differing only
in the local bond update: the discarded (orthogonal-complement) projector is
applied to the K/L generator *before* the exponential, and the basis is grown by
a plain direct sum ``[U0 | Qk]`` / ``[V0 ; Ql]`` with no augmented overlap
matrices. A nearest-neighbour Hamiltonian is evolved by odd/even Trotter sweeps
of these local updates; the bond grows only as far as the entanglement requires.

This package reuses the faithful kernel, the odd/even sweep machinery, and the
AutoMPO bond terms of :mod:`alice.algorithm.two_site_bug`; only
:mod:`alice.algorithm.discarded_bug.candidate` is new. Public API:

- `Options` — run options (shared with two-site BUG; loadable from TOML).
- `Summary` — output dataclass (shared with two-site BUG).
- `run`     — top-level entry point.
"""

from .discarded_bug import Options, Summary, run

__all__ = [
    'Options',
    'Summary',
    'run',
]
