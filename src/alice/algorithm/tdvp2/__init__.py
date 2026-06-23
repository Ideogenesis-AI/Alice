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


"""Two-site TDVP algorithm package.

A 2-site Time-Dependent Variational Principle integrator for an `MPS` evolving
under a Hamiltonian `MPO` (Haegeman et al., arXiv:1408.5056). A forward half-sweep
evolves each two-site block forward and the carried one-site tensor backward
(inverse-free backward correction); a reverse half-sweep mirrors it; a symmetric
step composes the two halves for second-order accuracy. The bond dimension adapts
through the per-bond SVD truncation.

Reuses the DMRG environment machinery and effective-Hamiltonian contractions
(`alice.algorithm.dmrg`) and the Hermitian Krylov exponential vendored with the
two-site BUG kernel. Public API:

- `Options` — run options (loadable from TOML).
- `Summary` — output dataclass.
- `run`     — top-level entry point ``run(mps, mpo, opts)``.
"""

from .tdvp2 import Options, Summary, run

__all__ = [
    'Options',
    'Summary',
    'run',
]
