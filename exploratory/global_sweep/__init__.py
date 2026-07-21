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

A rank-adaptive Basis-Update & Galerkin (BUG) time integrator — the MPS
specialisation of the rank-adaptive tree-tensor-network BUG of Ceruti–Lubich–Walach
/ Sulz (Alg. 5–7). Each step is a single **global sweep**: the basis growth is driven
by the **discarded** (orthogonal-complement) projector, applied explicitly
(``P_perp = I - U0 U0+``) and per basis matrix, with **no** augmented overlap matrices
``M``/``N`` and **no** backward correction.

A step forms the full Hamiltonian image ``phi = H psi`` (as an MPS), then sweeps the
chain building augmented left/right isometries that keep ``psi`` **exact** and admit
only the discarded part ``(I - U0 U0+) phi`` (SVD-truncated to the bond budget), so the
augmented bases span ``range(psi) + range(H psi)`` — the exact rank-adaptive BUG basis.
A single Galerkin centre connecting tensor is then integrated over the full step under
the two-site effective Hamiltonian. The bond dimension grows along the chain (the light
cone) as the wall melts; at full bond dimension the step is **exact** and it is second
order in ``dt`` (convergent — no forward-only floor). There is no Trotter splitting and
no backward (negative-time) substep — BUG is inverse-free by design.

This is the Alice port of the reference Julia ``discarded_bug_step!``. It reuses
Alice's DMRG environment machinery (:mod:`alice.algorithm.dmrg`) and is otherwise
self-contained — it carries its own symmetry-preserving Krylov exponentials and
local update, with no dependence on other integrators. Public API:

- `Options` — run options (loadable from TOML).
- `Summary` — output dataclass.
- `run`     — top-level entry point.
"""

from .discarded_bug import Options, Summary, run

__all__ = [
    'Options',
    'Summary',
    'run',
]
