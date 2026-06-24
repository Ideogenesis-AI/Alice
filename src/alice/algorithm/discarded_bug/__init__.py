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

A rank-adaptive **two-site** Basis-Update & Galerkin (BUG) time integrator — the
MPS specialisation of the tree-tensor-network BUG of Ceruti–Lubich–Walach, with
two modifications: every local update is two-site (through the two-site effective
Hamiltonian with the left/right MPO environments), and the basis growth is driven
by the **discarded** (orthogonal-complement) projector — the augmented frames are
read directly off the evolved two-site block (``qr([Theta1_left | U0])`` /
``qr([Theta1_right | V0])``), with **no** augmented overlap matrices and **no**
backward correction.

Acting with the Hamiltonian on a two-site window is what creates the new Schmidt
direction (a domain-wall interface block has Schmidt rank 2), so the bond grows as
the entanglement front reaches it. Following the Lubich tree BUG (whose tree is built
by recursive bisection of the 1D modes), a step recursively bisects the chain and
applies one two-site node update at each bisection bond; because every bond is a tree
node, the bond dimension grows along the whole chain (the full light cone), matching
forward two-site TDVP's bond profile. There is no Trotter splitting and no backward
(negative-time) substep — BUG is inverse-free by design.

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
