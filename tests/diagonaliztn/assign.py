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


"""Temporary model-specific tensor assignment helpers for AutoMPO tests.

Each function populates the `leading_tnsr` and `terminal_tnsr` fields of a
list of `Interaction2Site` objects in-place, following the same operator
conventions as the corresponding `build_*` functions in `system.py`.

These helpers will be superseded once a proper model builder is implemented.
"""

from __future__ import annotations

from typing import Dict, List

from nicole import Direction, Tensor
from nicole import oplus, capcup, contract
from nicole.index import Index


def assign_heisenberg(
    interactions: List,
    spc: Index,
    ops: Dict[str, Tensor],
    symmetry: str = 'U1',
    J: float = 1.0,
) -> None:
    """Populate `leading_tnsr` and `terminal_tnsr` for Heisenberg interactions.

    Sets up the spin–spin coupling `J * S†_i · S_j` for each interaction by
    constructing the same `S4` / `S4dag` 4-index operator templates used by
    `build_heisenberg` in `system.py`.

    Parameters
    ----------
    interactions:
        List of `Interaction2Site` objects to populate (all nearest-neighbor
        bonds of a Heisenberg chain).
    spc:
        Physical `Index` for spin sites, obtained from `load_space`.
    ops:
        Operator dictionary from `load_space` (keys: `'S'` for SU2,
        `'Sp'`/`'Sm'`/`'Sz'` for U1).
    symmetry:
        `'U1'` or `'SU2'`.
    J:
        Coupling constant (baked into `terminal_tnsr`).
    """
    if symmetry == 'SU2':
        S = ops['S']
    else:
        Sz = ops['Sz'].clone()
        Sz.insert_index(2, direction=Direction.OUT)
        S = ops['Sp'] + ops['Sm'] + Sz

    # S†: (bra, ket, op) → conjugate + swap bra/ket → scale by J.
    Sdag = S.conj().permute([1, 0, 2]) * J

    # S4: (L_trivial_IN, op_OUT, bra_OUT, ket_IN).
    S4 = S.clone()
    S4.insert_index(0, direction=Direction.IN, itag='L')
    S4 = S4.permute([0, 3, 1, 2])

    # S4dag: (op_IN, R_trivial_OUT, bra_OUT, ket_IN).
    S4dag = Sdag.clone()
    S4dag.insert_index(3, direction=Direction.OUT, itag='R')
    S4dag = S4dag.permute([2, 3, 0, 1])

    for intr in interactions:
        intr.leading_tnsr  = S4.clone()
        intr.terminal_tnsr = S4dag.clone()


def assign_freefermion(
    interactions: List,
    spc: Index,
    ops: Dict[str, Tensor],
    t: float = 1.0,
) -> None:
    """Populate `leading_tnsr` and `terminal_tnsr` for free-fermion interactions.

    Sets up the hopping term `-t (c†_i c_j + h.c.)` using the same `G4` /
    `G4dag` templates as `build_freefermion` in `system.py`.

    Parameters
    ----------
    interactions:
        List of `Interaction2Site` objects to populate.
    spc:
        Physical `Index` for fermionic sites.
    ops:
        Operator dictionary (key: `'F'` — fermionic annihilator).
    t:
        Hopping amplitude (baked into `terminal_tnsr` as `-t`).
    """
    F  = ops['F'].clone()
    C  = F.conj().permute([1, 0, 2])
    Fd = F.conj().permute([1, 0, 2])
    Cd = C.conj().permute([1, 0, 2])

    # capcup flips op directions so oplus(F, C) and oplus(Fd, Cd) are valid.
    capcup(C, 2, Cd, 2)

    # Normalize bra/ket to the full physical index.
    for op in (F, C, Fd, Cd):
        op.indices = (spc, spc.flip()) + op.indices[2:]

    # G: leading-site operator (annihilator + creator in separate op slots).
    # Gdag: terminal-site operator (creator + annihilator), scaled by -t.
    G    = oplus(F,  C,  axes=2)
    Gdag = oplus(Fd, Cd, axes=2) * (-t)

    # G4: (L_trivial_IN, op_OUT, bra_OUT, ket_IN).
    G4 = G.clone()
    G4.insert_index(0, direction=Direction.IN, itag='L')
    G4 = G4.permute([0, 3, 1, 2])

    # G4dag: (op_IN, R_trivial_OUT, bra_OUT, ket_IN).
    G4dag = Gdag.clone()
    G4dag.insert_index(3, direction=Direction.OUT, itag='R')
    G4dag = G4dag.permute([2, 3, 0, 1])

    for intr in interactions:
        intr.leading_tnsr  = G4.clone()
        intr.terminal_tnsr = G4dag.clone()


def assign_conductor(
    interactions: List,
    spc: Index,
    ops: Dict[str, Tensor],
    symmetry: str = 'U1,SU2',
    t: float = 1.0,
) -> None:
    """Populate `leading_tnsr` and `terminal_tnsr` for spinful conductor interactions.

    Sets up `-t Σ_σ (c†_{i,σ} c_{j,σ} + h.c.)` using the same Jordan-Wigner
    (JW) dressed `G4` / `G4dag` templates as `build_conductor` in `system.py`.

    Parameters
    ----------
    interactions:
        List of `Interaction2Site` objects to populate.
    spc:
        Physical `Index` for band sites.
    ops:
        Operator dictionary (keys: `'F'` for SU2, `'F_up'`/`'F_dn'`/`'Z'`
        for Abelian symmetries).
    symmetry:
        Band symmetry string, e.g. `'U1,SU2'`, `'U1,U1'`, `'Z2,U1'`.
    t:
        Hopping amplitude (baked into `terminal_tnsr` as `-t`).
    """
    is_abelian = 'SU2' not in symmetry
    F = (ops['F_up'] + ops['F_dn']) if is_abelian else ops['F']

    Z  = ops['Z']
    ZF   = contract(Z, F, axes=(1, 0))
    C_ZF = ZF.conj().permute([1, 0, 2])

    Fd     = F.conj().permute([1, 0, 2])
    F_copy = F.clone()

    # Normalize bra/ket to the full physical index.
    for op in (ZF, C_ZF, Fd, F_copy):
        op.indices = (spc, spc.flip()) + op.indices[2:]

    # capcup flips op of C_ZF (IN→OUT) and op of F_copy (OUT→IN) simultaneously.
    capcup(C_ZF, 2, F_copy, 2)

    # G: JW annihilator + JW creator (leading site).
    # Gdag: bare creator + bare annihilator, scaled -t (terminal site).
    G    = oplus(ZF, C_ZF,   axes=2)
    Gdag = oplus(Fd, F_copy, axes=2) * (-t)

    # G4: (L_trivial_IN, op_OUT, bra_OUT, ket_IN).
    G4 = G.clone()
    G4.insert_index(0, direction=Direction.IN, itag='L')
    G4 = G4.permute([0, 3, 1, 2])

    # G4dag: (op_IN, R_trivial_OUT, bra_OUT, ket_IN).
    G4dag = Gdag.clone()
    G4dag.insert_index(3, direction=Direction.OUT, itag='R')
    G4dag = G4dag.permute([2, 3, 0, 1])

    for intr in interactions:
        intr.leading_tnsr  = G4.clone()
        intr.terminal_tnsr = G4dag.clone()
