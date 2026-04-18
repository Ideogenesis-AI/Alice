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


"""Physical operator builders for MPO construction.

Each builder calls `load_space` to obtain a physical `Index` `Spc` and an
initial operator dictionary `Op`, then enriches `Op` with derived 4th-order
MPO operator templates and returns `(Spc, Op)`.

4th-order leading-site templates follow the axis layout
`(L_trivial_IN, op_OUT, bra_IN, ket_OUT)` and are keyed with the suffix `4`
(e.g. `'S4'`, `'G4'`).  Terminal-site templates follow
`(op_IN, R_trivial_OUT, bra_IN, ket_OUT)` and are keyed with the suffix
`4dag` (e.g. `'S4dag'`, `'G4dag'`).  On-site 4th-order tensors
(`'I4'`, `'N4'`, `'Z4'`, `'NN4'`) carry trivial auxiliary bonds and
follow the layout `(aux_IN, aux_OUT, bra_IN, ket_OUT)`.
"""

from __future__ import annotations

from typing import Dict, Tuple

from nicole import Direction, Tensor
from nicole import identity, oplus, capcup, contract
from nicole import load_space
from nicole.index import Index


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _make_i4(spc: Index) -> Tensor:
    """Return a 4th-order identity tensor with trivial `L` and `R` bonds.

    The returned tensor has axes
    `(L_trivial_IN, R_trivial_OUT, bra_OUT, ket_IN)`.
    """
    I = identity(spc)
    I4 = I.clone()
    I4.insert_index(0, direction=Direction.IN,  itag='_aux_')
    I4.insert_index(1, direction=Direction.OUT, itag='_aux_')
    return I4


def _make_onsite4(op2: Tensor) -> Tensor:
    """Attach trivial `L` and `R` bonds to a 2nd-order on-site operator.

    Parameters
    ----------
    op2:
        2nd-order operator with axes `(bra_OUT, ket_IN)`.

    Returns
    -------
    Tensor
        4th-order tensor with axes
        `(L_trivial_IN, R_trivial_OUT, bra_OUT, ket_IN)`.
    """
    op4 = op2.clone()
    op4.insert_index(0, direction=Direction.IN,  itag='_aux_')
    op4.insert_index(1, direction=Direction.OUT, itag='_aux_')
    return op4


def _make_leading4(op3: Tensor) -> Tensor:
    """Convert a 3rd-order operator to a 4th-order leading-site template.

    Parameters
    ----------
    op3:
        3rd-order operator with axes `(bra_OUT, ket_IN, op_OUT)` —
        the standard layout returned by `oplus` for the leading site.

    Returns
    -------
    Tensor
        4th-order tensor with axes
        `(L_trivial_IN, op_OUT, bra_OUT, ket_IN)`.
    """
    op4 = op3.clone()
    op4.insert_index(0, direction=Direction.IN, itag='_aux_')
    # After insertion: (L, bra, ket, op) → permute to (L, op, bra, ket).
    op4 = op4.permute([0, 3, 1, 2])
    return op4


def _make_terminal4(op3: Tensor) -> Tensor:
    """Convert a 3rd-order operator to a 4th-order terminal-site template.

    Parameters
    ----------
    op3:
        3rd-order operator with axes `(bra_OUT, ket_IN, op_IN)` —
        the standard layout returned by `oplus` for the terminal site.

    Returns
    -------
    Tensor
        4th-order tensor with axes
        `(op_IN, R_trivial_OUT, bra_OUT, ket_IN)`.
    """
    op4 = op3.clone()
    op4.insert_index(3, direction=Direction.OUT, itag='_aux_')
    # After insertion: (bra, ket, op, R) → permute to (op, R, bra, ket).
    op4 = op4.permute([2, 3, 0, 1])
    return op4


def _make_spin_templates(S: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Build `Sdag`, `S4`, and `S4dag` from a 3rd-order spin operator.

    Parameters
    ----------
    S:
        3rd-order spin operator with axes `(bra_OUT, ket_IN, op_OUT)`.

    Returns
    -------
    tuple
        `(Sdag, S4, S4dag)` where `Sdag` has axes `(bra, ket, op_IN)`,
        `S4` has axes `(L_trivial_IN, op_OUT, bra_OUT, ket_IN)`, and
        `S4dag` has axes `(op_IN, R_trivial_OUT, bra_OUT, ket_IN)`.
    """
    Sdag = S.conj().permute([1, 0, 2])
    S4    = _make_leading4(S)
    S4dag = _make_terminal4(Sdag)
    return Sdag, S4, S4dag


# ---------------------------------------------------------------------------
#  Public builders
# ---------------------------------------------------------------------------

def build_bosonic(
    symmetry: str = 'U1',
    spin: float = 0.5,
) -> Tuple[Index, Dict[str, Tensor]]:
    """Load spin space and build MPO operator templates for spin models.

    Calls `load_space('Spin', symmetry, {'J': spin})` and enriches the
    returned `Op` dictionary with derived operators for use in MPO
    construction.

    Parameters
    ----------
    symmetry:
        Symmetry string — `'U1'` (conserve `S_z`) or `'SU2'` (full
        spin-rotation invariance).
    spin:
        Site spin quantum number (0.5, 1.0, …).

    Returns
    -------
    tuple
        `(Spc, Op)` where `Spc` is the physical `Index` and `Op` is the
        enriched operator dictionary.

    Notes
    -----
    New entries added to `Op`:

    - `'S'` — combined 3rd-order spin operator; for U1 this is
      `Sp + Sm + Sz` (with an op axis inserted on `Sz`).
    - `'Sdag'` — `S†` in `(bra, ket, op_IN)` layout.
    - `'S4'` — 4th-order leading-site template
      `(L_trivial_IN, op_OUT, bra_OUT, ket_IN)`.
    - `'S4dag'` — 4th-order terminal-site template
      `(op_IN, R_trivial_OUT, bra_OUT, ket_IN)`.
    - `'Sz4'`, `'Sz4dag'` *(U1 only)* — same layouts as `S4`/`S4dag`
      but for `Sz` alone; useful for anisotropic (XXZ) couplings.
    - `'I4'` — 4th-order identity tensor
      `(L_trivial_IN, R_trivial_OUT, bra_OUT, ket_IN)`.
    """
    Spc, Op = load_space('Spin', symmetry, {'J': spin})

    if 'SU2' in symmetry:
        S = Op['S']
    else:
        # Sz from load_space has no op axis; insert one so it can be
        # summed with Sp and Sm, which already carry an op axis.
        Op['Sz'].insert_index(2, direction=Direction.OUT)
        S = Op['Sp'] + Op['Sm'] + Op['Sz']
        Op['S'] = S

    Sdag, S4, S4dag = _make_spin_templates(S)
    Op['Sdag']  = Sdag
    Op['S4']    = S4
    Op['S4dag'] = S4dag

    if 'SU2' not in symmetry:
        # Sz alone is useful for the diagonal Ising channel in XXZ models.
        Sz = Op['Sz']
        Szdag = Sz.conj().permute([1, 0, 2])
        Op['Sz4']    = _make_leading4(Sz)
        Op['Sz4dag'] = _make_terminal4(Szdag)

    Op['I4'] = _make_i4(Spc)

    return Spc, Op


def build_fermionic(
    symmetry: str = 'U1',
) -> Tuple[Index, Dict[str, Tensor]]:
    """Load spinless-fermion space and build MPO operator templates.

    Calls `load_space('Ferm', symmetry)` and enriches the returned `Op`
    dictionary with derived operators for use in MPO construction.

    Parameters
    ----------
    symmetry:
        Symmetry string — `'U1'` (conserve particle number) or `'Z2'`
        (fermion parity).

    Returns
    -------
    tuple
        `(Spc, Op)` where `Spc` is the physical `Index` and `Op` is the
        enriched operator dictionary.

    Notes
    -----
    New entries added to `Op` (beyond the `'F'`, `'Z'`, `'vac'` returned
    by `load_space`):

    Hopping-type (2-site) operators:

    - `'C'` — creator with `op_OUT`; built from `F†` and adjusted via
      `capcup` so its op direction matches `F` for use in `oplus`.
    - `'Fd'` — adjoint annihilator (= `F†`) with `op_IN`; used as the
      creator channel in the terminal-site operator.
    - `'Cd'` — adjoint creator with `op_IN`; built from `C†` and adjusted
      via `capcup`.
    - `'G'` — `oplus(F, C, axes=2)`, combined 3rd-order leading-site
      operator (annihilation and creation channels share the op axis).
    - `'Gdag'` — `oplus(Fd, Cd, axes=2)`, combined 3rd-order
      terminal-site operator.
    - `'G4'`, `'G4dag'` — 4th-order leading- and terminal-site templates.

    On-site operators:

    - `'N'` — 2nd-order number operator `c†c` in `(bra, ket)` layout.
    - `'N4'`, `'I4'`, `'Z4'` — 4th-order on-site tensors with trivial
      `L`/`R` bonds.
    """
    Spc, Op = load_space('Ferm', symmetry)

    F  = Op['F']
    # Build creator (C) and its adjoint (Cd) for Gdag. capcup flips the op
    # directions of C (IN→OUT) and Cd (OUT→IN) simultaneously, enabling
    # oplus(F, C) and oplus(Fd, Cd) where all op axes share the same direction.
    C  = F.conj().permute([1, 0, 2])
    Fd = F.conj().permute([1, 0, 2])
    Cd = C.conj().permute([1, 0, 2])
    capcup(C, 2, Cd, 2)

    # After conj/permute operations the bra and ket indices of C and Cd carry
    # negated charge sectors (e.g. {0,−1} vs {0,1}) that would block oplus.
    # Reset them to the canonical physical index and its dual.
    for _op in (F, C, Fd, Cd):
        _op.indices = (Spc, Spc.flip()) + _op.indices[2:]

    Op['C']  = C
    Op['Fd'] = Fd
    Op['Cd'] = Cd

    G    = oplus(F, C,   axes=2)
    Gdag = oplus(Fd, Cd, axes=2)
    Op['G']    = G
    Op['Gdag'] = Gdag

    Op['G4']    = _make_leading4(G)
    Op['G4dag'] = _make_terminal4(Gdag)

    # N = c†c: Fd (op_IN) contracted with F (op_OUT) over both the op axis
    # and the connecting physical index (ket of Fd, bra of F). The opposite
    # op directions satisfy charge conservation and sum over charge sectors.
    N = contract(Fd, F, axes=([1, 2], [0, 2]))
    Op['N']  = N
    Op['N4'] = _make_onsite4(N)

    Op['I4'] = _make_i4(Spc)

    Z = Op['Z']
    Op['Z4'] = _make_onsite4(Z)

    return Spc, Op


def build_conductor(
    symmetry: str = 'U1,SU2',
) -> Tuple[Index, Dict[str, Tensor]]:
    """Load spinful-fermion (Band) space and build MPO operator templates.

    Calls `load_space('Band', symmetry)` and enriches the returned `Op`
    dictionary with derived operators for use in MPO construction of
    spinful tight-binding and Hubbard-type models.

    Parameters
    ----------
    symmetry:
        Band symmetry string — `'U1,U1'`, `'Z2,U1'`, `'U1,SU2'`
        (default), or `'Z2,SU2'`.

    Returns
    -------
    tuple
        `(Spc, Op)` where `Spc` is the physical `Index` and `Op` is the
        enriched operator dictionary.

    Notes
    -----
    New entries added to `Op` (beyond what `load_space` returns):

    Hopping-type (2-site) operators — Jordan-Wigner dressed:

    - `'F'` *(Abelian only)* — combined annihilator `F_up + F_dn`;
      overwrites the key for uniformity with the SU2 case.
    - `'ZF'` — JW-dressed annihilator `Z × F` for the leading site.
    - `'ZC'` — JW creator `(ZF)†` with `op_OUT` (adjusted via
      `capcup`); leading-site creation channel.
    - `'Fd'` — bare creator `F†` with `op_IN`; terminal-site creation
      channel.
    - `'Cd'` — bare annihilator copy with `op_IN` (adjusted via
      `capcup`); terminal-site annihilation channel.
    - `'G'` — `oplus(ZF, ZC, axes=2)`, JW combined leading-site op.
    - `'Gdag'` — `oplus(Fd, Cd, axes=2)`, bare terminal-site op.
    - `'G4'`, `'G4dag'` — 4th-order leading- and terminal-site templates.

    Spin operators:

    - `'S'` — 3rd-order spin operator; for SU2 taken directly from
      `Op['S']`; for Abelian built as `Sp + Sm + Sz` (with op axis
      inserted on `Sz`, which `load_space` returns without one).
    - `'Sdag'`, `'S4'`, `'S4dag'` — adjoint and 4th-order templates.
    - `'Sz4'`, `'Sz4dag'` *(Abelian only)* — templates for `Sz` alone.

    On-site operators:

    - `'N'` — 2nd-order total number operator `n_up + n_dn`.
    - `'NN'` — 2nd-order double-occupancy operator `n_up × n_dn`.
    - `'N4'`, `'NN4'`, `'I4'`, `'Z4'` — 4th-order on-site tensors.
    """
    Spc, Op = load_space('Band', symmetry)

    is_abelian = 'SU2' not in symmetry

    # -----------------------------------------------------------------------
    # Combined annihilator F
    # -----------------------------------------------------------------------
    if is_abelian:
        # F_up and F_dn occupy different charge sectors of the op index;
        # their sum produces a single F with both spin sectors.
        F = Op['F_up'] + Op['F_dn']
        Op['F'] = F
    else:
        F = Op['F']

    # -----------------------------------------------------------------------
    # JW-dressed hopping operators
    # -----------------------------------------------------------------------
    Z  = Op['Z']
    ZF = contract(Z, F, axes=(1, 0))
    ZC = ZF.conj().permute([1, 0, 2])

    Fd = F.conj().permute([1, 0, 2])
    Cd = F.clone()

    # Reset bra/ket to the canonical physical index and its dual before
    # capcup and oplus, for the same reason as in build_fermionic.
    for _op in (ZF, ZC, Fd, Cd):
        _op.indices = (Spc, Spc.flip()) + _op.indices[2:]

    # capcup flips op of ZC (IN→OUT) and op of Cd (OUT→IN) together,
    # so oplus(ZF, ZC) and oplus(Fd, Cd) receive consistently directed
    # op axes. Individual inversions would break charge conservation.
    capcup(ZC, 2, Cd, 2)

    Op['ZF'] = ZF
    Op['ZC'] = ZC
    Op['Fd'] = Fd
    Op['Cd'] = Cd

    G    = oplus(ZF, ZC, axes=2)
    Gdag = oplus(Fd, Cd, axes=2)
    Op['G']    = G
    Op['Gdag'] = Gdag

    Op['G4']    = _make_leading4(G)
    Op['G4dag'] = _make_terminal4(Gdag)

    # -----------------------------------------------------------------------
    # Spin operators
    # -----------------------------------------------------------------------
    if is_abelian:
        # load_space('Band', Abelian) provides Sp, Sm (with op axis) and Sz
        # (without op axis). Insert an op axis on Sz before combining.
        Op['Sz'].insert_index(2, direction=Direction.OUT)
        S = Op['Sp'] + Op['Sm'] + Op['Sz']
        Op['S'] = S
    else:
        S = Op['S']

    Sdag, S4, S4dag = _make_spin_templates(S)
    Op['Sdag']  = Sdag
    Op['S4']    = S4
    Op['S4dag'] = S4dag

    if is_abelian:
        Sz    = Op['Sz']
        Szdag = Sz.conj().permute([1, 0, 2])
        Op['Sz4']    = _make_leading4(Sz)
        Op['Sz4dag'] = _make_terminal4(Szdag)

    # -----------------------------------------------------------------------
    # Number operators
    # -----------------------------------------------------------------------
    if is_abelian:
        F_up  = Op['F_up']
        F_dn  = Op['F_dn']
        Fd_up = F_up.conj().permute([1, 0, 2])
        Fd_dn = F_dn.conj().permute([1, 0, 2])
        # n_σ = c†_σ c_σ: contract creator (op_IN) with annihilator (op_OUT)
        # over the op axis and the connecting physical index.
        n_up = contract(Fd_up, F_up, axes=([1, 2], [0, 2]))
        n_dn = contract(Fd_dn, F_dn, axes=([1, 2], [0, 2]))
        N  = n_up + n_dn
        NN = contract(n_up, n_dn, axes=([1], [0]))
    else:
        # For SU2, F covers both spin channels; contracting over op sums them.
        N = contract(Fd, F, axes=([1, 2], [0, 2]))
        # NN = (n² - n) / 2 follows from n_σ² = n_σ and n = n_up + n_dn.
        N_sq = contract(N, N, axes=([1], [0]))
        NN   = (N_sq - N) * 0.5

    Op['N']   = N
    Op['NN']  = NN
    Op['N4']  = _make_onsite4(N)
    Op['NN4'] = _make_onsite4(NN)

    Op['I4'] = _make_i4(Spc)
    Op['Z4'] = _make_onsite4(Z)

    return Spc, Op
