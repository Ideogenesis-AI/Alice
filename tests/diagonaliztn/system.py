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


"""MPO builders for standard quantum lattice models. (test helpers)

Each builder returns a list of Nicole `Tensor` objects in the axis layout
expected by `alice.network.observe`:

    axis 0 — left MPO bond  (IN direction)
    axis 1 — right MPO bond (OUT direction)
    axis 2 — phys_bra       (OUT direction)
    axis 3 — phys_ket       (IN direction)

Bond itags follow the `W{i:02d}` / `W{i+1:02d}` convention.
"""

from __future__ import annotations

from typing import List

import torch

from nicole import Direction, Tensor
from nicole import identity, oplus, capcup, contract
from nicole import load_space
from nicole.index import Index
from nicole.symmetry.delegate import Bridge


def _make_zero_mid(op_idx: Index, zero4: Tensor) -> Tensor:
    """Build a zero tensor whose bond axes match the charge sectors of `op_idx`.

    In an MPO matrix, off-diagonal zero blocks in an operator row must be
    structurally compatible with the non-zero operator block in the same row.
    When the operator has no charge-0 sector (e.g. SU(2) spin operator, or a
    fermionic annihilation operator), a naïve `op * 0` tensor carries the
    wrong bond sector set and causes `oplus` to fail. This helper constructs
    the correct zero tensor for those positions.

    Parameters
    ----------
    op_idx:
        The op axis (axis 2) of the operator tensor. Its charge sectors
        determine the bond sectors of the output.
    zero4:
        A zero tensor with shape `(left, right, bra, ket)` built from the
        physical identity. Supplies the physical charge sectors and the
        `intw` field (non-`None` for non-Abelian groups).

    Returns
    -------
    Tensor
        Zero tensor with indices
        `(op_idx.flip(), op_idx, zero4.indices[2], zero4.indices[3])`.
    """
    _op_sdm  = op_idx.sector_dim_map()
    _bra_sdm = zero4.indices[2].sector_dim_map()
    _ket_sdm = zero4.indices[3].sector_dim_map()
    
    # Bond axes: op_idx.flip() for left (IN) and op_idx for right (OUT),
    # so the block key (lc, lc, bc, bc) is diagonal in both bond and physical charges
    # and satisfies charge conservation for any symmetry group.
    _zp_idx  = (op_idx.flip(), op_idx, zero4.indices[2], zero4.indices[3])
    _zp_data = {}
    # Non-None intw signals a non-Abelian group; each block then needs a Bridge intertwiner.
    _zp_intw = {} if zero4.intw is not None else None

    for lc, ld in _op_sdm.items():
        for bc, bd in _bra_sdm.items():
            kd = _ket_sdm.get(bc, 0)
            if kd == 0:
                continue
            key = (lc, lc, bc, bc)
            if _zp_intw is None:
                _zp_data[key] = torch.zeros(ld, ld, bd, kd, dtype=torch.float64)
            else:
                # Bridge encodes the Clebsch-Gordan structure for non-Abelian blocks.
                bridge = Bridge.from_block(
                    op_idx.group, key,
                    [_zp_idx[i].direction for i in range(4)],
                )
                _zp_data[key] = torch.zeros(
                    ld, ld, bd, kd, bridge.num_components, dtype=torch.float64
                )
                _zp_intw[key] = bridge

    zero_mid = zero4.clone()
    zero_mid.indices = _zp_idx
    zero_mid.data    = _zp_data
    zero_mid.intw    = _zp_intw

    return zero_mid


def build_heisenberg(
    N: int = 50,
    J: float = 1.0,
    spin: float = 0.5,
    symmetry: str = 'U1',
) -> List[Tensor]:
    """Build an MPO for the Heisenberg Hamiltonian.

    The Hamiltonian is

        H = J Σ_i S_i† · S_{i+1}

    The MPO uses bond dimension 3 with the block structure

        first site:   [0,  S,  I]
        middle sites: [[I, 0, 0], [S†, 0, 0], [0, S, I]]
        last site:    [I, S†, 0]^T

    Parameters
    ----------
    N:
        Chain length.
    J:
        Spin-spin coupling constant.
    spin:
        Site spin quantum number (0.5, 1.0, …).
    symmetry:
        `'U1'` (conserve S^z) or `'SU2'` (full spin-rotation symmetry).

    Returns
    -------
    List[Tensor]
        MPO tensors with axes `(left=IN, right=OUT, phys_bra=OUT, phys_ket=IN)`
        and itags `['W{i:02d}', 'W{i+1:02d}', 's{i:02d}', 's{i:02d}']`.
    """
    Spc, Op = load_space('Spin', symmetry, {'J': spin})

    # U1: Sz lacks an op axis by default; insert one so it can be summed with Sp and Sm.
    if symmetry == 'SU2':
        S = Op['S']
    else:
        Op['Sz'].insert_index(2, direction=Direction.OUT)
        S = Op['Sp'] + Op['Sm'] + Op['Sz']

    # Sdag = S† in (bra, ket, op) layout: swap bra/ket via permute, then scale by J.
    Sdag = S.conj().permute([1, 0, 2]) * J

    I = identity(Spc)

    # I4 and zero4 are 4-index base templates (left, right, bra, ket) retagged per site.
    I4 = I.clone()
    I4.insert_index(0, direction=Direction.IN,  itag='L')
    I4.insert_index(1, direction=Direction.OUT, itag='R')

    zero4 = (I * 0.0).clone()
    zero4.insert_index(0, direction=Direction.IN,  itag='L')
    zero4.insert_index(1, direction=Direction.OUT, itag='R')

    # S with left bond: (bra, ket, op) → (left, bra, ket, op) → (left, op, bra, ket).
    # The op axis (axis 1) is left untagged; oplus merges it as the new right bond slot.
    S4 = S.clone()
    S4.insert_index(0, direction=Direction.IN, itag='L')
    S4 = S4.permute([0, 3, 1, 2])

    # S† with right bond: (bra, ket, op) → (bra, ket, op, right) → (op, right, bra, ket).
    # The op axis (axis 0) is left untagged; oplus merges it as the new left bond slot.
    S4dag = Sdag.clone()
    S4dag.insert_index(3, direction=Direction.OUT, itag='R')
    S4dag = S4dag.permute([2, 3, 0, 1])

    # zero_mid is needed for the off-diagonal zero blocks in the S† row: a naïve `op * 0`
    # carries the wrong bond sectors when the op index lacks a charge-0 sector.
    zero_mid = _make_zero_mid(S.indices[2], zero4)

    mpo: List[Tensor] = []

    for i in range(N):
        if i == 0:
            # First site: row vector [0, S, I] — oplus grows the right bond.
            W = zero4.clone()
            W.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])

            S_copy = S4.clone()
            S_copy.retag([0, 2, 3], [f'W{i:02d}', f's{i:02d}', f's{i:02d}'])
            W = oplus(W, S_copy, axes=[1])

            I_copy = I4.clone()
            I_copy.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            W = oplus(W, I_copy, axes=[1])

        elif i == N - 1:
            # Last site: column vector [I, S†, 0]^T — oplus grows the left bond.
            W = I4.clone()
            W.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])

            Sdag_copy = S4dag.clone()
            Sdag_copy.retag([1, 2, 3], [f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            W = oplus(W, Sdag_copy, axes=[0])

            zero_copy = zero4.clone()
            zero_copy.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            W = oplus(W, zero_copy, axes=[0])

        else:
            # Middle sites: 3×3 block matrix [[I, 0, 0], [S†, 0, 0], [0, S, I]].
            # oplus along axis 1 concatenates right bonds (columns); along axis 0, left bonds (rows).

            # Row 0: [I, 0, 0] — r0c1 uses S4*0 (not zero4) to match S's bond sector layout.
            r0c0 = I4.clone()
            r0c0.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r0c1 = S4.clone() * 0
            r0c1.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r0c2 = zero4.clone()
            r0c2.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            row0 = oplus(oplus(r0c0, r0c1, axes=[1]), r0c2, axes=[1])

            # Row 1: [S†, 0, 0] — r1c1 uses zero_mid because S†'s op has no charge-0 sector.
            r1c0 = S4dag.clone()
            r1c0.retag([1, 2, 3], [f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r1c1 = zero_mid.clone()
            r1c1.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r1c2 = S4dag.clone() * 0
            r1c2.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            row1 = oplus(oplus(r1c0, r1c1, axes=[1]), r1c2, axes=[1])

            # Row 2: [0, S, I]
            r2c0 = zero4.clone()
            r2c0.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r2c1 = S4.clone()
            r2c1.retag([0, 2, 3], [f'W{i:02d}', f's{i:02d}', f's{i:02d}'])
            r2c2 = I4.clone()
            r2c2.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            row2 = oplus(oplus(r2c0, r2c1, axes=[1]), r2c2, axes=[1])

            W = oplus(oplus(row0, row1, axes=[0]), row2, axes=[0])

        mpo.append(W)

    return mpo


def build_freefermion(
    N: int = 50,
    t: float = 1.0,
    symmetry: str = 'U1',
) -> List[Tensor]:
    """Build an MPO for the free spinless tight-binding Hamiltonian.

    The Hamiltonian is

        H = -t Σ_i (c†_i c_{i+1} + h.c.)

    The MPO uses bond dimension 3. Separate op-axis slots for the annihilation
    channel (F) and creation channel (C) prevent spurious cross-terms under Z2
    symmetry.

    Parameters
    ----------
    N:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.
    symmetry:
        `'U1'` (conserve particle number) or `'Z2'` (fermion parity).

    Returns
    -------
    List[Tensor]
        MPO tensors with axes `(left=IN, right=OUT, phys_bra=OUT, phys_ket=IN)`
        and itags `['W{i:02d}', 'W{i+1:02d}', 's{i:02d}', 's{i:02d}']`.
    """
    Spc, Op = load_space('Ferm', symmetry)

    # F  : annihilation, axes (bra=IN, ket=OUT, op=OUT).
    # C  : F†, swap bra/ket via permute — axes (bra=IN, ket=OUT, op=IN).
    # Fd : same data as C for real F (used as the right-site partner in Gdag).
    # Cd : C† = F for real operators (op direction flipped back to OUT).
    F  = Op['F']
    C  = F.conj().permute([1, 0, 2])
    Fd = F.conj().permute([1, 0, 2])
    Cd = C.conj().permute([1, 0, 2])

    # capcup flips op of C (IN→OUT) and op of Cd (OUT→IN) simultaneously, so
    # F and C share op=OUT for oplus(F, C) and Fd and Cd share op=IN for oplus(Fd, Cd).
    capcup(C, 2, Cd, 2)

    # Normalize bra/ket to the full physical index so oplus can merge the op axis (axis 2)
    # regardless of which individual bra/ket charge sectors each operator occupies.
    for op in (F, C, Fd, Cd):
        op.indices = (Spc, Spc.flip()) + op.indices[2:]

    # G keeps annihilation (F, slot 0) and creation (C, slot 1) in separate op-axis slots.
    # Keeping them separate prevents spurious c†c†/cc cross-terms under Z2 symmetry,
    # where both channels share the same Z2 charge and would otherwise collapse.
    G    = oplus(F, C,   axes=2)
    Gdag = oplus(Fd, Cd, axes=2) * (-t)

    I = identity(Spc)

    # I4 and zero4 are 4-index base templates (left, right, bra, ket) retagged per site.
    I4 = I.clone()
    I4.insert_index(0, direction=Direction.IN,  itag='L')
    I4.insert_index(1, direction=Direction.OUT, itag='R')

    zero4 = (I * 0.0).clone()
    zero4.insert_index(0, direction=Direction.IN,  itag='L')
    zero4.insert_index(1, direction=Direction.OUT, itag='R')

    # zero_mid for the off-diagonal blocks in the Gdag row (G's op has no charge-0 sector).
    zero_mid = _make_zero_mid(G.indices[2], zero4)

    # G with left bond: (bra, ket, op) → (left, bra, ket, op) → (left, op, bra, ket).
    # The op axis (axis 1) is left untagged; oplus merges it as the new right bond slot.
    G4 = G.clone()
    G4.insert_index(0, direction=Direction.IN, itag='L')
    G4 = G4.permute([0, 3, 1, 2])

    # Gdag with right bond: (bra, ket, op) → (bra, ket, op, right) → (op, right, bra, ket).
    # The op axis (axis 0) is left untagged; oplus merges it as the new left bond slot.
    G4dag = Gdag.clone()
    G4dag.insert_index(3, direction=Direction.OUT, itag='R')
    G4dag = G4dag.permute([2, 3, 0, 1])

    mpo: List[Tensor] = []

    for i in range(N):
        if i == 0:
            # First site: row vector [0, G, I] — oplus grows the right bond.
            W = zero4.clone()
            W.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])

            G_copy = G4.clone()
            G_copy.retag([0, 2, 3], [f'W{i:02d}', f's{i:02d}', f's{i:02d}'])
            W = oplus(W, G_copy, axes=[1])

            I_copy = I4.clone()
            I_copy.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            W = oplus(W, I_copy, axes=[1])

        elif i == N - 1:
            # Last site: column vector [I, Gdag, 0]^T — oplus grows the left bond.
            W = I4.clone()
            W.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])

            Gdag_copy = G4dag.clone()
            Gdag_copy.retag([1, 2, 3], [f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            W = oplus(W, Gdag_copy, axes=[0])

            zero_copy = zero4.clone()
            zero_copy.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            W = oplus(W, zero_copy, axes=[0])

        else:
            # Middle sites: 3×3 block matrix [[I, 0, 0], [Gdag, 0, 0], [0, G, I]].

            # Row 0: [I, 0, 0] — r0c1 uses G4*0 (not zero4) to match G's bond sector layout.
            r0c0 = I4.clone()
            r0c0.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r0c1 = G4.clone() * 0
            r0c1.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r0c2 = zero4.clone()
            r0c2.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            row0 = oplus(oplus(r0c0, r0c1, axes=[1]), r0c2, axes=[1])

            # Row 1: [Gdag, 0, 0] — r1c1 uses zero_mid because Gdag's op has no charge-0 sector.
            r1c0 = G4dag.clone()
            r1c0.retag([1, 2, 3], [f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r1c1 = zero_mid.clone()
            r1c1.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r1c2 = G4dag.clone() * 0
            r1c2.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            row1 = oplus(oplus(r1c0, r1c1, axes=[1]), r1c2, axes=[1])

            # Row 2: [0, G, I]
            r2c0 = zero4.clone()
            r2c0.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r2c1 = G4.clone()
            r2c1.retag([0, 2, 3], [f'W{i:02d}', f's{i:02d}', f's{i:02d}'])
            r2c2 = I4.clone()
            r2c2.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            row2 = oplus(oplus(r2c0, r2c1, axes=[1]), r2c2, axes=[1])

            W = oplus(oplus(row0, row1, axes=[0]), row2, axes=[0])

        mpo.append(W)

    return mpo


def build_conductor(
    N: int = 50,
    t: float = 1.0,
    symmetry: str = 'U1,SU2',
) -> List[Tensor]:
    """Build an MPO for the free spinful tight-binding (band conductor) Hamiltonian.

    The Hamiltonian is

        H = -t Σ_{i,σ} (c†_{i,σ} c_{i+1,σ} + h.c.)

    The MPO uses bond dimension 3. The Jordan-Wigner (JW) string Z_i = (-1)^{N_i}
    belongs on the LEFT site of each bond, so the left-site operator is

        G    = oplus(ZF, (ZF)†, axes=2)
        Gdag = oplus(Fd, F, axes=2) * (-t)

    Parameters
    ----------
    N:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.
    symmetry:
        Band symmetry: `'U1,U1'`, `'Z2,U1'`, `'U1,SU2'` (default), or
        `'Z2,SU2'`.

    Returns
    -------
    List[Tensor]
        MPO tensors with axes `(left=IN, right=OUT, phys_bra=OUT, phys_ket=IN)`
        and itags `['W{i:02d}', 'W{i+1:02d}', 's{i:02d}', 's{i:02d}']`.
    """
    Spc, Op = load_space('Band', symmetry)

    # For Abelian symmetries, spin flavours have distinct op charges and are combined
    # by addition so the op index carries both spin channels, identical in structure
    # to the single SU(2) doublet F used for non-Abelian symmetries.
    is_abelian = 'SU2' not in symmetry
    F = Op['F_up'] + Op['F_dn'] if is_abelian else Op['F']

    # ZF = Z×F: the JW-dressed annihilator for the LEFT site of each bond.
    #   c†_{σ,i} c_{σ,i+1} = (ZF)†_{σ,i} ⊗ F_{σ,i+1}
    # Z = (-1)^{N_i} accounts for the parity of all modes within site i.
    Z  = Op['Z']
    ZF   = contract(Z, F, axes=(1, 0))
    # C_ZF = (ZF)†: JW creator; conj + permute gives (bra=IN, ket=OUT, op=IN).
    C_ZF = ZF.conj().permute([1, 0, 2])

    # Gdag uses bare operators (no Z) on the RIGHT site.
    Fd     = F.conj().permute([1, 0, 2])   # bare creator,      op=IN
    F_copy = F.clone()                      # bare annihilator,  op=OUT

    # Normalize bra/ket to the full physical index so oplus can merge the op axis (axis 2)
    # regardless of which individual bra/ket charge sectors each operator occupies.
    for op in (ZF, C_ZF, Fd, F_copy):
        op.indices = (Spc, Spc.flip()) + op.indices[2:]

    # capcup flips op of C_ZF (IN→OUT) and op of F_copy (OUT→IN) simultaneously so
    # oplus(ZF, C_ZF) and oplus(Fd, F_copy) are valid.
    # Must be done together; individual inversions would break charge conservation.
    capcup(C_ZF, 2, F_copy, 2)

    # G: left-site operator — JW annihilator (slot 0) + JW creator (slot 1).
    # Gdag: right-site operator — bare creator (slot 0) + bare annihilator (slot 1), scaled −t.
    G    = oplus(ZF, C_ZF,   axes=2)
    Gdag = oplus(Fd, F_copy, axes=2) * (-t)

    I = identity(Spc)

    # I4 and zero4 are 4-index base templates (left, right, bra, ket) retagged per site.
    I4 = I.clone()
    I4.insert_index(0, direction=Direction.IN,  itag='L')
    I4.insert_index(1, direction=Direction.OUT, itag='R')

    zero4 = (I * 0.0).clone()
    zero4.insert_index(0, direction=Direction.IN,  itag='L')
    zero4.insert_index(1, direction=Direction.OUT, itag='R')

    # zero_mid for the off-diagonal blocks in the Gdag row (G's op has no charge-0 sector).
    zero_mid = _make_zero_mid(G.indices[2], zero4)

    # G with left bond: (bra, ket, op) → (left, bra, ket, op) → (left, op, bra, ket).
    # The op axis (axis 1) is left untagged; oplus merges it as the new right bond slot.
    G4 = G.clone()
    G4.insert_index(0, direction=Direction.IN, itag='L')
    G4 = G4.permute([0, 3, 1, 2])

    # Gdag with right bond: (bra, ket, op) → (bra, ket, op, right) → (op, right, bra, ket).
    # The op axis (axis 0) is left untagged; oplus merges it as the new left bond slot.
    G4dag = Gdag.clone()
    G4dag.insert_index(3, direction=Direction.OUT, itag='R')
    G4dag = G4dag.permute([2, 3, 0, 1])

    mpo: List[Tensor] = []

    for i in range(N):
        if i == 0:
            # First site: row vector [0, G, I] — oplus grows the right bond.
            W = zero4.clone()
            W.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])

            G_copy = G4.clone()
            G_copy.retag([0, 2, 3], [f'W{i:02d}', f's{i:02d}', f's{i:02d}'])
            W = oplus(W, G_copy, axes=[1])

            I_copy = I4.clone()
            I_copy.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            W = oplus(W, I_copy, axes=[1])

        elif i == N - 1:
            # Last site: column vector [I, Gdag, 0]^T — oplus grows the left bond.
            W = I4.clone()
            W.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])

            Gdag_copy = G4dag.clone()
            Gdag_copy.retag([1, 2, 3], [f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            W = oplus(W, Gdag_copy, axes=[0])

            zero_copy = zero4.clone()
            zero_copy.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            W = oplus(W, zero_copy, axes=[0])

        else:
            # Middle sites: 3×3 block matrix [[I, 0, 0], [Gdag, 0, 0], [0, G, I]].

            # Row 0: [I, 0, 0] — r0c1 uses G4*0 (not zero4) to match G's bond sector layout.
            r0c0 = I4.clone()
            r0c0.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r0c1 = G4.clone() * 0
            r0c1.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r0c2 = zero4.clone()
            r0c2.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            row0 = oplus(oplus(r0c0, r0c1, axes=[1]), r0c2, axes=[1])

            # Row 1: [Gdag, 0, 0] — r1c1 uses zero_mid because Gdag's op has no charge-0 sector.
            r1c0 = G4dag.clone()
            r1c0.retag([1, 2, 3], [f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r1c1 = zero_mid.clone()
            r1c1.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r1c2 = G4dag.clone() * 0
            r1c2.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            row1 = oplus(oplus(r1c0, r1c1, axes=[1]), r1c2, axes=[1])

            # Row 2: [0, G, I]
            r2c0 = zero4.clone()
            r2c0.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            r2c1 = G4.clone()
            r2c1.retag([0, 2, 3], [f'W{i:02d}', f's{i:02d}', f's{i:02d}'])
            r2c2 = I4.clone()
            r2c2.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
            row2 = oplus(oplus(r2c0, r2c1, axes=[1]), r2c2, axes=[1])

            W = oplus(oplus(row0, row1, axes=[0]), row2, axes=[0])

        mpo.append(W)

    return mpo
