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


"""Iterative diagonalization for the spin chain (Heisenberg model).

Implements a site-by-site iterative diagonalization algorithm that computes the
ground-state energy and MPS representation of the Heisenberg Hamiltonian

    H = J Σ_i (S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+ + S_i^z S_{i+1}^z)

for arbitrary spin quantum number. The algorithm grows the chain one site at a
time, diagonalizes the Hamiltonian at each step, and truncates to keep at most
`Nkeep` lowest-energy states.

For spin-1/2 chains the exact ground-state energy per site in the thermodynamic
limit is E_exact/N = 1/4 - ln(2) ≈ -0.443147.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from nicole import Direction, Tensor
from nicole import einsum, identity, isometry, diag, load_space
from nicole.decomp import eig


def iter_diag_spin(
    N: int = 50,
    Nkeep: int = 300,
    J: float = 1.0,
    spin: float = 0.5,
    symmetry: str = 'U1',
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Tensor]]:
    """Run iterative diagonalization for the Heisenberg spin chain.

    Parameters
    ----------
    N:
        Chain length.
    Nkeep:
        Maximum number of states kept after each truncation step. For
        `symmetry='SU2'` this counts SU(2) multiplets.
    J:
        Spin-spin coupling constant.
    spin:
        Site spin quantum number (0.5, 1.0, 1.5, …).
    symmetry:
        Symmetry exploited during diagonalization: `'U1'` (conserve S^z) or
        `'SU2'` (full spin-rotation symmetry).
    verbose:
        Print per-step progress and final summary when `True`.

    Returns
    -------
    Eg:
        Ground-state energy at each iteration, shape `(N,)`.
    Egs:
        Ground-state energy per site at each iteration, shape `(N,)`.
    mps:
        Left-isometry tensors produced at each iteration (length `N`).
        Each tensor has axes `(left_bond, right_bond, physical)` with itags
        `(R{i-1:02d}, R{i:02d}, s{i:02d})` (site 0 uses `L00` for the left bond).
    """
    if symmetry not in ('U1', 'SU2'):
        raise ValueError(f"symmetry must be 'U1' or 'SU2', got '{symmetry}'")

    Spc, Op = load_space('Spin', symmetry, {'J': spin})

    # U1: Sz lacks an op axis by default; insert one so it can be summed with Sp and Sm.
    # S has axes (bra, ket, op); the op index encodes the spin-operator channel.
    if symmetry == 'U1':
        Op['Sz'].insert_index(2, direction=Direction.OUT)
        S = Op['Sp'] + Op['Sm'] + Op['Sz']
    else:
        S = Op['S']
    I = identity(Spc)

    S.retag(['s00', 's00', 'op'])

    # Tiny non-zero placeholder so diag() has a well-defined block structure at site 1.
    H0 = I * 1e-30
    H0.retag(['s00', 's00'])

    # A0 maps vacuum ⊗ physical → (vacuum, bond, phys); permute to (vacuum, bond, phys).
    A0 = isometry(Op['vac'], Spc).permute([0, 2, 1])
    A0.retag(['L00', 'R00', 's00'])

    Eg = np.zeros(N)
    bond_index = None
    Sprev = None   # spin operator accumulated in the truncated left-block basis
    mps: List[Tensor] = []

    for itN in range(1, N + 1):
        # Clone S with site-specific itags for the current physical index.
        Snow = S.clone()
        Snow.retag([f's{itN-1:02d}', f's{itN-1:02d}', 'op'])

        if itN == 1:
            Anow = A0
            # Project H0 into the one-site basis; a = vacuum (dim-1) is summed trivially.
            # 'abg,gh,adh->bd': A†[a,b,g], H0[g,h], A[a,d,h] → Hnow[b,d]
            Hnow = einsum('abg,gh,adh->bd', Anow.conj(), H0, Anow)
        else:
            # Grow the isometry: (left, phys, right) → permute to (left, right, phys).
            Anow = isometry(bond_index.flip(), Spc).permute([0, 2, 1])
            Anow.retag([f'R{itN-2:02d}', f'R{itN-1:02d}', f's{itN-1:02d}'])

            # Renormalize H into the enlarged one-site basis: A†.Hprev.A.
            # 'ae,ehg,abg->bh': Hprev[a,e], A[e,h,g], A†[a,b,g] → Hnow[b,h]
            Hnow = einsum('ae,ehg,abg->bh', Hprev, Anow, Anow.conj())

            # Spin-spin coupling: S†_prev (left block) ⊗ S_now (new site), sandwiched by A.
            # Sn = S†_now permuted to (op, ket, bra) so its op index pairs with Sprev's op.
            # 'okg,edg,qeo,qbk->bd': Sn[o,k,g], A[e,d,g], Sprev[q,e,o], A†[q,b,k] → HSS[b,d]
            Sn = Snow.conj().permute([2, 1, 0])
            HSS = einsum('okg,edg,qeo,qbk->bd', Sn, Anow, Sprev, Anow.conj())
            Hnow = Hnow + HSS * J

        # Symmetrize explicitly to suppress floating-point asymmetry before diag.
        Hnow_sym = (Hnow + Hnow.conj().transpose()) * 0.5

        if itN == 1:
            V, D = eig(Hnow_sym, is_hermitian=True)
        elif itN == N:
            # Last site: keep only the ground state.
            V, D = eig(Hnow_sym, trunc={'nkeep': 1}, is_hermitian=True)
        else:
            V, D = eig(Hnow_sym, trunc={'nkeep': Nkeep}, is_hermitian=True)

        V.retag([f'R{itN-1:02d}', f'R{itN-1:02d}'])

        all_eigvals = np.concatenate([eigvals for eigvals in D.values()])
        Eg[itN - 1] = np.min(all_eigvals)

        # Absorb V into Anow to form the left-isometry AK: axes (left, right_new, phys).
        # 'abg,bd->adg': A[a,b,g], V[b,d] → AK[a,d,g]
        AK = einsum('abg,bd->adg', Anow, V)
        mps.append(AK.clone())

        bond_index = V.indices[1]
        # Hprev is diagonal in the truncated eigenbasis; feeds directly into next Hnow.
        Hprev = diag(D, bond_index, itags=(f'R{itN-1:02d}', f'R{itN-1:02d}'))

        # Project S_now into the new truncated basis for the next inter-site coupling.
        # 'qdk,gko,qbg->bdo': AK[q,d,k], S[g,k,o], AK†[q,b,g] → Sprev[b,d,o]
        Sprev = einsum('qdk,gko,qbg->bdo', AK, Snow, AK.conj())

        if verbose:
            dim_fn = (lambda idx: idx.num_states) if symmetry == 'SU2' else (lambda idx: idx.dim)
            print(f'#{itN:02d}/{N:02d} : NK={dim_fn(AK.indices[0])}/{dim_fn(Hnow.indices[1])}')

    Egs = Eg / np.arange(1, N + 1)

    if verbose and spin == 0.5:
        Eexact = 0.25 - np.log(2)
        print(f'\nExact GS energy per site (N→∞): {Eexact:.6f}')
        print(f'Final iterative estimate:        {Egs[-1]:.6f}')

    return Eg, Egs, mps
