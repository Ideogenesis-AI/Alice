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


"""Iterative diagonalization for the free-fermion tight-binding chain.

Implements a site-by-site iterative diagonalization algorithm that computes the
ground-state energy and MPS representation of the tight-binding Hamiltonian

    H = -t Σ_i (c†_i c_{i+1} + c†_{i+1} c_i)

The algorithm grows the chain one site at a time, diagonalizes the Hamiltonian
at each step, and truncates to keep at most `Nkeep` lowest-energy states.

For the infinite chain at half-filling the exact ground-state energy per site is

    E/N = -2t/π ≈ -0.6366 t.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from nicole import Tensor
from nicole import einsum, identity, isometry, diag, load_space
from nicole.decomp import eig


def exact_halffilling_energy(N: int, t: float = 1.0) -> float:
    """Exact ground-state energy of an N-site open tight-binding chain at half-filling.

    Uses the analytic single-particle spectrum

        ε_k = -2t cos(kπ / (N+1)),  k = 1, …, N

    and fills the ⌊N/2⌋ lowest levels.

    Parameters
    ----------
    N:
        Chain length.
    t:
        Hopping amplitude.

    Returns
    -------
    float
        Exact ground-state energy at half-filling (⌊N/2⌋ particles).
    """
    n_particles = N // 2
    if n_particles == 0:
        return 0.0
    k = np.arange(1, n_particles + 1)
    return float(-2 * t * np.sum(np.cos(k * np.pi / (N + 1))))


def iter_diag_ferm(
    N: int = 50,
    Nkeep: int = 300,
    t: float = 1.0,
    symmetry: str = 'U1',
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Tensor]]:
    """Run iterative diagonalization for the free-fermion tight-binding chain.

    Parameters
    ----------
    N:
        Chain length.
    Nkeep:
        Maximum number of states kept after each truncation step.
    t:
        Nearest-neighbor hopping amplitude.
    symmetry:
        Symmetry exploited: `'U1'` (conserve particle number) or `'Z2'`
        (fermion parity only).
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
    if symmetry not in ('U1', 'Z2'):
        raise ValueError(f"symmetry must be 'U1' or 'Z2', got '{symmetry}'")

    Spc, Op = load_space('Ferm', symmetry)
    # F has axes (bra, ket, op); the op index carries the charge quantum number.
    F = Op['F']
    I = identity(Spc)

    F.retag(['s00', 's00', 'op'])

    # No on-site energy in tight-binding; tiny placeholder to give diag() a block structure.
    H0 = I * 1e-30
    H0.retag(['s00', 's00'])

    # A0 maps vacuum ⊗ physical → (vacuum, bond, phys); permute to (vacuum, bond, phys).
    A0 = isometry(Op['vac'], Spc).permute([0, 2, 1])
    A0.retag(['L00', 'R00', 's00'])

    Eg = np.zeros(N)
    bond_index = None
    Fprev = None   # annihilation operator accumulated in the truncated left-block basis
    mps: List[Tensor] = []

    for itN in range(1, N + 1):
        # Clone F with site-specific itags for the current physical index.
        Fnow = F.clone()
        Fnow.retag([f's{itN-1:02d}', f's{itN-1:02d}', 'op'])

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

            # Hopping term: F†_prev F_now, sandwiched by Anow.
            # Fn_dag = F†_now permuted to (op, ket, bra) so its op pairs with Fprev's op.
            # 'okg,edg,qeo,qbk->bd': Fn_dag[o,k,g], A[e,d,g], Fprev[q,e,o], A†[q,b,k] → HFF[b,d]
            # HFF + h.c. gives the full hopping F†_prev F_now + F†_now F_prev.
            Fn_dag = Fnow.conj().permute([2, 1, 0])
            HFF = einsum('okg,edg,qeo,qbk->bd', Fn_dag, Anow, Fprev, Anow.conj())
            Hnow = Hnow + (HFF + HFF.conj().transpose()) * (-t)

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

        # Project F_now into the new truncated basis for the next step's hopping term.
        # 'qdk,gko,qbg->bdo': AK[q,d,k], F[g,k,o], AK†[q,b,g] → Fprev[b,d,o]
        Fprev = einsum('qdk,gko,qbg->bdo', AK, Fnow, AK.conj())

        if verbose:
            print(f'#{itN:02d}/{N:02d} : NK={AK.indices[0].dim}/{Hnow.indices[1].dim}')

    Egs = Eg / np.arange(1, N + 1)

    if verbose:
        E_inf = -2 * t / np.pi
        E_exact_N = exact_halffilling_energy(N, t)
        print(f'\nExact GS energy per site ({N} sites, half-filling): {E_exact_N / N:.6f}')
        print(f'Exact GS energy per site (N→∞, half-filling):      {E_inf:.6f}')
        print(f'Final iterative estimate:                            {Egs[-1]:.6f}')

    return Eg, Egs, mps
