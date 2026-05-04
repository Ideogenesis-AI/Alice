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


"""Iterative diagonalization for the free spinful tight-binding (band conductor) chain.

Implements a site-by-site iterative diagonalization algorithm that computes the
ground-state energy and MPS representation of the spinful tight-binding Hamiltonian

    H = -t Σ_{i,σ} (c†_{i,σ} c_{i+1,σ} + h.c.)

The four-state physical site (|0⟩, |↑⟩, |↓⟩, |↑↓⟩) uses the `Band` preset.
Because the Hamiltonian is spin-diagonal, each spin species is an independent
spinless chain. At half-filling (2 electrons per site) the exact ground-state
energy per site converges to

    E/N = -4t/π ≈ -1.2732 t.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from nicole import Tensor
from nicole import einsum, identity, isometry, diag, load_space
from nicole.decomp import eig


_ABELIAN_SYMMETRIES    = ('U1,U1', 'Z2,U1')
_NONABELIAN_SYMMETRIES = ('U1,SU2', 'Z2,SU2')
_ALL_SYMMETRIES        = _ABELIAN_SYMMETRIES + _NONABELIAN_SYMMETRIES


def exact_halffilling_energy_band(N: int, t: float = 1.0) -> float:
    """Exact ground-state energy of an N-site spinful tight-binding chain at half-filling.

    The Hamiltonian is spin-diagonal, so the total energy is twice that of a
    single spinless chain:

        E = 2 × Σ_{k=1}^{⌊N/2⌋} (-2t cos(kπ / (N+1)))

    Parameters
    ----------
    N:
        Chain length.
    t:
        Hopping amplitude.

    Returns
    -------
    float
        Exact ground-state energy at half-filling.
    """
    n_particles = N // 2
    if n_particles == 0:
        return 0.0
    k = np.arange(1, n_particles + 1)
    return float(-4 * t * np.sum(np.cos(k * np.pi / (N + 1))))


def _compute_hff(ZFprev: Tensor, Fnow: Tensor, Anow: Tensor) -> Tensor:
    """One-body hopping contribution (ZF)†_prev × F_now sandwiched by Anow.

    `ZFprev` must carry the Jordan-Wigner (JW) string, i.e. it is the
    accumulated (Z × F) operator on the left-block edge, not the bare F.

    Parameters
    ----------
    ZFprev:
        Accumulated (Z × F) operator from the previous step.
    Fnow:
        Annihilation operator for the current site.
    Anow:
        Current-site isometry tensor.

    Returns
    -------
    Tensor
        Un-scaled, un-symmetrized hopping contribution in the current basis.
    """
    # Fn_dag = F†_now (bare, no Z on the right site), permuted to (op, ket, bra).
    # The op index sums over all spin/charge channels automatically.
    # 'okg,edg,qeo,qbk->bd': Fn_dag[o,k,g], A[e,d,g], ZFprev[q,e,o], A†[q,b,k] → HFF[b,d]
    Fn_dag = Fnow.conj().permute([2, 1, 0])
    return einsum('okg,edg,qeo,qbk->bd', Fn_dag, Anow, ZFprev, Anow.conj())


def _zf_product(Z: Tensor, F: Tensor) -> Tensor:
    """Return the operator product Z × F on a single physical site."""
    # Z[a,b], F[b,c,d] → ZF[a,c,d]; contracts Z's ket with F's bra.
    return einsum('ab,bcd->acd', Z, F)


def _push_zf(Z: Tensor, F: Tensor, AK: Tensor) -> Tensor:
    """Accumulate the (Z × F) operator through isometry AK for the next step."""
    ZF = _zf_product(Z, F)
    # Project ZF into the truncated basis; result has axes (bra, ket, op) = (b, d, o).
    # 'qdk,gko,qbg->bdo': AK[q,d,k], ZF[g,k,o], AK†[q,b,g] → ZFprev[b,d,o]
    return einsum('qdk,gko,qbg->bdo', AK, ZF, AK.conj())


def iter_diag_band(
    N: int = 50,
    Nkeep: int = 300,
    t: float = 1.0,
    symmetry: str = 'U1,U1',
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Tensor]]:
    """Run iterative diagonalization for the free spinful tight-binding (band) chain.

    Parameters
    ----------
    N:
        Chain length.
    Nkeep:
        Maximum number of states kept per truncation step. For non-Abelian
        symmetries this counts SU(2) multiplets.
    t:
        Nearest-neighbor hopping amplitude.
    symmetry:
        Band symmetry to exploit. Accepted values:

        * `'U1,U1'`  — particle number × spin-z  (default)
        * `'Z2,U1'`  — fermion parity × spin-z
        * `'U1,SU2'` — particle number × SU(2) spin-rotation
        * `'Z2,SU2'` — fermion parity × SU(2) spin-rotation

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
    sym_norm = symmetry.replace(' ', '')
    if sym_norm not in _ALL_SYMMETRIES:
        raise ValueError(
            f"symmetry must be one of {_ALL_SYMMETRIES}, got '{symmetry}'"
        )

    is_su2     = 'SU2' in sym_norm
    is_abelian = sym_norm in _ABELIAN_SYMMETRIES

    Spc, Op = load_space('Band', symmetry)
    I = identity(Spc)

    Z = Op['Z']
    Z.retag(['s00', 's00'])

    # For Abelian symmetries, F_up and F_dn have distinct op charges and are combined
    # by addition so the op index carries both spin channels — identical in structure
    # to the single SU(2) doublet F used for non-Abelian symmetries.
    F = Op['F_up'] + Op['F_dn'] if is_abelian else Op['F']
    F.retag(['s00', 's00', 'op'])

    # No on-site energy; tiny placeholder to give diag() a well-defined block structure.
    H0 = I * 1e-30
    H0.retag(['s00', 's00'])

    # A0 maps vacuum ⊗ physical → (vacuum, bond, phys); permute to (vacuum, bond, phys).
    A0 = isometry(Op['vac'], Spc).permute([0, 2, 1])
    A0.retag(['L00', 'R00', 's00'])

    Eg         = np.zeros(N)
    bond_index = None
    ZFprev     = None   # (Z×F) operator accumulated in the truncated left-block basis
    mps: List[Tensor] = []

    for itN in range(1, N + 1):
        # Clone Z and F with site-specific itags for the current physical index.
        Z_now = Z.clone()
        Z_now.retag([f's{itN-1:02d}', f's{itN-1:02d}'])

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

            # Hopping with JW string: (ZF)†_prev × F_now (bare on the right site).
            # The Jordan-Wigner factor Z = (-1)^N_i accounts for parity of all modes
            # within the left site when commuting c†_{σ,i} past site i's other modes.
            # HFF + h.c. gives the full hopping Σ_σ (c†_{σ,i} c_{σ,i+1} + h.c.).
            HFF = _compute_hff(ZFprev, Fnow, Anow)
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

        all_eigvals = np.concatenate([v.numpy() for v in D.values()])
        Eg[itN - 1] = float(np.min(all_eigvals))

        # Absorb V into Anow to form the left-isometry AK: axes (left, right_new, phys).
        # 'abg,bd->adg': A[a,b,g], V[b,d] → AK[a,d,g]
        AK = einsum('abg,bd->adg', Anow, V)
        mps.append(AK.clone())

        bond_index = V.indices[1]
        # Hprev is diagonal in the truncated eigenbasis; feeds directly into next Hnow.
        Hprev = diag(D, bond_index, itags=(f'R{itN-1:02d}', f'R{itN-1:02d}'))

        # Accumulate Z×F through AK for the next step's hopping JW string.
        ZFprev = _push_zf(Z_now, Fnow, AK)

        if verbose:
            dim_fn = (lambda idx: idx.num_states) if is_su2 else (lambda idx: idx.dim)
            print(f'#{itN:02d}/{N:02d} : NK={dim_fn(AK.indices[0])}/{dim_fn(Hnow.indices[1])}')

    Egs = Eg / np.arange(1, N + 1)

    if verbose:
        E_inf     = -4.0 * t / np.pi
        E_exact_N = exact_halffilling_energy_band(N, t)
        print(f'\nExact GS energy per site ({N} sites, half-filling): {E_exact_N / N:.6f}')
        print(f'Exact GS energy per site (N→∞, half-filling):      {E_inf:.6f}')
        print(f'Final iterative estimate:                            {Egs[-1]:.6f}')

    return Eg, Egs, mps
