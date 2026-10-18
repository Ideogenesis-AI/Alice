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


"""Exact references for free-fermion real-time dynamics.

Shared by the TDVP examples in this directory, spinless and spinful alike.
Everything here is closed form: the free-fermion chain is quadratic, so a
Slater-determinant initial state stays a Slater determinant and the full
many-body evolution is carried by the `L x L` single-particle propagator.
That makes these functions a reference the tensor-network code cannot
accidentally agree with.

The spinful examples reuse all of it. At `U = 0` the Hubbard chain is two
decoupled spin channels, each identical to the spinless chain, so a
doubly-occupied/empty initial state gives exactly twice the spinless
occupations and twice the spinless entropy. That turns the functions here
into an analytic reference for the SU(2) code path as well.

For `H = -t Σ_j (c†_j c_{j+1} + h.c.)` the single-particle Hamiltonian is the
hopping matrix `h` with `h[j, j+1] = h[j+1, j] = -t`, and `U(t) = exp(-i h t)`.

Two kinds of initial state are covered:

- A **single particle** in an arbitrary superposition `ψ_j`, which evolves as
  `ψ(t) = U(t) ψ(0)` with occupations `n_j(t) = |ψ_j(t)|²`. Used by the
  wavepacket example, where the phases of `ψ` matter and so the sign in
  `exp(-i H t)` is observable.
- A **product state** with sites in `occ` filled, which is the Slater
  determinant whose equal-time correlation matrix is `C(0) = diag(occ)`.
  Used by the domain-wall example.

The many-body observables follow from the correlation matrix
`M = U diag(occ) U†`, whose diagonal gives the occupations and whose
subsystem blocks give the entanglement entropy.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from nicole import Direction, Tensor, einsum
from nicole.decomp import svd
from nicole.index import Index, Sector

from alice import MPS, build_hamiltonian, observe
from alice.network.interaction import Interaction1Site


# ---------------------------------------------------------------------------
# Single-particle machinery
# ---------------------------------------------------------------------------

def hopping_matrix(L: int, t: float = 1.0) -> np.ndarray:
    """Build the `L x L` single-particle hopping matrix for an open chain.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.

    Returns
    -------
    numpy.ndarray
        Real symmetric matrix with `-t` on both off-diagonals.
    """
    h = np.zeros((L, L), dtype=float)
    idx = np.arange(L - 1)
    h[idx, idx + 1] = -t
    h[idx + 1, idx] = -t
    return h


def propagator(L: int, t: float, time: float) -> np.ndarray:
    """Compute `exp(-i h t)` by diagonalizing the hopping matrix.

    `h` is real symmetric, so `eigh` is exact up to round-off and no matrix
    exponential routine is involved — this stays an independent reference.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Hopping amplitude.
    time:
        Evolution time.

    Returns
    -------
    numpy.ndarray
        Complex `L x L` unitary single-particle propagator.
    """
    eps, modes = np.linalg.eigh(hopping_matrix(L, t))
    return (modes * np.exp(-1j * eps * time)) @ modes.conj().T


def gaussian_wavepacket(L: int, x0: float, k0: float, sigma: float) -> np.ndarray:
    """Build a normalized Gaussian wavepacket amplitude on the chain.

    Parameters
    ----------
    L:
        Chain length.
    x0:
        Center of the packet, in sites.
    k0:
        Mean momentum. The group velocity is `2 t sin(k0)`, so `k0 = ±π/2`
        moves fastest and `k0 = 0` does not move at all.
    sigma:
        Width of the packet, in sites.

    Returns
    -------
    numpy.ndarray
        Complex amplitudes of unit norm, length `L`.
    """
    j = np.arange(L)
    psi = np.exp(-((j - x0) ** 2) / (2.0 * sigma ** 2)) * np.exp(1j * k0 * j)
    return psi / np.linalg.norm(psi)


def exact_single_particle(psi0: np.ndarray, t: float, time: float) -> np.ndarray:
    """Evolve a single-particle amplitude and return the site occupations.

    Parameters
    ----------
    psi0:
        Initial amplitudes, length `L`.
    t:
        Hopping amplitude.
    time:
        Evolution time.

    Returns
    -------
    numpy.ndarray
        Occupations `n_j(t) = |ψ_j(t)|²`, length `L`.
    """
    return np.abs(propagator(len(psi0), t, time) @ psi0) ** 2


# ---------------------------------------------------------------------------
# Many-body machinery for Slater determinants
# ---------------------------------------------------------------------------

def correlation_matrix(occ: Sequence[float], t: float, time: float) -> np.ndarray:
    """Compute the equal-time correlation matrix of an evolved product state.

    Starting from the product state with occupations `occ`, the correlation
    matrix `C_ij(t) = ⟨c†_i c_j⟩` is the transpose of
    `M(t) = U(t) diag(occ) U(t)†`. `M` is returned, since its diagonal is
    already the occupations and its subsystem blocks already have the right
    spectrum for the entanglement entropy.

    Parameters
    ----------
    occ:
        Initial site occupations, each `0` or `1`.
    t:
        Hopping amplitude.
    time:
        Evolution time.

    Returns
    -------
    numpy.ndarray
        Hermitian `L x L` matrix `M(t)`.
    """
    U = propagator(len(occ), t, time)
    return (U * np.asarray(occ, dtype=float)) @ U.conj().T


def exact_occupations(occ: Sequence[float], t: float, time: float) -> np.ndarray:
    """Return the exact site occupations of an evolved product state.

    Parameters
    ----------
    occ:
        Initial site occupations, each `0` or `1`.
    t:
        Hopping amplitude.
    time:
        Evolution time.

    Returns
    -------
    numpy.ndarray
        Occupations `n_j(t)`, length `L`.
    """
    return np.real(np.diag(correlation_matrix(occ, t, time)))


def exact_entropy(M: np.ndarray, ell: int) -> float:
    """Compute the exact entanglement entropy of sites `0 … ell-1`.

    For a Slater determinant the reduced density matrix is Gaussian, so the
    entropy follows from the eigenvalues `ν` of the correlation matrix
    restricted to the subsystem:

        S = -Σ [ν log ν + (1-ν) log(1-ν)]

    Parameters
    ----------
    M:
        Full correlation matrix from `correlation_matrix`.
    ell:
        Number of sites in the left subsystem, `1 ≤ ell ≤ L-1`.

    Returns
    -------
    float
        Von Neumann entropy in nats.
    """
    nu = np.linalg.eigvalsh(M[:ell, :ell]).clip(1e-14, 1.0 - 1e-14)
    return float(-np.sum(nu * np.log(nu) + (1.0 - nu) * np.log1p(-nu)))


# ---------------------------------------------------------------------------
# MPS-side measurements
# ---------------------------------------------------------------------------

def onsite_mpos(L: int, spc, ops, key: str = 'N4', name: str = 'N') -> list:
    """Build one MPO per site measuring a given on-site operator.

    Parameters
    ----------
    L:
        Chain length.
    spc:
        Physical space passed to `build_hamiltonian` (from `build_interaction`).
    ops:
        Operator dictionary from `build_fermionic` or `build_conductor`.
    key:
        Key of the 4th-order on-site template in `ops`. `'N4'` is the particle
        number, `'NN4'` the double occupancy of a spinful site.
    name:
        Label recorded on the interaction.

    Returns
    -------
    list
        `L` MPOs, where element `j` measures the operator on site `j`.
    """
    return [
        build_hamiltonian(
            [Interaction1Site(cpl=1.0, label=[name, 'onsite'], site=j,
                              tnsr=ops[key].clone())],
            L, spc,
        )
        for j in range(L)
    ]


def occupation_mpos(L: int, spc, ops) -> list:
    """Build one MPO per site measuring that site's particle number.

    For a spinful space this is the total `n_up + n_dn`, since `ops['N4']` is
    built that way by `build_conductor`.

    Parameters
    ----------
    L:
        Chain length.
    spc:
        Physical space passed to `build_hamiltonian` (from `build_interaction`).
    ops:
        Operator dictionary from `build_fermionic` or `build_conductor`.

    Returns
    -------
    list
        `L` MPOs, where element `j` measures `n_j`.
    """
    return onsite_mpos(L, spc, ops, key='N4', name='N')


def mps_occupations(mps: MPS, n_obs: list) -> np.ndarray:
    """Measure every site occupation of an MPS.

    Parameters
    ----------
    mps:
        State to measure.
    n_obs:
        Occupation MPOs from `occupation_mpos`.

    Returns
    -------
    numpy.ndarray
        Occupations, length `L`.
    """
    return np.array([observe(mps, n) for n in n_obs])


def mps_overlap(a: MPS, b: MPS) -> complex:
    """Compute `⟨a|b⟩` by a left-to-right transfer-matrix sweep.

    Works for non-Abelian groups too: the intertwiners ride along inside the
    contractions, and the single remaining block carries a weight that has to
    be folded in when the scalar is read out.

    Parameters
    ----------
    a:
        Bra MPS, conjugated here.
    b:
        Ket MPS, with the same index structure as `a`.

    Returns
    -------
    complex
        The overlap, not normalized by either state's norm.
    """
    E = None
    for i in range(a.L):
        E = (einsum('abr,acr->bc', a[i].conj(), b[i]) if E is None
             else einsum('bc,bdr,cer->de', E, a[i].conj(), b[i]))
    if not E.data:
        return 0.0 + 0.0j
    key, val = next(iter(E.data.items()))
    weight = 1.0 if E.intw is None else complex(E.intw[key].weights[0, 0].item())
    return complex(val.item()) * weight


def mps_entropy(mps: MPS, ell: int) -> float:
    """Compute the entanglement entropy of an MPS across the bond before `ell`.

    Moves the orthogonality center to site `ell`, which makes everything to
    its left left-isometric and everything to its right right-isometric, so a
    single SVD of that site tensor separating its left bond yields the Schmidt
    spectrum of the cut. Mutates the center of `mps`, nothing else.

    Under a non-Abelian group each singular value describes a whole *multiplet*
    rather than one Schmidt state, so it stands for `g = irrep_dim(q)`
    degenerate states of equal weight. Writing `w_a` for the total weight of
    multiplet `a`, every one of its `g_a` states carries probability
    `w_a / g_a` and the entropy is

        S = -Σ_a w_a log(w_a / g_a)

    which reduces to the usual `-Σ p log p` whenever every `g_a` is 1. Leaving
    the degeneracies out does not merely shift the answer, it loses the
    `Σ w_a log g_a` term entirely: for the SU(2) Hubbard chain that is a
    factor approaching two.

    Parameters
    ----------
    mps:
        State to measure.
    ell:
        Number of sites in the left subsystem, `1 ≤ ell ≤ L-1`.

    Returns
    -------
    float
        Von Neumann entropy in nats, counting degenerate states individually.
    """
    mps.canonical(ell)
    _, blocks, _ = svd(mps[ell], axis=0)
    group = mps[ell].indices[0].group

    weights, degeneracies = [], []
    for key, block in blocks.items():
        sigma = np.abs(block.detach().cpu().numpy().ravel())
        # Singular-value blocks are keyed by a (charge, charge) pair; both
        # entries are the same sector, since S is diagonal.
        g = float(group.irrep_dim(key[0]))
        weights.append(g * sigma ** 2)
        degeneracies.append(np.full(sigma.shape, g))

    w = np.concatenate(weights)
    g = np.concatenate(degeneracies)
    keep = w > 1e-16
    w, g = w[keep], g[keep]
    w = w / w.sum()
    return float(-np.sum(w * np.log(w / g)))


# ---------------------------------------------------------------------------
# Single-particle MPS construction
# ---------------------------------------------------------------------------

def wavepacket_mps(psi: np.ndarray, spc, ops) -> MPS:
    """Build the MPS of a one-particle superposition `Σ_j ψ_j c†_j |vac⟩`.

    `init_mps` only makes product states, and a wavepacket is not one: it
    needs bond dimension 2, carrying the two possibilities "the particle is
    still ahead" and "the particle is already behind". The site tensor is

        (ahead → ahead,   empty)    = 1
        (ahead → behind,  occupied) = ψ_j
        (behind → behind, empty)    = 1

    With a single particle present no Jordan-Wigner sign arises, so the MPS
    amplitude of the configuration with one particle at `j` is exactly `ψ_j`.

    The U(1) charge of a bond in Alice's convention accumulates `+1` per
    occupied site and `-1` per empty one, so the two sectors of the bond
    before site `i` sit at `q_vac - i` ("ahead") and `q_vac - i + 2`
    ("behind") — position-dependent, hence rebuilt per site.

    Parameters
    ----------
    psi:
        Complex single-particle amplitudes, length `L`. Normalized here.
    spc:
        Physical Index from `build_fermionic`.
    ops:
        Operator dictionary from `build_fermionic`.

    Returns
    -------
    MPS
        Complex128 MPS with `center == 0`, normalized, bond dimension 2.
    """
    L = len(psi)
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    group = spc.group
    q_vac = ops['vac'].sectors[0].charge

    tensors = []
    for i in range(L):
        left_q = [q_vac - i] if i == 0 else [q_vac - i, q_vac - i + 2]
        right_q = ([q_vac - (i + 1) + 2] if i == L - 1
                   else [q_vac - (i + 1), q_vac - (i + 1) + 2])
        l_idx = Index(direction=Direction.IN, group=group,
                      sectors=tuple(Sector(charge=q, dim=1) for q in left_q))
        r_idx = Index(direction=Direction.OUT, group=group,
                      sectors=tuple(Sector(charge=q, dim=1) for q in right_q))
        tnsr = Tensor.zeros(
            [l_idx, r_idx, spc],
            dtype=torch.complex128,
            itags=[f'A{i:02d}', f'A{i + 1:02d}', f's{i:02d}'],
        )
        # Blocks are keyed (q_left, q_right, q_phys); the occupied physical
        # sector carries charge +1 and is the only one that places a particle.
        for key, block in tnsr.data.items():
            block[0, 0, 0] = psi[i] if key[2] == 1 else 1.0
        tensors.append(tnsr)

    mps = MPS(tensors, center=None)
    mps.canonical(0)
    mps.normalize()
    return mps


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

_RAMP = ' .:-=+*#%@'


def density_row(values: np.ndarray, vmax: float = 1.0) -> str:
    """Render a density profile as a one-line shaded bar.

    Parameters
    ----------
    values:
        Non-negative values, one per site.
    vmax:
        Value mapped to the densest glyph.

    Returns
    -------
    str
        One character per site, from `' '` (empty) to `'@'` (full).
    """
    scaled = np.clip(np.asarray(values) / vmax, 0.0, 1.0)
    return ''.join(_RAMP[round(v * (len(_RAMP) - 1))] for v in scaled)


def error_table(
    label: str,
    numeric: np.ndarray,
    exact: np.ndarray,
    sites: Sequence[int],
) -> Tuple[str, float]:
    """Format a site-by-site comparison against an exact reference.

    Parameters
    ----------
    label:
        Column heading for the measured quantity.
    numeric:
        Values measured from the MPS.
    exact:
        Exact reference values.
    sites:
        Sites to tabulate.

    Returns
    -------
    str
        Multi-line table.
    float
        Maximum absolute error over *all* entries, not just the tabulated ones.
    """
    lines = [f"{'site':>5}  {label:>14}  {'exact':>14}  {'error':>11}",
             '-' * 50]
    for j in sites:
        lines.append(f"{j:>5}  {numeric[j]:>14.9f}  {exact[j]:>14.9f}  "
                     f"{numeric[j] - exact[j]:>+11.2e}")
    return '\n'.join(lines), float(np.max(np.abs(numeric - exact)))
