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


"""Pytest fixtures for TgTRG algorithm tests.

The benchmarks are the same free-fermion chains XTRG is tested on, so the two
algorithms answer to the same analytic reference:

- `spinless_fermion_L4`: tight-binding chain `H = -t Σ (c†_{i+1} c_i + h.c.)`,
  L=4, t=1, U(1), OBC. Returns `(mpo, spc, exact_log_z_fn)`.
- `spinful_fermion_L4`: U=0 Hubbard chain (two decoupled spin channels),
  L=4, t=1, U(1)×U(1). Returns `(mpo, spc, exact_log_z_fn)`.
- `spinless_fermion_L2`: the smallest chain on which an effective Hamiltonian
  exists, for the algebraic unit tests.

Exact formulas
--------------
For spinless OBC free fermions with hopping `-t`, single-particle energies are

    ε_k = -2t cos(kπ / (L+1)),  k = 1, …, L,

and the grand-canonical partition function at chemical potential µ=0 is

    log Z(β) = Σ_k log(1 + exp(-β ε_k)).

For the spinful (U=0 Hubbard) case each spin channel is independent, so
`log Z_spinful = 2 log Z_spinless`.

Dense rendering
---------------
`flatten`, `unflatten`, and `dense` render an effective Hamiltonian as a plain
matrix by fixing an order on the surviving symmetry blocks. Only allowed
blocks appear, so the flattened space is the variational space and nothing
else — the same device the TDVP tests use, carried over to the rank-4 tensors
of a density matrix.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
import pytest
import torch
from nicole import Index

from alice.algorithm.tgtrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
)
from alice.network import MPO, build_hamiltonian, build_interaction
from alice.network.thermal import NormalMPO, thermal_mpo


# ---------------------------------------------------------------------------
# Exact reference functions
# ---------------------------------------------------------------------------

def _spinless_single_particle_energies(L: int, t: float = 1.0):
    """Single-particle energy levels for spinless OBC free fermions."""
    return [-2.0 * t * math.cos(k * math.pi / (L + 1)) for k in range(1, L + 1)]


def exact_log_z_spinless(L: int, t: float = 1.0) -> Callable[[float], float]:
    """Return a callable `log_z(beta)` for a spinless OBC free-fermion chain.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Hopping amplitude.

    Returns
    -------
    Callable[[float], float]
        Function mapping β to log Z(β).
    """
    eps = _spinless_single_particle_energies(L, t)

    def log_z(beta: float) -> float:
        return sum(math.log1p(math.exp(-beta * e)) for e in eps)

    return log_z


def exact_energy_spinless(L: int, t: float = 1.0) -> Callable[[float], float]:
    """Return a callable `u(beta)` giving the exact energy per site.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Hopping amplitude.

    Returns
    -------
    Callable[[float], float]
        Function mapping β to `Σ_k ε_k / (1 + e^{β ε_k}) / L`.
    """
    eps = _spinless_single_particle_energies(L, t)

    def energy(beta: float) -> float:
        return sum(e / (1.0 + math.exp(beta * e)) for e in eps) / L

    return energy


def exact_log_z_spinful(L: int, t: float = 1.0) -> Callable[[float], float]:
    """Return a callable `log_z(beta)` for a U=0 Hubbard chain.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Hopping amplitude.

    Returns
    -------
    Callable[[float], float]
        Function mapping β to log Z(β), twice the spinless value.
    """
    log_z_spinless = exact_log_z_spinless(L, t)

    def log_z(beta: float) -> float:
        return 2.0 * log_z_spinless(beta)

    return log_z


# ---------------------------------------------------------------------------
# Hamiltonian builders
# ---------------------------------------------------------------------------

def build_spinless(L: int, t: float = 1.0) -> Tuple[MPO, Index]:
    """Build the spinless free-fermion chain MPO and its physical space."""
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': L, 'bcx': 'OBC', 'n2x': True},
        'model': {
            'category': 'fermionic',
            'label': 'FreeFermion',
            'symmetry': 'U1',
            't': t,
        },
    }
    interactions, spc, geo = build_interaction(cfg)
    return build_hamiltonian(interactions, geo.L, spc), spc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def spinless_fermion_L2() -> Tuple[MPO, Index]:
    """Spinless free-fermion chain, L=2, t=1, OBC, U(1).

    Returns
    -------
    tuple
        `(mpo, spc)`.
    """
    return build_spinless(2)


@pytest.fixture(scope='session')
def spinless_fermion_L4() -> Tuple[MPO, Index, Callable[[float], float]]:
    """Spinless free-fermion chain, L=4, t=1, OBC, U(1).

    Returns
    -------
    tuple
        `(mpo, spc, exact_log_z_fn)` where `exact_log_z_fn(beta)` returns
        the exact grand-canonical log Z at inverse temperature β.
    """
    mpo, spc = build_spinless(4)
    return mpo, spc, exact_log_z_spinless(4, t=1.0)


@pytest.fixture(scope='session')
def spinful_fermion_L4() -> Tuple[MPO, Index, Callable[[float], float]]:
    """U=0 Hubbard (spinful free-fermion) chain, L=4, t=1, OBC.

    Returns
    -------
    tuple
        `(mpo, spc, exact_log_z_fn)` where `exact_log_z_fn(beta)` returns
        the exact grand-canonical log Z at inverse temperature β.
    """
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': 4, 'bcx': 'OBC', 'n2x': True},
        'model': {
            'category': 'conductor',
            'label': 'Hubbard',
            'symmetry': 'U1,U1',
            't': 1.0,
            'U': 0.0,
        },
    }
    interactions, spc, geo = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, geo.L, spc)
    return mpo, spc, exact_log_z_spinful(geo.L, t=1.0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def thermal_state(mpo: MPO, spc: Index, beta: float, order: int = 12) -> NormalMPO:
    """Build `ρ(β)` by Taylor expansion, right-canonical at `center == 0`.

    Parameters
    ----------
    mpo:
        Hamiltonian MPO.
    spc:
        Physical space index, for the identity term of the series.
    beta:
        Inverse temperature.
    order:
        Taylor truncation order.

    Returns
    -------
    NormalMPO
        Density matrix with `center == 0`.
    """
    rho = thermal_mpo(mpo, beta, order, spc)
    rho.canonical(0)
    return rho


def fresh_envs(rho: NormalMPO, mpo: MPO, **kwargs):
    """Build an `(env_left, env_right)` pair seeded for a forward sweep.

    Parameters
    ----------
    rho:
        Density matrix with `center == 0`.
    mpo:
        Hamiltonian MPO.
    **kwargs:
        Forwarded to both `Environment` constructors.

    Returns
    -------
    tuple
        `(env_left, env_right)` with `env_left[0]` and every right block set.
    """
    env_left = Environment(rho.L, **kwargs)
    env_right = Environment(rho.L, **kwargs)
    env_left[0] = left_env_boundary(rho, mpo)
    build_right_envs(rho, mpo, env_right)
    return env_left, env_right


def energy_reference(rho: NormalMPO, mpo: MPO) -> float:
    """Compute `Tr[ρ̂ H ρ̂]` independently of the environment machinery.

    `ρ̂` is the internal, unit-norm tensor train of `rho` — the object the
    effective Hamiltonians of the cooling sweep actually act on, with the
    magnitude left in `log_scale`. The value is assembled with `NormalMPO`
    products and `log_trace`, so it shares no code with `environ`.

    Parameters
    ----------
    rho:
        Density matrix; only its site tensors are used.
    mpo:
        Hamiltonian MPO.

    Returns
    -------
    float
        `Tr[ρ̂ H ρ̂]`.
    """
    unit = NormalMPO([t.clone() for t in rho], log_scale=0.0, bc=rho.bc,
                     center=rho.center)
    log_abs, sign = (unit @ NormalMPO.from_mpo(mpo) @ unit).log_trace()
    return sign * math.exp(log_abs)


def frobenius(rho_a: NormalMPO, rho_b: NormalMPO) -> float:
    """Compute `Tr[ρ_a† ρ_b]` over the internal (unit-norm) tensors.

    The scales are left out deliberately: every comparison in these tests is
    between tensors on the same footing, and folding `log_scale` back in would
    only reintroduce the overflow the representation exists to avoid.

    Parameters
    ----------
    rho_a:
        Bra density matrix.
    rho_b:
        Ket density matrix, with the same index structure as `rho_a`.

    Returns
    -------
    float
        The Frobenius product of the two internal tensor trains.
    """
    from nicole import einsum

    E = None
    for i in range(rho_a.L):
        E = (einsum('abrs,acrs->bc', rho_a[i].conj(), rho_b[i]) if E is None
             else einsum('bc,bdrs,cers->de', E, rho_a[i].conj(), rho_b[i]))
    if not E.data:
        return 0.0
    key, val = next(iter(E.data.items()))
    weight = 1.0 if E.intw is None else float(E.intw[key].weights[0, 0].real)
    return float(complex(val.item()).real) * weight


# ---------------------------------------------------------------------------
# Dense rendering of an effective Hamiltonian
# ---------------------------------------------------------------------------

def flatten(t):
    """Flatten the surviving blocks of `t` into a vector plus their key order."""
    keys = sorted(t.data.keys(), key=str)
    return np.concatenate([t.data[k].reshape(-1).numpy() for k in keys]), keys


def unflatten(vec, ref, keys):
    """Rebuild a tensor shaped like `ref` from a flat vector."""
    out = ref.clone()
    offset = 0
    for k in keys:
        n = ref.data[k].numel()
        out.data[k] = torch.tensor(
            vec[offset:offset + n].reshape(tuple(ref.data[k].shape)),
            dtype=ref.data[k].dtype,
        )
        offset += n
    return out


def dense(matvec_fn, ref):
    """Render `matvec_fn` as a dense matrix in the flattened block basis."""
    _, keys = flatten(ref)
    n = sum(ref.data[k].numel() for k in keys)
    basis = np.eye(n, dtype=np.float64)
    cols = [flatten(matvec_fn(unflatten(basis[j], ref, keys)))[0] for j in range(n)]
    return np.stack(cols, axis=1)
