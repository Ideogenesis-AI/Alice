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


"""Pytest fixtures for XTRG algorithm tests.

Shared fixtures provide:

- `spinless_fermion_L4`: tight-binding chain H = -t Σ(c†_{i+1} c_i + h.c.),
  L=4, t=1, U(1), OBC. Returns `(mpo, spc, exact_log_z_fn)` where
  `exact_log_z_fn(beta)` computes the exact log Z via single-particle energies.

- `spinful_fermion_L4`: U=0 Hubbard chain (two decoupled spin channels),
  L=4, t=1, U(1)×U(1) (treated as conductor symmetry). Returns
  `(mpo, spc, exact_log_z_fn)`.

Exact formulas
--------------
For spinless OBC free fermions with hopping -t, single-particle energies are:

    ε_k = -2t cos(kπ / (L+1)),  k = 1, …, L.

The grand-canonical partition function at chemical potential µ=0 is:

    log Z(β) = Σ_k log(1 + exp(-β ε_k)).

For the spinful (U=0 Hubbard) case, each spin channel is independent:

    log Z_spinful(β) = 2 × log Z_spinless(β).
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import pytest
from nicole import Index

from alice.network import MPO, build_hamiltonian, build_interaction


# ---------------------------------------------------------------------------
# Working-directory isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Run every test with cwd set to a fresh tmp_path."""
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Exact reference functions
# ---------------------------------------------------------------------------

def _spinless_single_particle_energies(L: int, t: float = 1.0):
    """Single-particle energy levels for spinless OBC free fermions."""
    return [-2.0 * t * math.cos(k * math.pi / (L + 1)) for k in range(1, L + 1)]


def _exact_log_z_spinless(L: int, t: float = 1.0) -> Callable[[float], float]:
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


def _exact_log_z_spinful(L: int, t: float = 1.0) -> Callable[[float], float]:
    """Return a callable `log_z(beta)` for a U=0 Hubbard chain.

    Each spin channel is independent, so log Z = 2 × log Z_spinless.

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
    log_z_spinless = _exact_log_z_spinless(L, t)

    def log_z(beta: float) -> float:
        return 2.0 * log_z_spinless(beta)

    return log_z


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def spinless_fermion_L4() -> Tuple[MPO, Index, Callable[[float], float]]:
    """Spinless free-fermion chain, L=4, t=1, OBC, U(1).

    Returns
    -------
    tuple
        `(mpo, spc, exact_log_z_fn)` where `exact_log_z_fn(beta)` returns
        the exact grand-canonical log Z at inverse temperature β.
    """
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': 4, 'bcx': 'OBC', 'n2x': True},
        'model': {
            'category': 'fermionic',
            'label': 'FreeFermion',
            'symmetry': 'U1',
            't': 1.0,
        },
    }
    interactions, spc, geo = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, geo.L, spc)
    exact_log_z_fn = _exact_log_z_spinless(geo.L, t=1.0)
    return mpo, spc, exact_log_z_fn


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
    exact_log_z_fn = _exact_log_z_spinful(geo.L, t=1.0)
    return mpo, spc, exact_log_z_fn
