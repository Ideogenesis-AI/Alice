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


"""Top-level 2-site TDVP driver: options, summary, and entry-point function.

Two-site Time-Dependent Variational Principle (TDVP) integrator on an `MPS`,
following the Haegeman et al. projector-splitting scheme (arXiv:1408.5056) with a
2-site update so the bond dimension can adapt. It is the Alice counterpart of the
reference Julia `tdvp2_step!` (`../../../../src/TDVP/tdvp2_sweep.jl`): a forward
half-sweep evolves each 2-site block forward by ``dt`` and the carried one-site
tensor backward by ``dt`` (inverse-free backward correction), a reverse half-sweep
mirrors it, and a symmetric step composes ``forward(dt/2)`` + ``reverse(dt/2)`` for
second-order accuracy.

Unlike the BUG integrators (which apply bare two-site gates), TDVP exponentiates
the full *effective Hamiltonian* with the left/right MPO environments, so it takes
a Hamiltonian `MPO` (from `build_hamiltonian`) — exactly like `alice.algorithm.dmrg`
— and reuses the DMRG environment machinery and 2-site/1-site contractions.

Typical usage::

    from alice import build_interaction, build_hamiltonian, init_mps
    from alice.algorithm import tdvp2

    interactions, spc, geo = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, geo.L, spc)
    mps = init_mps(geo.L, spc, Op, config=[0, 1] * (geo.L // 2), target_qn=0)
    opts = tdvp2.Options(dt=0.05, n_steps=20, max_bond=64)
    summary = tdvp2.run(mps, mpo, opts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from alice.network import MPS, MPO
from alice.network.network import Network

from ..interface import AlgorithmOptions, AlgorithmSummary
from ..dmrg.environ import (
    Environment,
    left_env_boundary,
    right_env_boundary,
    step_left_env,
    step_right_env,
)
from ._krylov import to_complex, with_time_prefactor
from .sweep import forward_sweep, reverse_sweep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class Options(AlgorithmOptions):
    """2-site TDVP run options.

    Parameters
    ----------
    dt:
        Time step. Real time (``exp(-i dt H)``) unless ``imaginary_time`` is set.
    n_steps:
        Number of time steps to perform.
    max_bond:
        Maximum bond dimension kept by the per-bond SVD truncation. ``None`` means
        no explicit cap (the bond grows up to the local capacity).
    cutoff:
        Singular-value threshold of the per-bond SVD truncation.
    lanczos_tol:
        Termination tolerance of the local Lanczos ``expv`` solves.
    lanczos_maxiter:
        Maximum Lanczos iterations per local substep.
    imaginary_time:
        If ``True``, evolve with ``exp(-dt H)`` (imaginary time) instead of
        ``exp(-i dt H)``. Combined with ``normalize`` this cools toward the
        ground state.
    normalize:
        If ``True`` (default), renormalise the state after every step.
    """

    dt: float = 0.05
    n_steps: int = 10
    max_bond: Optional[int] = None
    cutoff: float = 1e-12
    lanczos_tol: float = 1e-15
    lanczos_maxiter: int = 30
    imaginary_time: bool = False
    normalize: bool = True


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class Summary(AlgorithmSummary):
    """2-site TDVP output.

    Attributes
    ----------
    state:
        Evolved MPS after all steps (orthogonality center at site 0).
    n_steps:
        Number of steps performed.
    times:
        Cumulative evolution time after each step (length ``n_steps``).
    norms:
        State norm after each step *before* renormalisation (length ``n_steps``).
    bond_dims:
        Bond dimensions of ``state`` after the final step (length ``L - 1``).
    max_bond_dims:
        Maximum kept bond dimension after each step (length ``n_steps``).
    """

    state: MPS
    n_steps: int = 0
    times: List[float] = field(default_factory=list)
    norms: List[float] = field(default_factory=list)
    bond_dims: List[int] = field(default_factory=list)
    max_bond_dims: List[int] = field(default_factory=list)

    def serialize(self) -> Dict:
        """Serialize the summary to a plain dict compatible with ``torch.save``."""
        return {
            'version': 1,
            'n_steps': self.n_steps,
            'times': self.times,
            'norms': self.norms,
            'bond_dims': self.bond_dims,
            'max_bond_dims': self.max_bond_dims,
            'state': self.state.serialize(),
        }

    @classmethod
    def deserialize(cls, data: Dict, device: str = 'cpu') -> Summary:
        """Reconstruct a `Summary` from a dict produced by `serialize`."""
        version = data.get('version', 1)
        if version != 1:
            raise ValueError(f"Unsupported Summary serialization version: {version!r}")
        return cls(
            state=Network.deserialize(data['state'], device=device),
            n_steps=data['n_steps'],
            times=data['times'],
            norms=data['norms'],
            bond_dims=data['bond_dims'],
            max_bond_dims=data.get('max_bond_dims', []),
        )


# ---------------------------------------------------------------------------
# Half-sweep environment preparation
# ---------------------------------------------------------------------------

def _do_forward(mps: MPS, mpo: MPO, tau: float, maxdim, cutoff, lanczos_tol, lanczos_maxiter):
    """Right-canonicalise, build all right environments, run a forward half-sweep.

    The right environments are built here (rather than via the DMRG bulk builder)
    so the dim-1 boundary block can be promoted to ``complex128`` — for real-time
    evolution the state and MPO are complex, and the transfer contractions require
    all three tensors to share a dtype.
    """
    L = mps.L
    mps.canonical(0)
    env_left = Environment(L, fetch_lo=0, fetch_hi=L - 2)
    env_right = Environment(L, fetch_lo=1, fetch_hi=L - 1)
    env_left[0] = to_complex(left_env_boundary(mps, mpo))
    env_right[L - 1] = to_complex(right_env_boundary(mps, mpo))
    for i in range(L - 2, 0, -1):  # env_right[i] accumulates sites i+1 … L-1
        env_right[i] = step_right_env(env_right[i + 1], mps[i + 1], mpo[i + 1])
    forward_sweep(mps, mpo, env_left, env_right, tau,
                  maxdim=maxdim, cutoff=cutoff,
                  lanczos_tol=lanczos_tol, lanczos_maxiter=lanczos_maxiter)


def _do_reverse(mps: MPS, mpo: MPO, tau: float, maxdim, cutoff, lanczos_tol, lanczos_maxiter):
    """Left-canonicalise, build all left environments, run a reverse half-sweep."""
    L = mps.L
    mps.canonical(L - 1)
    env_left = Environment(L, fetch_lo=0, fetch_hi=L - 2)
    env_right = Environment(L, fetch_lo=1, fetch_hi=L - 1)
    env_left[0] = to_complex(left_env_boundary(mps, mpo))
    env_right[L - 1] = to_complex(right_env_boundary(mps, mpo))
    for i in range(L - 1):  # env_left[i+1] accumulates sites 0 … i
        env_left[i + 1] = step_left_env(env_left[i], mps[i], mpo[i])
    reverse_sweep(mps, mpo, env_left, env_right, tau,
                  maxdim=maxdim, cutoff=cutoff,
                  lanczos_tol=lanczos_tol, lanczos_maxiter=lanczos_maxiter)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(mps: MPS, mpo: MPO, opts: Optional[Options] = None) -> Summary:
    """Evolve an MPS under a Hamiltonian MPO with the 2-site TDVP integrator.

    Performs ``opts.n_steps`` symmetric (Strang) steps: each step is a forward
    half-sweep of duration ``dt/2`` followed by a reverse half-sweep of ``dt/2``.
    The state is returned with ``center == 0``.

    Parameters
    ----------
    mps:
        Initial MPS state. Promoted to ``complex128`` and canonicalised in-place.
    mpo:
        Hamiltonian MPO of the same length as ``mps``.
    opts:
        Run options. Defaults to ``Options()`` if ``None``.

    Returns
    -------
    Summary
        Evolved state and time/norm/bond-dimension history.

    Raises
    ------
    ValueError
        If ``mps`` has fewer than two sites, or ``mps`` and ``mpo`` differ in length.
    """
    if opts is None:
        opts = Options()
    if mps.L < 2:
        raise ValueError(f"2-site TDVP evolution requires at least 2 sites, got L={mps.L}")
    if mps.L != mpo.L:
        raise ValueError(f"mps and mpo must have the same length, got {mps.L} and {mpo.L}")

    maxdim = opts.max_bond
    prefactor: complex = -1.0 if opts.imaginary_time else -1j

    # Promote both state and Hamiltonian to complex128 so every effective-H
    # contraction and local exponential shares the backend dtype (the DMRG path
    # keeps these real; real-time TDVP needs the complex exponential).
    for site in range(mps.L):
        mps[site] = to_complex(mps[site])
    mpo = MPO([to_complex(mpo[b]) for b in range(mpo.L)])
    mps.canonical(0)

    half = 0.5 * opts.dt
    times: List[float] = []
    norms: List[float] = []
    max_bond_dims: List[int] = []

    logger.info("─" * 60)
    logger.info("Commencing: Two-Site TDVP Time Evolution".center(60))
    logger.info("─" * 60)
    logger.info("")
    logger.info("  chain length      : %d", mps.L)
    logger.info("  time step         : %g", opts.dt)
    logger.info("  steps             : %d", opts.n_steps)
    logger.info("  evolution         : %s", "imaginary" if opts.imaginary_time else "real")
    logger.info("  max bond dim      : %s", opts.max_bond if opts.max_bond is not None else 'unlimited')
    logger.info("")

    w = len(str(opts.n_steps))
    with with_time_prefactor(prefactor):
        for step in range(opts.n_steps):
            # Symmetric Strang step: forward(dt/2) then reverse(dt/2).
            _do_forward(mps, mpo, half, maxdim, opts.cutoff, opts.lanczos_tol, opts.lanczos_maxiter)
            _do_reverse(mps, mpo, half, maxdim, opts.cutoff, opts.lanczos_tol, opts.lanczos_maxiter)

            norm = mps.norm()
            if opts.normalize:
                mps.normalize()

            times.append((step + 1) * opts.dt)
            norms.append(norm)
            max_bond_dims.append(max(mps.bond_dims) if mps.bond_dims else 1)

            logger.info("step %*d / %d: t = %g, norm = %.10f, kept bond = %d",
                        w, step + 1, opts.n_steps, times[-1], norm, max_bond_dims[-1])

    if mps.center != 0:
        mps.canonical(0)

    logger.info("")

    return Summary(
        state=mps,
        n_steps=opts.n_steps,
        times=times,
        norms=norms,
        bond_dims=list(mps.bond_dims),
        max_bond_dims=max_bond_dims,
    )
