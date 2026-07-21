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


"""Top-level discarded-projector BUG driver: options, summary, and entry point.

Discarded-projector basis-update-and-Galerkin (BUG) integrator on an `MPS`: a
rank-adaptive two-site time integrator derived from the Ceruti–Kusch–Lubich BUG
scheme, but with the basis growth driven by the **discarded** (orthogonal
complement) projectors and **without** forming the augmented overlap matrices, and
**without** a backward correction. This is the Alice port of the reference Julia
``discarded_bug_step!`` (``../../../../src/BUG/discarded_bug.jl``).

Like 2-site TDVP (and unlike a bare-gate TEBD BUG), the Galerkin core exponentiates
the full *effective Hamiltonian* with the left/right MPO environments, so this
integrator takes a Hamiltonian `MPO` (from `build_hamiltonian`) — exactly like
`alice.algorithm.dmrg` — and reuses the DMRG environment machinery and the 2-site
contraction. A step is a single global sweep (:func:`~.sweep.global_step`): form
`phi = H psi`, build augmented left/right isometries that keep `psi` exact and admit
only the discarded part `(I - U0 U0+) phi`, then integrate one Galerkin centre tensor
under the two-site effective Hamiltonian — so the bond dimension grows along the whole
chain (the light cone). There is no Trotter splitting and (by design, since BUG is
inverse-free) no backward substep — the step is exact at full bond dimension and second
order in `dt` (convergent under truncation).

Typical usage::

    from alice import build_interaction, build_hamiltonian, init_mps
    from alice.algorithm import discarded_bug

    interactions, spc, geo = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, geo.L, spc)
    mps = init_mps(geo.L, spc, Op, config=[0, 1] * (geo.L // 2), target_qn=0)
    opts = discarded_bug.Options(dt=0.02, n_steps=25, max_bond=64)
    summary = discarded_bug.run(mps, mpo, opts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from alice.network import MPS, MPO
from alice.network.network import Network

from ..interface import AlgorithmOptions, AlgorithmSummary
from ._krylov import to_complex
from .sweep import global_step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class Options(AlgorithmOptions):
    """Discarded-projector BUG run options.

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
        Relative singular-value threshold of the per-bond SVD truncation (the final
        centre-core truncation, the discarded-weight knob shared with TDVP).
    aug_cutoff:
        Optional separate threshold for *admitting* the discarded complement in the
        K/L augmentation sweeps. ``None`` (default) reuses ``cutoff`` (original
        behaviour). A looser value admits fewer new directions, capping the
        augmented bond growth — the global analogue of the two-site BUG
        ``kl_cutoff``.
    lanczos_tol:
        Termination tolerance of the local Krylov ``expv`` solves.
    lanczos_maxiter:
        Maximum Krylov dimension per local substep.
    imaginary_time:
        If ``True``, evolve with ``exp(-dt H)`` (imaginary time) instead of
        ``exp(-i dt H)``. Combined with ``normalize`` this cools toward the ground
        state.
    normalize:
        If ``True`` (default), renormalise the state after every step.
    """

    dt: float = 0.02
    n_steps: int = 10
    max_bond: Optional[int] = None
    cutoff: float = 1e-12
    aug_cutoff: Optional[float] = None
    lanczos_tol: float = 1e-14
    lanczos_maxiter: int = 40
    imaginary_time: bool = False
    normalize: bool = True
    solver: str = 'krylov'
    solver_substeps: int = 1

    def __post_init__(self) -> None:
        from ..two_site_bug._kernel.local_solvers import LOCAL_SOLVERS
        if self.solver not in LOCAL_SOLVERS:
            raise ValueError(
                f"unknown local solver {self.solver!r}; recognised values are: "
                f"{', '.join(LOCAL_SOLVERS)}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class Summary(AlgorithmSummary):
    """Discarded-projector BUG output.

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
    aug_dims:
        Maximum *proposed* augmented central-window bond dimension (``max`` of the
        K and L sides) before the final SVD truncation, after each step
        (length ``n_steps``). Comparing it with ``max_bond_dims`` shows how much
        rank the truncation discards.
    aug_k_dims, aug_l_dims:
        The K-side (``mid_u``) and L-side (``mid_v``) proposed augmented central
        bonds separately, after each step.
    """

    state: MPS
    n_steps: int = 0
    times: List[float] = field(default_factory=list)
    norms: List[float] = field(default_factory=list)
    bond_dims: List[int] = field(default_factory=list)
    max_bond_dims: List[int] = field(default_factory=list)
    aug_dims: List[int] = field(default_factory=list)
    aug_k_dims: List[int] = field(default_factory=list)
    aug_l_dims: List[int] = field(default_factory=list)
    disc_weights: List[float] = field(default_factory=list)

    def serialize(self) -> Dict:
        """Serialize the summary to a plain dict compatible with ``torch.save``."""
        return {
            'version': 1,
            'n_steps': self.n_steps,
            'times': self.times,
            'norms': self.norms,
            'bond_dims': self.bond_dims,
            'max_bond_dims': self.max_bond_dims,
            'aug_dims': self.aug_dims,
            'aug_k_dims': self.aug_k_dims,
            'aug_l_dims': self.aug_l_dims,
            'disc_weights': self.disc_weights,
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
            aug_dims=data.get('aug_dims', []),
            aug_k_dims=data.get('aug_k_dims', []),
            aug_l_dims=data.get('aug_l_dims', []),
            disc_weights=data.get('disc_weights', []),
        )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(mps: MPS, mpo: MPO, opts: Optional[Options] = None) -> Summary:
    """Evolve an MPS under a Hamiltonian MPO with the discarded-projector BUG.

    Performs ``opts.n_steps`` steps. Each step is a single global discarded-projector
    sweep (:func:`~.sweep.global_step`): the bond dimension grows along the whole chain
    as the wall melts, and the state is returned with ``center == 0``.

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
        If ``mps`` has fewer than two sites or ``mps`` and ``mpo`` differ in length.
    """
    if opts is None:
        opts = Options()
    if mps.L < 2:
        raise ValueError(f"discarded BUG evolution requires at least 2 sites, got L={mps.L}")
    if mps.L != mpo.L:
        raise ValueError(f"mps and mpo must have the same length, got {mps.L} and {mpo.L}")

    maxdim = opts.max_bond if opts.max_bond is not None else 1_000_000_000
    prefactor: complex = -1.0 if opts.imaginary_time else -1j

    # Promote both state and Hamiltonian to complex128 so every effective-H
    # contraction and local exponential shares the backend dtype.
    for site in range(mps.L):
        mps[site] = to_complex(mps[site])
    mpo = MPO([to_complex(mpo[b]) for b in range(mpo.L)])
    mps.canonical(0)

    times: List[float] = []
    norms: List[float] = []
    max_bond_dims: List[int] = []
    aug_dims: List[int] = []
    aug_k_dims: List[int] = []
    aug_l_dims: List[int] = []
    disc_weights: List[float] = []

    logger.info("─" * 60)
    logger.info("Commencing: Discarded-Projector BUG Time Evolution".center(60))
    logger.info("─" * 60)
    logger.info("")
    logger.info("  chain length      : %d", mps.L)
    logger.info("  time step         : %g", opts.dt)
    logger.info("  steps             : %d", opts.n_steps)
    logger.info("  local solver      : %s (substeps %d)", opts.solver, opts.solver_substeps)
    logger.info("  evolution         : %s", "imaginary" if opts.imaginary_time else "real")
    logger.info("  max bond dim      : %s", opts.max_bond if opts.max_bond is not None else 'unlimited')
    logger.info("")

    w = len(str(opts.n_steps))
    for step in range(opts.n_steps):
        # One global discarded-projector sweep per time step: form phi = H psi, keep
        # psi exact and admit only the discarded part of phi into the augmented bases,
        # then integrate one Galerkin centre tensor. The bond dimension grows along the
        # whole chain (the light cone) as the wall melts.
        kept, disc, aug_k, aug_l = global_step(mps, mpo, prefactor * opts.dt,
                                      maxdim=maxdim, cutoff=opts.cutoff, aug_cutoff=opts.aug_cutoff,
                                      lanczos_tol=opts.lanczos_tol, lanczos_maxiter=opts.lanczos_maxiter,
                                      solver=opts.solver, solver_substeps=opts.solver_substeps)

        norm = mps.norm()
        if opts.normalize:
            mps.normalize()

        times.append((step + 1) * opts.dt)
        norms.append(norm)
        max_bond_dims.append(kept)
        aug_k_dims.append(aug_k)
        aug_l_dims.append(aug_l)
        aug_dims.append(max(aug_k, aug_l))
        disc_weights.append(disc)

        logger.info("step %*d / %d: t = %g, norm = %.10f, kept bond = %d, aug(K,L) = (%d,%d), disc = %.2e",
                    w, step + 1, opts.n_steps, times[-1], norm, kept, aug_k, aug_l, disc)

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
        aug_dims=aug_dims,
        aug_k_dims=aug_k_dims,
        aug_l_dims=aug_l_dims,
        disc_weights=disc_weights,
    )
