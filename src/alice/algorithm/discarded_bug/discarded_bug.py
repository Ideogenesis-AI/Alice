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

The discarded-projector BUG is a rank-adaptive Basis-Update & Galerkin integrator
derived from the faithful Ceruti–Kusch–Lubich scheme (arXiv:2304.05660), but with
the basis growth driven by the *discarded* (orthogonal-complement) projectors and
*without* the augmented overlap matrices M, N. Concretely, against the faithful
two-site BUG it changes only the local bond update (see
:mod:`alice.algorithm.discarded_bug.candidate`):

- the discarded projector ``P⊥`` is applied to the K/L *generator* before the
  exponential (``project-before``), and
- the augmented frame is the direct sum ``[U0 | Qk]`` / ``[V0 ; Ql]`` (no overlap
  matrix), with the S-step projecting ``Θ0`` straight onto the augmented bases.

Everything else — the odd/even Trotter sweep, the AutoMPO bond Hamiltonians, the
Krylov ``expv`` substeps, and the Alice `MPS` plumbing — is shared with
:mod:`alice.algorithm.two_site_bug`, so `Options` and `Summary` are reused as-is.

Typical usage::

    from alice import build_interaction, init_mps
    from alice.algorithm import discarded_bug

    interactions, spc, geo = build_interaction(cfg)
    mps = init_mps(geo.L, spc, Op, config=[0, 1] * (geo.L // 2), target_qn=0)
    opts = discarded_bug.Options(dt=0.05, n_steps=20, order='strang', max_bond=64)
    summary = discarded_bug.run(mps, interactions, opts)
    print(summary.bond_dims)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from alice.network import MPS
from alice.network.interaction import Interaction

# Reuse the faithful driver's Options/Summary verbatim — the discarded variant has
# the same controls and the same output record.
from ..two_site_bug._kernel import with_expv_backend, with_time_prefactor
from ..two_site_bug.bond import build_bond_generators, kernel_gate, to_complex
from ..two_site_bug.two_site_bug import Options, Summary, _UNLIMITED_BOND
from .scheme import parity_sweep

logger = logging.getLogger(__name__)

__all__ = ['Options', 'Summary', 'run']


def run(mps: MPS, interactions: List[Interaction], opts: Optional[Options] = None) -> Summary:
    """Evolve an MPS under a nearest-neighbour Hamiltonian with the discarded-projector BUG.

    Builds the per-bond Hamiltonian terms once from the AutoMPO interaction list,
    then applies `opts.n_steps` odd/even Trotter steps of the discarded-projector
    K/L/S local update. The state is canonicalised to `center = 0` before the
    first step and returned with `center = 0`.

    Parameters
    ----------
    mps:
        Initial MPS state. Promoted to `complex128` and canonicalised in-place to
        `center = 0` first. Start from a low-rank state to exercise the
        rank-adaptive growth.
    interactions:
        Interaction list from `build_interaction`. Every active term must be a
        nearest-neighbour `Interaction2Site`.
    opts:
        Run options. Defaults to `Options()` if `None`.

    Returns
    -------
    Summary
        Evolved state, time/norm/bond-dimension history, and step count.

    Raises
    ------
    ValueError
        If `mps` has fewer than two sites.
    """
    if opts is None:
        opts = Options()
    if mps.L < 2:
        raise ValueError(f"discarded-projector BUG evolution requires at least 2 sites, got L={mps.L}")

    maxdim = opts.max_bond if opts.max_bond is not None else _UNLIMITED_BOND
    prefactor: complex = -1.0 if opts.imaginary_time else -1j

    for site in range(mps.L):
        mps[site] = to_complex(mps[site])
    mps.canonical(0)

    generators = build_bond_generators(interactions, mps.L)
    gates = [
        None if h is None else kernel_gate(h, mps[b].itags[2], mps[b + 1].itags[2])
        for b, h in enumerate(generators)
    ]

    def sweep(parity: str, tau: float):
        return parity_sweep(
            mps, gates, parity, tau, maxdim,
            opts.augment, opts.aug_krylov_depth, opts.trunc_thresh,
            opts.lanczos_tol, opts.lanczos_maxiter,
        )

    times: List[float] = []
    norms: List[float] = []
    max_bond_dims: List[int] = []
    aug_dims: List[int] = []
    disc_weights: List[float] = []

    n_active = sum(1 for h in generators if h is not None)
    logger.info("─" * 60)
    logger.info("Commencing: Discarded-Projector BUG Time Evolution".center(60))
    logger.info("─" * 60)
    logger.info("")
    logger.info("  order             : %s", opts.order)
    logger.info("  chain length      : %d", mps.L)
    logger.info("  active bonds      : %d / %d", n_active, mps.L - 1)
    logger.info("  time step         : %g", opts.dt)
    logger.info("  steps             : %d", opts.n_steps)
    logger.info("  evolution         : %s", "imaginary" if opts.imaginary_time else "real")
    logger.info("  max bond dim      : %s", opts.max_bond if opts.max_bond is not None else 'unlimited')
    logger.info("  augment           : %s", opts.augment)
    logger.info("")

    w = len(str(opts.n_steps))
    with with_time_prefactor(prefactor), with_expv_backend('native_hermitian_lanczos'):
        for step in range(opts.n_steps):
            if opts.order == 'strang':
                results = [
                    sweep('even', 0.5 * opts.dt),
                    sweep('odd', opts.dt),
                    sweep('even', 0.5 * opts.dt),
                ]
            else:
                results = [
                    sweep('even', opts.dt),
                    sweep('odd', opts.dt),
                ]
            augmented = max(aug for aug, _ in results)
            discarded = max(disc for _, disc in results)

            norm = mps.norm()
            if opts.normalize:
                mps.normalize()

            times.append((step + 1) * opts.dt)
            norms.append(norm)
            max_bond_dims.append(max(mps.bond_dims) if mps.bond_dims else 1)
            aug_dims.append(augmented)
            disc_weights.append(discarded)

            logger.info(
                "step %*d / %d: t = %g, norm = %.10f, kept bond = %d, augmented = %d, disc = %.2e",
                w, step + 1, opts.n_steps, times[-1], norm, max_bond_dims[-1], augmented, discarded,
            )

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
        disc_weights=disc_weights,
    )
