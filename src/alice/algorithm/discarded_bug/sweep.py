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


"""Recursive-bisection step of the rank-adaptive discarded-projector BUG integrator.

This is the MPS specialisation of the **Lubich tree-tensor-network BUG** (the
rank-adaptive Basis-Update & Galerkin integrator of Ceruti–Lubich–Walach). The
reference builds its tree by **recursive bisection** of the 1D modes (a balanced
binary tree whose leaves are the physical sites); the MPS realisation therefore
recursively bisects the chain and performs one **two-site** node update at each
bisection bond, using the **discarded** projector in each K-step and L-step. Two
modifications from the reference:

* the reference performs **single-site** node updates — here each node update is a
  **two-site** update of the bisection bond through the two-site effective Hamiltonian
  :func:`~alice.algorithm.dmrg.scheme_2s.matvec_2s`;
* the reference builds the **augmented overlap projectors** ``M = Û1† U0`` to
  transport the core — here we never form ``M``/``N`` and instead read the
  augmented frames directly off the evolved two-site block and obtain the augmented
  core by projecting that block onto them (the *discarded* projector).

The node update (:func:`~.candidate.block_local_update`)
----------------------------------------------------------
At each bisection bond the two-site block is evolved once under the two-site effective
Hamiltonian, ``Theta1 = exp(tau H_2site) Theta0``; the K-step and L-step grow the
left/right frames with the discarded projector — the direct sums of ``U0``/``V0`` with
the column/row space of ``Theta1`` (``qr([Theta1_left | U0])`` /
``qr([Theta1_right | V0])``); and the Galerkin core is the projection
``U_aug+ Theta1 V_aug+`` of the already-evolved block, SVD-truncated to set the rank.

The bisection step (:func:`global_step`)
----------------------------------------
A single step recursively bisects the chain (:func:`_bisect`): it updates the central
bisection bond, then recurses into the left and right half-chains, updating each
sub-centre and recursing again until every bond — every tree node — has had its
two-site node update. Because every bond is a node, the bond dimension grows along the
whole chain (the full light cone) as the wall melts, matching the bond growth of
forward two-site TDVP.

First order, no backward correction
    The bisection composes the node updates in a fixed (depth-first) order, so the step
    is first order in ``dt``; there is no TDVP-style backward (negative-time) substep
    and no overlap-matrix inverse — BUG is inverse-free by design. The validated
    property is the rank growth / light-cone spread. (A second-order symmetric
    composition is left to future work — a naive node-order-reversed Strang pass does
    not lift the order, because the per-node basis truncations are not a reversible
    flow.)

Sector-order re-gauge
    The augmenter's QR orders each grown bond's charge sectors canonically
    (ascending), which can differ from the chain's existing order; an *incremental*
    recanonicalise would leave some bonds inconsistent for some charge patterns (the
    bug that made an even-length chain's step collapse the state). After the recursion
    the centre is therefore reset with ``mps._center = None`` so
    :meth:`~alice.network.network.Network.canonical` performs a full whole-chain sweep
    that re-gauges **every** bond consistently.
"""

from __future__ import annotations

from alice.network import MPS, MPO

from ..dmrg.environ import (
    left_env_boundary,
    right_env_boundary,
    step_left_env,
    step_right_env,
)
from ._krylov import to_complex
from .candidate import block_local_update, bond_snapshot


def _update_bond(
    mps: MPS,
    mpo: MPO,
    b: int,
    tau: complex,
    *,
    maxdim: int,
    cutoff: float,
    lanczos_tol: float,
    lanczos_maxiter: int,
) -> int:
    """Apply one Lubich node update at bond ``(b, b+1)``: canonicalise there, build the
    MPO environments bracketing the two-site window, run the discarded
    :func:`~.candidate.block_local_update`, and write the two new cores back (centre at
    ``b + 1``). Returns the kept bond dimension."""
    L = mps.L
    mps.canonical(b, trunc=None)
    e_left = to_complex(left_env_boundary(mps, mpo))
    for k in range(b):
        e_left = step_left_env(e_left, mps[k], mpo[k])
    e_right = to_complex(right_env_boundary(mps, mpo))
    for k in range(L - 2, b, -1):
        e_right = step_right_env(e_right, mps[k + 1], mpo[k + 1])

    snap = bond_snapshot(mps[b], mps[b + 1], mps._bond_itag(b + 1))
    update = block_local_update(
        snap, mpo[b], mpo[b + 1], e_left, e_right, tau,
        maxdim=maxdim, cutoff=cutoff,
        lanczos_tol=lanczos_tol, lanczos_maxiter=lanczos_maxiter,
    )
    mps[b] = update.left_core
    mps[b + 1] = update.right_core
    mps._center = b + 1
    return update.kept


def _bisect(
    mps: MPS,
    mpo: MPO,
    tau: complex,
    lo: int,
    hi: int,
    *,
    maxdim: int,
    cutoff: float,
    lanczos_tol: float,
    lanczos_maxiter: int,
) -> int:
    """Recursive bisection of the sub-chain on sites ``[lo, hi]`` (its bonds are
    ``lo … hi-1``). Updates the bisection-bond node, then recurses into the left and
    right halves — the MPS realisation of the Lubich balanced-binary-tree ``Step`` (each
    bond is a tree node). Returns the maximum kept bond dimension in the subtree."""
    if hi - lo < 1:
        return 1
    mid = (lo + hi) // 2
    kept = _update_bond(
        mps, mpo, mid, tau,
        maxdim=maxdim, cutoff=cutoff,
        lanczos_tol=lanczos_tol, lanczos_maxiter=lanczos_maxiter,
    )
    kept_l = _bisect(
        mps, mpo, tau, lo, mid,
        maxdim=maxdim, cutoff=cutoff,
        lanczos_tol=lanczos_tol, lanczos_maxiter=lanczos_maxiter,
    )
    kept_r = _bisect(
        mps, mpo, tau, mid + 1, hi,
        maxdim=maxdim, cutoff=cutoff,
        lanczos_tol=lanczos_tol, lanczos_maxiter=lanczos_maxiter,
    )
    return max(kept, kept_l, kept_r)


def global_step(
    mps: MPS,
    mpo: MPO,
    tau: complex,
    *,
    maxdim: int,
    cutoff: float,
    lanczos_tol: float,
    lanczos_maxiter: int,
) -> int:
    """Advance ``mps`` by one discarded-BUG step via recursive bisection of the chain.

    The MPS realisation of the Lubich tree-tensor-network BUG ``Step`` on a balanced
    binary tree (the reference builds the tree by recursive bisection of the 1D modes).
    Canonicalises to ``center == 0`` (truncation-free), then recursively bisects the
    chain (:func:`_bisect`): at each bisection bond it applies one discarded two-site
    :func:`~.candidate.block_local_update` (the K-step and L-step grow that node's
    left/right frames with the **discarded** projector; the two-site Galerkin core
    update evolves the connecting tensor), then recurses into the two halves. Because
    every bond is a tree node, every bond's basis is updated — so the bond dimension
    grows along the whole chain (the full light cone) as the wall melts. After the
    recursion the centre is reset to ``None`` and the state is fully recanonicalised to
    ``center == 0`` so every bond's charge-sector order is consistent.

    The step is first order in ``dt`` (the bisection composes the node updates in a
    fixed order); there is no backward (negative-time) substep — BUG is inverse-free by
    design, and the validated property is the rank growth / light-cone spread.

    Parameters
    ----------
    mps:
        State to evolve in place. Promoted/canonicalised by the caller.
    mpo:
        Hamiltonian MPO of the same length.
    tau:
        Generator coefficient ``prefactor * dt`` (``-1j*dt`` real time, ``-dt``
        imaginary time).
    maxdim:
        Maximum bond dimension kept by each node's SVD.
    cutoff:
        Relative singular-value threshold of each node's SVD.
    lanczos_tol, lanczos_maxiter:
        Krylov termination tolerance and maximum dimension for every block evolution.

    Returns
    -------
    int
        The maximum kept bond dimension produced over the step.
    """
    mps.canonical(0, trunc=None)
    kept = _bisect(
        mps, mpo, tau, 0, mps.L - 1,
        maxdim=maxdim, cutoff=cutoff,
        lanczos_tol=lanczos_tol, lanczos_maxiter=lanczos_maxiter,
    )
    # Full re-gauge: the augmenter orders each grown bond's charge sectors canonically,
    # which can differ from the chain's existing order; resetting the center to None
    # forces canonical() to right-canonicalise the whole chain first, re-gauging every
    # bond consistently (an incremental sweep would leave some bonds inconsistent for
    # some charge patterns).
    mps._center = None
    mps.canonical(0)
    return kept
