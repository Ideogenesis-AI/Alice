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


"""Update orchestration for VUMPS over one unit cell.

A VUMPS iteration solves every position of the unit cell against the same
set of environments, so the positions are independent of each other and
the order they are visited in does not matter. `cell_update` does one such
iteration: `N` center-tensor eigenproblems, `N` bond eigenproblems, then
the gauge step that turns the two sets of answers back into a unit cell.

Environments
------------
An infinite chain has no boundary, so the environments are the fixed
points solved in `environ`, not accumulated boundary blocks.
`cell_environments` solves the two on bond 0 and absorbs sites to reach
the rest, through `AL` going right and `AR` going left.

They go stale the moment the state changes, so `run` re-solves them once
per iteration. That is the dominant cost of the algorithm, around half the
running time.

Installing the result
---------------------
`AL`, `AR` and `C` go straight into the state through
`iMPS.set_canonical`, which takes all three as given. Deriving them from
the updated tensors instead would require the transfer operator of a state
caught mid-update to have a single dominant eigenvalue, which it need not.
The three are mutually consistent only at the fixed point; their mismatch
is the gradient this module reports.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from nicole import Tensor

from alice.network import iMPO, iMPS

from .environ import left_environment, right_environment, step_left_env, step_right_env
from .scheme_1s import gauge_tensors, optimize_bond, optimize_center

if TYPE_CHECKING:
    from .idmrg import Options

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environments over the unit cell
# ---------------------------------------------------------------------------

def cell_environments(
    imps: iMPS,
    impo: iMPO,
    *,
    tol: float = 1e-12,
    maxiter: int = 200,
) -> Tuple[List[Tensor], List[Tensor], float]:
    """Solve the boundary fixed points and spread them across the unit cell.

    `imps` must be gauge-fixed. The two fixed points live on bond 0; the
    remaining blocks follow from them by absorbing one site at a time, the
    left ones through `AL` and the right ones through `AR`.

    Parameters
    ----------
    imps:
        Gauge-fixed infinite MPS.
    impo:
        Hamiltonian iMPO sharing the unit cell of `imps`.
    tol:
        Convergence tolerance forwarded to the fixed-point solver.
    maxiter:
        Maximum number of unit cells absorbed by the fixed-point solver.

    Returns
    -------
    env_left:
        Block `i` sits on the bond to the left of site `i` and holds
        everything further left. Length `N`.
    env_right:
        Block `i` sits on the bond to the right of site `i` and holds
        everything further right. Length `N`.
    energy:
        Energy of one unit cell, from the left fixed point.
    """
    N = imps.N
    left_isos = imps.left_isometries()
    right_isos = imps.right_isometries()

    env_left: List[Optional[Tensor]] = [None] * N
    env_right: List[Optional[Tensor]] = [None] * N

    env_left[0], energy = left_environment(imps, impo, tol=tol, maxiter=maxiter)
    for i in range(N - 1):
        env_left[i + 1] = step_left_env(env_left[i], left_isos[i], impo[i])

    env_right[N - 1], _ = right_environment(imps, impo, tol=tol, maxiter=maxiter)
    for i in range(N - 2, -1, -1):
        env_right[i] = step_right_env(env_right[i + 1], right_isos[i + 1], impo[i + 1])

    return env_left, env_right, energy


# ---------------------------------------------------------------------------
# One VUMPS iteration
# ---------------------------------------------------------------------------

def cell_update(
    imps: iMPS,
    impo: iMPO,
    env_left: List[Tensor],
    env_right: List[Tensor],
    opts: Options,
) -> float:
    """Run one VUMPS iteration over the whole unit cell, in place.

    `imps` must be gauge-fixed on entry and is left gauge-fixed on return,
    carrying the updated `AL`, `AR` and `C`.

    Parameters
    ----------
    imps:
        Gauge-fixed infinite MPS, updated in place.
    impo:
        Hamiltonian iMPO sharing the unit cell.
    env_left:
        Left environment blocks from `cell_environments`.
    env_right:
        Right environment blocks from `cell_environments`.
    opts:
        Run options (scheme, Davidson parameters).

    Returns
    -------
    float
        Largest `|AC - AL C|` over the cell: the VUMPS gradient norm, and
        the quantity `run` converges on. It measures how far the state is
        from being a stationary point of the energy, which is a sharper
        signal than the change in energy between iterations.
    """
    if opts.scheme != '1s':
        raise NotImplementedError(f"cell_update: unknown scheme {opts.scheme!r}")

    N = imps.N
    davidson_opts = {
        'max_iter': opts.davidson_max_iter,
        'tol': opts.davidson_tol,
        'max_subspace': opts.davidson_max_subspace,
    }
    w = len(str(N - 1))

    # Everything is read from the current gauge before anything is written,
    # since every position is solved against the same environments.
    centers = [imps.center_tensor(i) for i in range(N)]
    bonds = [imps.bond_matrix(k) for k in range(N)]

    for i in range(N):
        energy, centers[i], davidson_error = optimize_center(
            centers[i], impo[i], env_left[i], env_right[i], davidson_opts,
        )
        logger.debug(
            "  site %*d: E = %+.12g, davidson err = %.4e",
            w, i, energy, davidson_error,
        )

    for k in range(N):
        # The right environment on bond k belongs to the site to its left,
        # which is site N-1 when k is 0.
        energy, bonds[k], davidson_error = optimize_bond(
            bonds[k], env_left[k], env_right[(k - 1) % N], davidson_opts,
        )
        logger.debug(
            "  bond %*d: E = %+.12g, davidson err = %.4e",
            w, k, energy, davidson_error,
        )

    left_isos: List[Tensor] = []
    right_isos: List[Tensor] = []
    gradient = 0.0
    for i in range(N):
        left_iso, right_iso, error = gauge_tensors(
            centers[i], bonds[i], bonds[(i + 1) % N],
        )
        left_isos.append(left_iso)
        right_isos.append(right_iso)
        gradient = max(gradient, error)

    imps.set_canonical(left_isos, right_isos, bonds)
    return gradient
