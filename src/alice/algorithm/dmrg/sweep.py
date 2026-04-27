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


"""Half-sweep orchestration for 1-site DMRG.

This module is a thin coordination layer. Each half-sweep is a sequential
loop over sites that delegates all local computation to `optimize_site` from
`scheme_1s`. The sweep itself handles only:

- deciding the order of sites to visit,
- moving the orthogonality center via `mps.canonical`,
- routing environment updates through the `Environment` instances.

This separation keeps `optimize_site` a pure function (no MPS mutation), which
is the natural unit of work for future parallel or distributed execution. When
parallelism is introduced it lives here, without touching any other module.

Center movement
---------------
`mps.canonical(target, trunc=trunc)` is safe to call here because the MPS
`center` attribute is always set before a sweep begins (the initial
`mps.canonical(0)` call in `dmrg()` establishes `center = 0`). With
`center` set, `canonical` immediately takes the direct one-step path
(`_left_canon_site` or `_right_canon_site`) without performing a full sweep.
"""

from __future__ import annotations

import logging
from typing import Optional

from alice.network import MPS, MPO

from .environ import Environment, step_left_env, step_right_env
from .scheme_1s import optimize_site

_log = logging.getLogger(__name__)


def right_sweep(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    davidson_opts: dict,
) -> float:
    """Perform a left-to-right (right) half-sweep of 1-site DMRG.

    Visits sites from the current orthogonality center to `L-1`, optimising
    each site tensor with Davidson, moving the center one step to the right,
    and updating the left environment for the next site. The rightmost site
    is optimised last without moving the center further.

    After this call:
    - `mps.center == mps.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-1` (left environments built
      from the canonicalised tensors).

    Parameters
    ----------
    mps:
        MPS to optimise in-place. Must have `center` set.
    mpo:
        Hamiltonian MPO.
    env_left:
        Left environment blocks. `env_left[mps.center]` must be initialised.
    env_right:
        Right environment blocks. All slots must be initialised before the
        first call (populated by `build_right_envs`).
    trunc:
        Truncation options forwarded to `mps.canonical` at each step.
        Pass `None` to disable truncation.
    davidson_opts:
        Keyword arguments forwarded to the Davidson solver (`max_iter`,
        `tol`, `max_subspace`).

    Returns
    -------
    float
        Variational energy at the last optimised site (rightmost site).
    """
    L = mps.L
    energy = 0.0
    _log.info("right sweep: sites 0 → %d", L - 1)

    for i in range(mps.center, L - 1):
        # Optimise the site tensor at position i.
        energy, M_opt = optimize_site(
            mps[i], env_left[i], mpo[i], env_right[i], davidson_opts
        )
        mps[i] = M_opt
        _log.debug("  site %*d / %d  E = %+.12g", len(str(L - 1)), i, L - 1, energy)

        # Move the orthogonality center one step to the right.
        # Because mps.center == i (set), canonical takes the direct one-step path.
        mps.canonical(i + 1, trunc=trunc)

        # Build the left environment for site i+1 from the now-canonicalised tensor.
        env_left[i + 1] = step_left_env(env_left[i], mps[i], mpo[i])

    # Optimise the rightmost site without moving the center further.
    energy, mps[L - 1] = optimize_site(
        mps[L - 1], env_left[L - 1], mpo[L - 1], env_right[L - 1], davidson_opts
    )
    _log.debug("  site %*d / %d  E = %+.12g", len(str(L - 1)), L - 1, L - 1, energy)
    _log.info("right sweep done  E = %+.12g", energy)
    return energy


def left_sweep(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    davidson_opts: dict,
) -> float:
    """Perform a right-to-left (left) half-sweep of 1-site DMRG.

    Visits sites from the current orthogonality center down to `0`, optimising
    each site tensor with Davidson, moving the center one step to the left, and
    updating the right environment for the next site. Site `0` is optimised
    last without moving the center further.

    After this call:
    - `mps.center == 0`.
    - `env_right[i]` is populated for `i = 0, …, L-2` (right environments
      built from the canonicalised tensors).

    Parameters
    ----------
    mps:
        MPS to optimise in-place. Must have `center` set.
    mpo:
        Hamiltonian MPO.
    env_left:
        Left environment blocks. All slots must be populated from a prior
        `right_sweep`.
    env_right:
        Right environment blocks.  `env_right[mps.center]` must be initialised.
    trunc:
        Truncation options forwarded to `mps.canonical` at each step.
        Pass `None` to disable truncation.
    davidson_opts:
        Keyword arguments forwarded to the Davidson solver.

    Returns
    -------
    float
        Variational energy at the last optimised site (leftmost site, site 0).
    """
    L = mps.L
    energy = 0.0
    _log.info("left  sweep: sites %d → 0", L - 1)

    for i in range(mps.center, 0, -1):
        # Optimise the site tensor at position i.
        energy, M_opt = optimize_site(
            mps[i], env_left[i], mpo[i], env_right[i], davidson_opts
        )
        mps[i] = M_opt
        _log.debug("  site %*d / %d  E = %+.12g", len(str(L - 1)), i, L - 1, energy)

        # Move the orthogonality center one step to the left.
        mps.canonical(i - 1, trunc=trunc)

        # Build the right environment for site i-1 from the now-canonicalised tensor.
        env_right[i - 1] = step_right_env(env_right[i], mps[i], mpo[i])

    # Optimise site 0 without moving the center further.
    energy, mps[0] = optimize_site(
        mps[0], env_left[0], mpo[0], env_right[0], davidson_opts
    )
    _log.debug("  site %*d / %d  E = %+.12g", len(str(L - 1)), 0, L - 1, energy)
    _log.info("left  sweep done  E = %+.12g", energy)
    return energy
