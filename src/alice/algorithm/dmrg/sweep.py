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


"""Half-sweep orchestration for 1-site and 2-site DMRG.

This module is a thin coordination layer. `forward_sweep` and `backward_sweep`
are scheme-aware dispatchers: they inspect `opts.scheme` and delegate to the
appropriate private implementation (`_forward_1s`, `_backward_1s`,
`_forward_2s`, `_backward_2s`).

Each per-scheme implementation is a sequential loop that handles only:

- deciding the order of sites (1-site) or bonds (2-site) to visit,
- moving the orthogonality center,
- routing environment updates through the `Environment` instances.

All local computation is delegated to pure functions in `scheme_1s` and
`scheme_2s`, which are the natural units of work for future parallel or
distributed execution.

1-site center movement
----------------------
`mps.canonical(target, trunc=trunc)` is safe to call in `_forward_1s` /
`_backward_1s` because `center` is always set before a sweep begins. With
`center` set, `canonical` immediately takes the direct one-step path without
performing a full sweep.

2-site center tracking
----------------------
`_forward_2s` and `_backward_2s` bypass `mps.canonical` entirely. After
splitting Θ via SVD the resulting tensors are already isometric; the center
is updated by writing `mps._center` directly, mirroring what `canonical`
does internally.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from alice.network import MPS, MPO

from .environ import Environment, step_left_env, step_right_env
from .scheme_1s import optimize_1site
from .scheme_2s import optimize_2site, split_forward, split_backward

if TYPE_CHECKING:
    from .dmrg import Options

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dispatcher functions
# ---------------------------------------------------------------------------

def forward_sweep(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    opts: 'Options',
) -> float:
    """Perform a left-to-right (forward) half-sweep.

    Dispatches to the 1-site or 2-site implementation based on `opts.scheme`.

    After this call:
    - `mps.center == mps.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-1` (1-site) or `i = 1, …, L-2` (2-site).

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
    opts:
        DMRG run options (scheme, truncation, Davidson parameters).

    Returns
    -------
    float
        Variational energy at the last optimised site or bond.
    """
    trunc, davidson_opts = _unpack_opts(opts)
    if opts.scheme == '1s':
        return _forward_1s(mps, mpo, env_left, env_right, trunc, davidson_opts)
    if opts.scheme == '2s':
        return _forward_2s(mps, mpo, env_left, env_right, trunc, davidson_opts)
    raise NotImplementedError(f"forward_sweep: unknown scheme {opts.scheme!r}")


def backward_sweep(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    opts: 'Options',
) -> float:
    """Perform a right-to-left (backward) half-sweep.

    Dispatches to the 1-site or 2-site implementation based on `opts.scheme`.

    After this call:
    - `mps.center == 0`.
    - `env_right[i]` is populated for `i = 0, …, L-2` (1-site) or `i = 1, …, L-2` (2-site).

    Parameters
    ----------
    mps:
        MPS to optimise in-place. Must have `center` set.
    mpo:
        Hamiltonian MPO.
    env_left:
        Left environment blocks. All slots must be populated from a prior
        `forward_sweep`.
    env_right:
        Right environment blocks. `env_right[mps.center]` must be initialised.
    opts:
        DMRG run options (scheme, truncation, Davidson parameters).

    Returns
    -------
    float
        Variational energy at the last optimised site or bond.
    """
    trunc, davidson_opts = _unpack_opts(opts)
    if opts.scheme == '1s':
        return _backward_1s(mps, mpo, env_left, env_right, trunc, davidson_opts)
    if opts.scheme == '2s':
        return _backward_2s(mps, mpo, env_left, env_right, trunc, davidson_opts)
    raise NotImplementedError(f"backward_sweep: unknown scheme {opts.scheme!r}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _unpack_opts(opts: 'Options'):
    """Extract the truncation dict and Davidson keyword-args from `opts`."""
    trunc: Optional[dict] = {'thresh': opts.trunc_thresh}
    if opts.max_bond is not None:
        trunc['nkeep'] = opts.max_bond
    davidson_opts = {
        'max_iter': opts.davidson_max_iter,
        'tol': opts.davidson_tol,
        'max_subspace': opts.davidson_max_subspace,
    }
    return trunc, davidson_opts


# ---------------------------------------------------------------------------
# 1-site implementations
# ---------------------------------------------------------------------------

def _forward_1s(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    davidson_opts: dict,
) -> float:
    """Left-to-right half-sweep for 1-site DMRG.

    Visits sites from the current orthogonality center to `L-1`, optimising
    each site tensor with Davidson, moving the center one step to the right,
    and updating the left environment for the next site. The rightmost site
    is optimised last without moving the center further.

    After this call:
    - `mps.center == mps.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-1`.
    """
    L = mps.L
    energy = 0.0
    w = len(str(L - 1))

    for i in range(mps.center, L - 1):
        # Optimise the site tensor at position i.
        energy, M_opt, davidson_error = optimize_1site(
            mps[i], env_left[i], mpo[i], env_right[i], davidson_opts
        )
        mps[i] = M_opt
        logger.debug("  site %*d / %d  local E = %+.12g", w, i, L - 1, energy)
        logger.debug("    davidson err = %.4e", davidson_error)

        # Move the orthogonality center one step to the right.
        # Because mps.center == i (set), canonical takes the direct one-step path.
        mps.canonical(i + 1, trunc=trunc)

        # Build the left environment for site i+1 from the now-canonicalised tensor.
        env_left[i + 1] = step_left_env(env_left[i], mps[i], mpo[i])

    # Optimise the rightmost site without moving the center further.
    energy, mps[L - 1], davidson_error = optimize_1site(
        mps[L - 1], env_left[L - 1], mpo[L - 1], env_right[L - 1], davidson_opts
    )
    logger.debug("  site %*d / %d  local E = %+.12g", w, L - 1, L - 1, energy)
    logger.debug("    davidson err = %.4e", davidson_error)
    return energy


def _backward_1s(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    davidson_opts: dict,
) -> float:
    """Right-to-left half-sweep for 1-site DMRG.

    Visits sites from the current orthogonality center down to `0`, optimising
    each site tensor with Davidson, moving the center one step to the left, and
    updating the right environment for the next site. Site `0` is optimised
    last without moving the center further.

    After this call:
    - `mps.center == 0`.
    - `env_right[i]` is populated for `i = 0, …, L-2`.
    """
    L = mps.L
    energy = 0.0
    w = len(str(L - 1))

    for i in range(mps.center, 0, -1):
        # Optimise the site tensor at position i.
        energy, M_opt, davidson_error = optimize_1site(
            mps[i], env_left[i], mpo[i], env_right[i], davidson_opts
        )
        mps[i] = M_opt
        logger.debug("  site %*d / %d  local E = %+.12g", w, i, L - 1, energy)
        logger.debug("    davidson err = %.4e", davidson_error)

        # Move the orthogonality center one step to the left.
        mps.canonical(i - 1, trunc=trunc)

        # Build the right environment for site i-1 from the now-canonicalised tensor.
        env_right[i - 1] = step_right_env(env_right[i], mps[i], mpo[i])

    # Optimise site 0 without moving the center further.
    energy, mps[0], davidson_error = optimize_1site(
        mps[0], env_left[0], mpo[0], env_right[0], davidson_opts
    )
    logger.debug("  site %*d / %d  local E = %+.12g", w, 0, L - 1, energy)
    logger.debug("    davidson err = %.4e", davidson_error)
    return energy


# ---------------------------------------------------------------------------
# 2-site implementations
# ---------------------------------------------------------------------------

def _forward_2s(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    davidson_opts: dict,
) -> float:
    """Left-to-right half-sweep for 2-site DMRG.

    Visits all L-1 bonds from (mps.center, mps.center+1) to (L-2, L-1),
    optimising the 2-site bond tensor Θ at each step via Davidson, then
    splitting it with SVD (no `mps.canonical` call).

    After this call:
    - `mps.center == mps.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-2`.
    """
    L = mps.L
    energy = 0.0
    w = len(str(L - 1))

    for i in range(mps.center, L - 1):
        energy, Theta_opt, davidson_error = optimize_2site(
            mps[i], mps[i + 1],
            env_left[i], mpo[i], mpo[i + 1], env_right[i + 1],
            davidson_opts,
        )
        logger.debug(
            "  bond (%*d, %*d) / %d  local E = %+.12g",
            w, i, w, i + 1, L - 1, energy,
        )
        logger.debug("    davidson err = %.4e", davidson_error)

        # Split Θ: M[i] becomes left-isometric; M[i+1] carries the singular values.
        itag = mps._bond_itag(i + 1)
        M_i, M_i1 = split_forward(Theta_opt, itag, trunc)
        mps[i] = M_i
        mps[i + 1] = M_i1
        mps._center = i + 1

        # Update the left environment for the next bond — not needed after the last.
        if i < L - 2:
            env_left[i + 1] = step_left_env(env_left[i], mps[i], mpo[i])

    return energy


def _backward_2s(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    davidson_opts: dict,
) -> float:
    """Right-to-left half-sweep for 2-site DMRG.

    Visits all L-1 bonds from (L-2, L-1) down to (0, 1), optimising the
    2-site bond tensor Θ at each step via Davidson, then splitting it with SVD.

    After this call:
    - `mps.center == 0`.
    - `env_right[i]` is populated for `i = 1, …, L-2`.
    """
    L = mps.L
    energy = 0.0
    w = len(str(L - 1))

    for i in range(L - 2, -1, -1):
        energy, Theta_opt, davidson_error = optimize_2site(
            mps[i], mps[i + 1],
            env_left[i], mpo[i], mpo[i + 1], env_right[i + 1],
            davidson_opts,
        )
        logger.debug(
            "  bond (%*d, %*d) / %d  local E = %+.12g",
            w, i, w, i + 1, L - 1, energy,
        )
        logger.debug("    davidson err = %.4e", davidson_error)

        # Split Θ: M[i+1] becomes right-isometric; M[i] carries the singular values.
        itag = mps._bond_itag(i + 1)
        M_i, M_i1 = split_backward(Theta_opt, itag, trunc)
        mps[i] = M_i
        mps[i + 1] = M_i1
        mps._center = i

        # Update the right environment for the next bond — not needed after the last.
        if i > 0:
            env_right[i] = step_right_env(env_right[i + 1], mps[i + 1], mpo[i + 1])

    return energy
