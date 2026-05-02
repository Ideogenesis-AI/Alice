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


"""Half-sweep orchestration for 1-site, 2-site, and 1-site-plus DMRG.

This module is a thin coordination layer. `forward_sweep` and `backward_sweep`
are scheme-aware dispatchers: they inspect `opts.scheme` and delegate to the
appropriate private implementation (`_forward_1s`, `_backward_1s`,
`_forward_2s`, `_backward_2s`, `_forward_1sp`, `_backward_1sp`).

Each per-scheme implementation is a sequential loop that handles only:

- deciding the order of sites (1-site) or bonds (2-site / 1-site-plus) to visit,
- moving the orthogonality center,
- routing environment updates through the `Environment` instances.

All local computation is delegated to pure functions in `scheme_1s`, `scheme_2s`,
and `complement`, which are the natural units of work for future parallel or
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

1-site-plus center movement
---------------------------
`_forward_1sp` and `_backward_1sp` follow the 1-site pattern: after the
complement expansion, `mps.canonical` is called to move the center and apply
the truncation. Because the expanded M[i] and M[i+1] are set in `mps` before
the canonical call, the standard QR/SVD path operates on the expanded bond and
truncates it back to at most `max_bond`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

from alice.network import MPS, MPO

from .complement import expand_backward, expand_forward
from .environ import Environment, step_left_env, step_right_env
from .scheme_1s import optimize_1site
from .scheme_2s import optimize_2site, split_forward, split_backward, discarded_weight

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
    opts: Options,
) -> float:
    """Perform a left-to-right (forward) half-sweep.

    Dispatches to the 1-site, 2-site, or 1-site-plus implementation based on
    `opts.scheme`.

    After this call:
    - `mps.center == mps.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-1` (1-site / 1-site-plus)
      or `i = 1, …, L-2` (2-site).

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
    trunc, davidson_opts, cbe_opts = _unpack_opts(opts)
    if opts.scheme == '1s':
        return _forward_1s(mps, mpo, env_left, env_right, trunc, davidson_opts)
    if opts.scheme == '2s':
        return _forward_2s(mps, mpo, env_left, env_right, trunc, davidson_opts)
    if opts.scheme == '1sp':
        return _forward_1sp(mps, mpo, env_left, env_right, trunc, davidson_opts, **cbe_opts)
    raise NotImplementedError(f"forward_sweep: unknown scheme {opts.scheme!r}")


def backward_sweep(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    opts: Options,
) -> Tuple[float, float]:
    """Perform a right-to-left (backward) half-sweep.

    Dispatches to the 1-site, 2-site, or 1-site-plus implementation based on
    `opts.scheme`.

    After this call:
    - `mps.center == 0`.
    - `env_right[i]` is populated for `i = 0, …, L-2` (1-site / 1-site-plus)
      or `i = 1, …, L-2` (2-site).

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
    float
        Discarded weight at the center bond (2-site only; `0.0` for 1-site
        and 1-site-plus).
    """
    trunc, davidson_opts, cbe_opts = _unpack_opts(opts)
    if opts.scheme == '1s':
        return _backward_1s(mps, mpo, env_left, env_right, trunc, davidson_opts), 0.0
    if opts.scheme == '2s':
        return _backward_2s(mps, mpo, env_left, env_right, trunc, davidson_opts)
    if opts.scheme == '1sp':
        return _backward_1sp(mps, mpo, env_left, env_right, trunc, davidson_opts, **cbe_opts), 0.0
    raise NotImplementedError(f"backward_sweep: unknown scheme {opts.scheme!r}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _unpack_opts(opts: Options):
    """Extract the truncation dict, Davidson keyword-args, and CBE opts from `opts`."""
    trunc: Optional[dict] = {'thresh': opts.trunc_thresh}
    if opts.max_bond is not None:
        trunc['nkeep'] = opts.max_bond
    davidson_opts = {
        'max_iter': opts.davidson_max_iter,
        'tol': opts.davidson_tol,
        'max_subspace': opts.davidson_max_subspace,
    }
    cbe_opts = {
        'k_expand': opts.expand_k,
        'alpha': opts.expand_alpha,
    }
    return trunc, davidson_opts, cbe_opts


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
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i)
        energy, mps[i], davidson_error = optimize_1site(
            mps[i], mpo[i], E_left, E_right, davidson_opts
        )
        logger.debug("  site %*d / %d  local E = %+.12g", w, i, L - 1, energy)
        logger.debug("    davidson err = %.4e", davidson_error)

        # Move the orthogonality center one step to the right.
        # Because mps.center == i (set), canonical takes the direct one-step path.
        mps.canonical(i + 1, trunc=trunc)

        # Build the left environment for site i+1 from the now-canonicalised tensor.
        env_left[i + 1] = step_left_env(E_left, mps[i], mpo[i])

    # Optimise the rightmost site without moving the center further.
    energy, mps[L - 1], davidson_error = optimize_1site(
        mps[L - 1], mpo[L - 1], env_left.fetch(L - 1), env_right.fetch(L - 1), davidson_opts
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
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i)
        energy, mps[i], davidson_error = optimize_1site(
            mps[i], mpo[i], E_left, E_right, davidson_opts
        )
        logger.debug("  site %*d / %d  local E = %+.12g", w, i, L - 1, energy)
        logger.debug("    davidson err = %.4e", davidson_error)

        # Move the orthogonality center one step to the left.
        mps.canonical(i - 1, trunc=trunc)

        # Build the right environment for site i-1 from the now-canonicalised tensor.
        env_right[i - 1] = step_right_env(E_right, mps[i], mpo[i])

    # Optimise site 0 without moving the center further.
    energy, mps[0], davidson_error = optimize_1site(
        mps[0], mpo[0], env_left.fetch(0), env_right.fetch(0), davidson_opts
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
    pair_w = 2 * len(str(L - 1)) + 4

    for i in range(mps.center, L - 1):
        # Optimise the 2-site bond tensor at position (i, i+1).
        E_left = env_left.fetch(i)
        energy, theta_opt, davidson_error = optimize_2site(
            mps[i], mps[i + 1], mpo[i], mpo[i + 1],
            E_left, env_right.fetch(i + 1), davidson_opts
        )
        pair = ("(%d, %d)" % (i, i + 1)).center(pair_w)
        logger.debug("  site %s / %d  local E = %+.12g", pair, L - 1, energy)
        logger.debug("    davidson err = %.4e", davidson_error)

        # Split Θ: M[i] becomes left-isometric; M[i+1] carries the singular values.
        itag = mps._bond_itag(i + 1)
        mps[i], mps[i + 1] = split_forward(theta_opt, itag, trunc)
        mps._center = i + 1

        # Update the left environment for the next bond — not needed after the last.
        if i < L - 2:
            env_left[i + 1] = step_left_env(E_left, mps[i], mpo[i])

    return energy


def _backward_2s(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    davidson_opts: dict,
) -> Tuple[float, float]:
    """Right-to-left half-sweep for 2-site DMRG.

    Visits all L-1 bonds from (L-2, L-1) down to (0, 1), optimising the
    2-site bond tensor Θ at each step via Davidson, then splitting it with SVD.
    At the center bond (`i == L // 2 - 1`) the discarded weight is measured
    via a second SVD call.

    After this call:
    - `mps.center == 0`.
    - `env_right[i]` is populated for `i = 1, …, L-2`.
    """
    L = mps.L
    energy = 0.0
    dw = 0.0
    center_bond = L // 2 - 1
    pair_w = 2 * len(str(L - 1)) + 4

    for i in range(L - 2, -1, -1):
        # Optimise the 2-site bond tensor at position (i, i+1).
        E_right = env_right.fetch(i + 1)
        energy, theta_opt, davidson_error = optimize_2site(
            mps[i], mps[i + 1], mpo[i], mpo[i + 1],
            env_left.fetch(i), E_right, davidson_opts
        )
        pair = ("(%d, %d)" % (i, i + 1)).center(pair_w)
        logger.debug("  site %s / %d  local E = %+.12g", pair, L - 1, energy)
        logger.debug("    davidson err = %.4e", davidson_error)

        # Measure the discarded weight once at the center bond.
        if i == center_bond:
            dw = discarded_weight(theta_opt, trunc)
            logger.debug("    discarded weight = %.4e", dw)

        # Split Θ: M[i+1] becomes right-isometric; M[i] carries the singular values.
        itag = mps._bond_itag(i + 1)
        mps[i], mps[i + 1] = split_backward(theta_opt, itag, trunc)
        mps._center = i

        # Update the right environment for the next bond — not needed after the last.
        if i > 0:
            env_right[i] = step_right_env(E_right, mps[i + 1], mpo[i + 1])

    return energy, dw


# ---------------------------------------------------------------------------
# 1-site-plus (CBE) implementations
# ---------------------------------------------------------------------------

def _forward_1sp(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    davidson_opts: dict,
    k_expand: int,
    alpha: Optional[int],
) -> float:
    """Left-to-right half-sweep for 1-site-plus (CBE) DMRG.

    At each bond (i, i+1) before the Davidson step at site i:

    1. The bond is expanded via `expand_forward`: cheap truncated factors of
       Θ = M[i] ⊗ M[i+1] are formed, H is applied in factored half-sweeps,
       the kept subspace is projected out, and the top-`k_expand` complement
       directions are added via `oplus`.
    2. Davidson optimises the expanded M[i] using the original left environment
       and the expanded right environment from step 1.
    3. `mps.canonical(i+1, trunc=trunc)` moves the center and truncates the
       expanded bond back to at most `max_bond`.
    4. `env_left[i+1]` is updated from the truncated left-isometric M[i].

    The rightmost site (i = L-1) has no right neighbor to expand into and
    receives a plain 1-site Davidson update.

    After this call:
    - `mps.center == mps.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-1`.
    """
    L = mps.L
    energy = 0.0
    w = len(str(L - 1))

    for i in range(mps.center, L - 1):
        E_left = env_left.fetch(i)

        # CBE complement expansion for bond (i, i+1).
        # env_right.fetch(i+1) is the pre-built env to the right of site i+1.
        M_i_exp, M_i1_exp, E_right_i_exp = expand_forward(
            mps[i], mps[i + 1],
            mpo[i], mpo[i + 1],
            E_left, env_right.fetch(i + 1),
            k_expand, alpha,
        )

        # 1-site Davidson on the expanded M[i] with the expanded right environment.
        energy, M_i_opt, davidson_error = optimize_1site(
            M_i_exp, mpo[i], E_left, E_right_i_exp, davidson_opts
        )
        logger.debug("  site %*d / %d  local E = %+.12g", w, i, L - 1, energy)
        logger.debug("    davidson err = %.4e", davidson_error)

        # Store the expanded tensors then move the center (truncates expanded bond).
        mps[i]     = M_i_opt
        mps[i + 1] = M_i1_exp
        mps.canonical(i + 1, trunc=trunc)

        # Build the left environment for site i+1 from the now-truncated M[i].
        env_left[i + 1] = step_left_env(E_left, mps[i], mpo[i])

    # Rightmost site: no right neighbor, plain 1-site update.
    energy, mps[L - 1], davidson_error = optimize_1site(
        mps[L - 1], mpo[L - 1],
        env_left.fetch(L - 1), env_right.fetch(L - 1),
        davidson_opts,
    )
    logger.debug("  site %*d / %d  local E = %+.12g", w, L - 1, L - 1, energy)
    logger.debug("    davidson err = %.4e", davidson_error)
    return energy


def _backward_1sp(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    davidson_opts: dict,
    k_expand: int,
    alpha: Optional[int],
) -> float:
    """Right-to-left half-sweep for 1-site-plus (CBE) DMRG.

    Mirror of `_forward_1sp` for backward sweeps. At each bond (i-1, i)
    before the Davidson step at site i:

    1. `expand_backward` expands the bond using the complement of M[i-1]
       (left projector) and M[i] (right projector).
    2. Davidson optimises the expanded M[i] with the expanded left environment.
    3. `mps.canonical(i-1, trunc=trunc)` truncates and moves the center left.
    4. `env_right[i-1]` is updated from the truncated right-isometric M[i].

    The leftmost site (i = 0) receives a plain 1-site Davidson update.

    After this call:
    - `mps.center == 0`.
    - `env_right[i]` is populated for `i = 0, …, L-2`.
    """
    L = mps.L
    energy = 0.0
    w = len(str(L - 1))

    for i in range(mps.center, 0, -1):
        E_right = env_right.fetch(i)

        # CBE complement expansion for bond (i-1, i).
        # env_left.fetch(i-1) is the left env to the left of site i-1.
        M_i_exp, M_im1_exp, E_left_i_exp = expand_backward(
            mps[i], mps[i - 1],
            mpo[i], mpo[i - 1],
            env_left.fetch(i - 1), E_right,
            k_expand, alpha,
        )

        # 1-site Davidson on the expanded M[i] with the expanded left environment.
        energy, M_i_opt, davidson_error = optimize_1site(
            M_i_exp, mpo[i], E_left_i_exp, E_right, davidson_opts
        )
        logger.debug("  site %*d / %d  local E = %+.12g", w, i, L - 1, energy)
        logger.debug("    davidson err = %.4e", davidson_error)

        # Store the expanded tensors then move the center (truncates expanded bond).
        mps[i]     = M_i_opt
        mps[i - 1] = M_im1_exp
        mps.canonical(i - 1, trunc=trunc)

        # Build the right environment for site i-1 from the now-truncated M[i].
        env_right[i - 1] = step_right_env(E_right, mps[i], mpo[i])

    # Leftmost site: no left neighbor, plain 1-site update.
    energy, mps[0], davidson_error = optimize_1site(
        mps[0], mpo[0],
        env_left.fetch(0), env_right.fetch(0),
        davidson_opts,
    )
    logger.debug("  site %*d / %d  local E = %+.12g", w, 0, L - 1, energy)
    logger.debug("    davidson err = %.4e", davidson_error)
    return energy
