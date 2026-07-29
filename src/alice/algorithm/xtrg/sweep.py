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


"""Sweep orchestration for variational MPO-MPO compression.

A full variational sweep consists of one forward (left-to-right) half-sweep
followed by one backward (right-to-left) half-sweep. The number of full
sweeps per squaring step is controlled by `Options.n_sweeps`.

After a forward sweep:

- `mpo_c.center == mpo_c.L - 1`.
- `env_left[i]` is populated for `i = 1, …, L-1` (1-site)
  or `i = 1, …, L-2` (2-site).

After a backward sweep:

- `mpo_c.center == 0`.
- `env_right[i]` is populated for `i = 0, …, L-2` (1-site)
  or `i = 1, …, L-2` (2-site).
"""

from __future__ import annotations

import logging
from typing import Optional

from alice.network.thermal import NormalMPO

from .environ import Environment, step_left_env, step_right_env
from .scheme_1s import local_update_1s
from .scheme_2s import discarded_weight, local_update_2s, split_backward, split_forward

logger = logging.getLogger(__name__)


def forward_sweep(
    mpo_a: NormalMPO,
    mpo_b: NormalMPO,
    mpo_c: NormalMPO,
    env_left: Environment,
    env_right: Environment,
    opts,
) -> None:
    """Perform a left-to-right (forward) half-sweep updating `mpo_c` in-place.

    Dispatches to the 1-site or 2-site implementation based on `opts.scheme`.

    After this call `mpo_c.center == mpo_c.L - 1`.

    Parameters
    ----------
    mpo_a:
        Factor MPO A (read-only during sweep).
    mpo_b:
        Factor MPO B (read-only during sweep).
    mpo_c:
        Compressed MPO C; updated in-place. Must have `center` set.
    env_left:
        Left environment blocks. `env_left[mpo_c.center]` must be initialized.
    env_right:
        Right environment blocks. All slots must be initialized before the
        first call (populated by `build_right_envs`).
    opts:
        XTRG run options (scheme, truncation parameters).
    """
    trunc = _make_trunc(opts)
    if opts.scheme == '1s':
        _forward_1s(mpo_a, mpo_b, mpo_c, env_left, env_right, trunc)
    elif opts.scheme == '2s':
        _forward_2s(mpo_a, mpo_b, mpo_c, env_left, env_right, trunc)
    else:
        raise NotImplementedError(f"forward_sweep: unknown scheme {opts.scheme!r}")


def backward_sweep(
    mpo_a: NormalMPO,
    mpo_b: NormalMPO,
    mpo_c: NormalMPO,
    env_left: Environment,
    env_right: Environment,
    opts,
) -> float:
    """Perform a right-to-left (backward) half-sweep updating `mpo_c` in-place.

    Dispatches to the 1-site or 2-site implementation based on `opts.scheme`.

    After this call `mpo_c.center == 0`.

    Parameters
    ----------
    mpo_a:
        Factor MPO A (read-only during sweep).
    mpo_b:
        Factor MPO B (read-only during sweep).
    mpo_c:
        Compressed MPO C; updated in-place. Must have `center` set.
    env_left:
        Left environment blocks. All slots must be populated from a prior
        `forward_sweep`.
    env_right:
        Right environment blocks. `env_right[mpo_c.center]` must be initialized.
    opts:
        XTRG run options (scheme, truncation parameters).

    Returns
    -------
    float
        Discarded weight at the center bond (2-site only; `0.0` for 1-site).
    """
    trunc = _make_trunc(opts)
    if opts.scheme == '1s':
        _backward_1s(mpo_a, mpo_b, mpo_c, env_left, env_right, trunc)
        return 0.0
    if opts.scheme == '2s':
        return _backward_2s(mpo_a, mpo_b, mpo_c, env_left, env_right, trunc)
    raise NotImplementedError(f"backward_sweep: unknown scheme {opts.scheme!r}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_trunc(opts) -> Optional[dict]:
    """Build the truncation dict from options."""
    trunc: Optional[dict] = {'thresh': opts.trunc_thresh}
    if opts.max_bond is not None:
        trunc['nkeep'] = opts.max_bond
    return trunc


# ---------------------------------------------------------------------------
# 1-site implementations
# ---------------------------------------------------------------------------

def _forward_1s(
    mpo_a: NormalMPO,
    mpo_b: NormalMPO,
    mpo_c: NormalMPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
) -> None:
    """Forward half-sweep for the 1-site scheme.

    Visits sites from `mpo_c.center` to `L-1`. At each site (except the last)
    the local update is followed by a QR step to move the center rightward and
    update `env_left`. The last site is updated without moving the center.

    After this call `mpo_c.center == L-1`.
    """
    L = mpo_c.L
    w = len(str(L - 1))

    for i in range(mpo_c.center, L - 1):
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i)
        mpo_c[i] = local_update_1s(E_left, mpo_a[i], mpo_b[i], E_right)
        logger.debug("  [forward 1s] site %*d / %d", w, i, L - 1)

        # Move orthogonality center one step to the right.
        mpo_c.canonical(i + 1, trunc=trunc)

        # Update left environment for the next site from the new left-isometric tensor.
        env_left[i + 1] = step_left_env(E_left, mpo_a[i], mpo_b[i], mpo_c[i])

    # Update the rightmost site without moving the center.
    E_left = env_left.fetch(L - 1)
    E_right = env_right.fetch(L - 1)
    mpo_c[L - 1] = local_update_1s(E_left, mpo_a[L - 1], mpo_b[L - 1], E_right)
    logger.debug("  [forward 1s] site %*d / %d", w, L - 1, L - 1)


def _backward_1s(
    mpo_a: NormalMPO,
    mpo_b: NormalMPO,
    mpo_c: NormalMPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
) -> None:
    """Backward half-sweep for the 1-site scheme.

    Visits sites from `mpo_c.center` down to `0`. At each site (except site 0)
    the local update is followed by an LQ step to move the center leftward and
    update `env_right`. Site 0 is updated without moving the center.

    After this call `mpo_c.center == 0`.
    """
    L = mpo_c.L
    w = len(str(L - 1))

    for i in range(mpo_c.center, 0, -1):
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i)
        mpo_c[i] = local_update_1s(E_left, mpo_a[i], mpo_b[i], E_right)
        logger.debug("  [backward 1s] site %*d / %d", w, i, L - 1)

        # Move orthogonality center one step to the left.
        mpo_c.canonical(i - 1, trunc=trunc)

        # Update right environment for the next site.
        env_right[i - 1] = step_right_env(E_right, mpo_a[i], mpo_b[i], mpo_c[i])

    # Update site 0 without moving the center.
    E_left = env_left.fetch(0)
    E_right = env_right.fetch(0)
    mpo_c[0] = local_update_1s(E_left, mpo_a[0], mpo_b[0], E_right)
    logger.debug("  [backward 1s] site %*d / %d", w, 0, L - 1)


# ---------------------------------------------------------------------------
# 2-site implementations
# ---------------------------------------------------------------------------

def _forward_2s(
    mpo_a: NormalMPO,
    mpo_b: NormalMPO,
    mpo_c: NormalMPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
) -> None:
    """Forward half-sweep for the 2-site scheme.

    Visits all L-1 bonds from (center, center+1) to (L-2, L-1). At each bond
    the 2-site bond tensor Θ is formed and split via SVD.

    After this call `mpo_c.center == L-1`.
    """
    L = mpo_c.L
    pair_w = 2 * len(str(L - 1)) + 4

    for i in range(mpo_c.center, L - 1):
        E_left = env_left.fetch(i)
        theta = local_update_2s(
            E_left, mpo_a[i], mpo_b[i], mpo_a[i + 1], mpo_b[i + 1],
            env_right.fetch(i + 1),
        )
        pair = f"({i}, {i + 1})".center(pair_w)
        logger.debug("  [forward 2s] sites %s / %d", pair, L - 1)

        itag = mpo_c._bond_itag(i + 1)
        mpo_c[i], mpo_c[i + 1] = split_forward(theta, itag, trunc)
        mpo_c._center = i + 1

        # Update the left environment for the next bond (not needed after the last).
        if i < L - 2:
            env_left[i + 1] = step_left_env(E_left, mpo_a[i], mpo_b[i], mpo_c[i])


def _backward_2s(
    mpo_a: NormalMPO,
    mpo_b: NormalMPO,
    mpo_c: NormalMPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
) -> float:
    """Backward half-sweep for the 2-site scheme.

    Visits all L-1 bonds from (L-2, L-1) down to (0, 1). The discarded weight
    is measured at the center bond `L // 2 - 1`.

    After this call `mpo_c.center == 0`.

    Returns
    -------
    float
        Discarded weight at the center bond.
    """
    L = mpo_c.L
    dw = 0.0
    center_bond = L // 2 - 1
    pair_w = 2 * len(str(L - 1)) + 4

    for i in range(L - 2, -1, -1):
        E_right = env_right.fetch(i + 1)
        theta = local_update_2s(
            env_left.fetch(i), mpo_a[i], mpo_b[i], mpo_a[i + 1], mpo_b[i + 1],
            E_right,
        )
        pair = f"({i}, {i + 1})".center(pair_w)
        logger.debug("  [backward 2s] sites %s / %d", pair, L - 1)

        # Measure discarded weight once at the center bond.
        if i == center_bond:
            dw = discarded_weight(theta, trunc)
            logger.debug("    discarded weight = %.4e", dw)

        itag = mpo_c._bond_itag(i + 1)
        mpo_c[i], mpo_c[i + 1] = split_backward(theta, itag, trunc)
        mpo_c._center = i

        # Update the right environment for the next bond (not needed after the last).
        if i > 0:
            env_right[i] = step_right_env(E_right, mpo_a[i + 1], mpo_b[i + 1], mpo_c[i + 1])

    return dw
