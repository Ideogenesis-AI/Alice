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


"""Half-sweep orchestration for 1-site, 2-site and 1-site-plus TDVP.

This module is a thin coordination layer. `forward_sweep` and `backward_sweep`
are scheme-aware dispatchers: they inspect `opts.scheme` and delegate to the
appropriate private implementation (`_forward_1s`, `_backward_1s`,
`_forward_2s`, `_backward_2s`, `_forward_1sp`, `_backward_1sp`). `time_step`
composes them into one time step according to `opts.order`.

Each per-scheme implementation is a sequential loop that handles only:

- deciding the order of sites (1-site, 1-site-plus) or bonds (2-site) to visit,
- moving the orthogonality center,
- routing environment updates through the `Environment` instances.

All local computation is delegated to pure functions in `scheme_1s`,
`scheme_2s` and `complement`, which are the natural units of work for future
parallel or distributed execution.

The 1-site-plus half-sweeps are the 1-site ones with a bond expansion spliced
in ahead of every local evolution, so they reuse `scheme_1s` in full and differ
only in what they hand it: a site tensor living on a bond widened by up to
`k_expand` complement directions, an environment that is exact in that widened
basis, and a `trunc` passed to the split, which `'1s'` leaves at `None`. The
expansion itself leaves the state untouched, so a `'1sp'` sweep with
`expand_k=0` is a `'1s'` sweep.

A sweep *is* a time step
------------------------
This is the structural difference from DMRG. There a sweep is one iteration of
a variational search that repeats until the energy converges; here a sweep
advances physical time by a fixed amount and is never repeated. There is no
inner convergence loop and no `e_tol` analogue, which is why the sweeps below
take a `dt` argument and return error diagnostics rather than an energy.

Integrator orders
-----------------
- `order=1` (Lie-Trotter): one half-sweep per time step, with the full `dt`.
  The direction alternates between steps, since a half-sweep leaves the center
  at the far end of the chain.
- `order=2` (Lubich symmetric): a forward half-sweep with `dt/2` followed by a
  backward half-sweep with `dt/2`. The step is its own adjoint and the center
  returns to site 0 after every step. This is the default.

A single half-sweep is only a first-order map, but alternating its direction
makes any *pair* of consecutive `order=1` steps symmetric. An `order=1` run of
`2n` steps at `dt/2` therefore costs the same as, and agrees to roundoff with,
an `order=2` run of `n` steps at `dt` — both are globally second order. What
`order=2` buys is not accuracy per unit work but *where the accuracy lands*:
every recorded step ends a complete forward-and-backward sweep pair with the
center back at site 0, so observables and checkpoints are second-order accurate
at every snapshot, whereas `order=1` snapshots taken after an odd step carry
first-order error. Prefer `order=1` only when the per-step bookkeeping of
`order=2` is a measured cost and the trajectory is sampled sparsely at even
step counts.

Center movement
---------------
Neither scheme calls `mps.canonical`. The splits in `scheme_1s` and `scheme_2s`
already produce isometric tensors, so the center is updated by writing
`mps._center` directly, mirroring what `canonical` does internally. Calling
`canonical` here would be worse than redundant: it would re-decompose tensors
that the evolution has just placed in the correct gauge.

Sweeps must be complete
-----------------------
Unlike a DMRG sweep, which may start from a center anywhere in the chain, a
TDVP sweep has to visit every site exactly once — a site skipped is a site
left behind in time. The dispatchers therefore require the center to sit at the
appropriate end of the chain and raise `ValueError` otherwise.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

from alice.network import MPS, MPO

from .complement import expand_backward, expand_forward
from .environ import Environment, step_left_env, step_right_env
from .scheme_1s import (
    absorb_backward,
    absorb_forward,
    evolve_1site,
    evolve_bond,
    split_backward as split_site_backward,
    split_forward as split_site_forward,
)
from .scheme_2s import discarded_weight, evolve_2site, split_backward, split_forward

if TYPE_CHECKING:
    from .tdvp import Options

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dispatcher functions
# ---------------------------------------------------------------------------

def time_step(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    opts: Options,
    forward: bool = True,
) -> Tuple[float, float]:
    """Advance the state by one time step of `opts.dt`.

    Composes half-sweeps according to `opts.order`. For `order=2` the `forward`
    flag is ignored: the symmetric step always runs a forward half-sweep and
    then a backward one, returning the center to site 0. For `order=1` a single
    half-sweep is performed in the requested direction, and the caller is
    responsible for alternating `forward` between steps.

    Parameters
    ----------
    mps:
        MPS to evolve in-place.
    mpo:
        Hamiltonian MPO. Must be complex, as must `mps`.
    env_left:
        Left environment blocks.
    env_right:
        Right environment blocks.
    opts:
        TDVP run options (scheme, order, dt, truncation, Krylov parameters).
    forward:
        Direction of the half-sweep for `order=1`. Ignored for `order=2`.

    Returns
    -------
    float
        Largest Krylov error estimate over all local evolutions in the step.
    float
        Largest discarded weight at the mid-chain bond over the step
        (`'2s'` and `'1sp'`; `0.0` for `'1s'`, which never truncates).
    """
    if opts.order == 2:
        half = 0.5 * opts.dt
        err_f, dw_f = forward_sweep(mps, mpo, env_left, env_right, opts, half)
        err_b, dw_b = backward_sweep(mps, mpo, env_left, env_right, opts, half)

        return max(err_f, err_b), max(dw_f, dw_b)

    if opts.order == 1:
        sweep = forward_sweep if forward else backward_sweep

        return sweep(mps, mpo, env_left, env_right, opts, opts.dt)

    raise NotImplementedError(f"time_step: unsupported integrator order {opts.order!r}")


def forward_sweep(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    opts: Options,
    dt: float,
) -> Tuple[float, float]:
    """Perform a left-to-right (forward) half-sweep over `dt`.

    Dispatches to the 1-site, 2-site or 1-site-plus implementation based on
    `opts.scheme`.

    Before this call `mps.center` must be `0`. After it:
    - `mps.center == mps.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-1` (`'1s'`, `'1sp'`) or
      `i = 1, …, L-2` (`'2s'`).

    Parameters
    ----------
    mps:
        MPS to evolve in-place. Must have `center == 0`.
    mpo:
        Hamiltonian MPO.
    env_left:
        Left environment blocks. `env_left[0]` must be initialized.
    env_right:
        Right environment blocks. All slots must be initialized before the
        first call (populated by `build_right_envs`).
    opts:
        TDVP run options.
    dt:
        Time increment for this half-sweep: `opts.dt` for `order=1`,
        `opts.dt / 2` for `order=2`.

    Returns
    -------
    float
        Largest Krylov error estimate over the half-sweep.
    float
        Discarded weight at the mid-chain bond (`'2s'` and `'1sp'`; `0.0` for
        `'1s'`).

    Raises
    ------
    ValueError
        If `mps.center` is not `0`, which would leave part of the chain
        un-evolved.
    """
    if mps.center != 0:
        raise ValueError(
            f"forward_sweep requires the orthogonality center at site 0, "
            f"found {mps.center}; a TDVP sweep must visit every site"
        )

    trunc, krylov_opts, cbe_opts = _unpack_opts(opts)
    if opts.scheme == '1s':
        return _forward_1s(mps, mpo, env_left, env_right, krylov_opts, dt), 0.0
    if opts.scheme == '2s':
        return _forward_2s(mps, mpo, env_left, env_right, trunc, krylov_opts, dt)
    if opts.scheme == '1sp':
        return _forward_1sp(mps, mpo, env_left, env_right, trunc, krylov_opts, dt,
                            **cbe_opts)
    raise NotImplementedError(f"forward_sweep: unknown scheme {opts.scheme!r}")


def backward_sweep(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    opts: Options,
    dt: float,
) -> Tuple[float, float]:
    """Perform a right-to-left (backward) half-sweep over `dt`.

    Dispatches to the 1-site, 2-site or 1-site-plus implementation based on
    `opts.scheme`.

    Before this call `mps.center` must be `mps.L - 1`. After it:
    - `mps.center == 0`.
    - `env_right[i]` is populated for `i = 0, …, L-2` (`'1s'`, `'1sp'`) or
      `i = 1, …, L-2` (`'2s'`).

    Parameters
    ----------
    mps:
        MPS to evolve in-place. Must have `center == mps.L - 1`.
    mpo:
        Hamiltonian MPO.
    env_left:
        Left environment blocks. All slots must be populated from a prior
        `forward_sweep`.
    env_right:
        Right environment blocks. `env_right[mps.L - 1]` must be initialized.
    opts:
        TDVP run options.
    dt:
        Time increment for this half-sweep.

    Returns
    -------
    float
        Largest Krylov error estimate over the half-sweep.
    float
        Discarded weight at the mid-chain bond (`'2s'` and `'1sp'`; `0.0` for
        `'1s'`).

    Raises
    ------
    ValueError
        If `mps.center` is not `mps.L - 1`.
    """
    if mps.center != mps.L - 1:
        raise ValueError(
            f"backward_sweep requires the orthogonality center at site "
            f"{mps.L - 1}, found {mps.center}; a TDVP sweep must visit every site"
        )

    trunc, krylov_opts, cbe_opts = _unpack_opts(opts)
    if opts.scheme == '1s':
        return _backward_1s(mps, mpo, env_left, env_right, krylov_opts, dt), 0.0
    if opts.scheme == '2s':
        return _backward_2s(mps, mpo, env_left, env_right, trunc, krylov_opts, dt)
    if opts.scheme == '1sp':
        return _backward_1sp(mps, mpo, env_left, env_right, trunc, krylov_opts, dt,
                             **cbe_opts)
    raise NotImplementedError(f"backward_sweep: unknown scheme {opts.scheme!r}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _unpack_opts(opts: Options):
    """Extract the truncation dict, Krylov keyword-args and CBE opts from `opts`."""
    trunc: Optional[dict] = {'thresh': opts.trunc_thresh}
    if opts.max_bond is not None:
        trunc['nkeep'] = opts.max_bond
    krylov_opts = {
        'max_krylov': opts.krylov_dim,
        'tol': opts.krylov_tol,
    }
    cbe_opts = {
        'k_expand': opts.expand_k,
        'alpha': opts.expand_alpha,
    }
    return trunc, krylov_opts, cbe_opts


# ---------------------------------------------------------------------------
# 1-site implementations
# ---------------------------------------------------------------------------

def _forward_1s(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    krylov_opts: dict,
    dt: float,
) -> float:
    """Forward (left-to-right) half-sweep for 1-site TDVP.

    At each site `i < L-1`: evolve the site tensor forward by `dt`, split off
    the left isometry, update the left environment, evolve the exposed bond
    tensor *backward* by `dt`, and absorb it into site `i+1`. The rightmost
    site is evolved last; it has no bond to its right to correct.

    After this call:
    - `mps.center == mps.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-1`.
    """
    L = mps.L
    err_max = 0.0
    w = len(str(L - 1))

    for i in range(L - 1):
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i)

        # Forward evolution of the site tensor.
        M, err_site = evolve_1site(mps[i], mpo[i], E_left, E_right, dt, krylov_opts)

        # Peel off the isometry; the bond tensor is what double counts.
        itag = mps._bond_itag(i + 1)
        mps[i], C = split_site_forward(M, itag)
        env_left[i + 1] = step_left_env(E_left, mps[i], mpo[i])

        # Backward evolution of the bond tensor removes the double counting.
        # env_right[i] spans sites i+1, …, L-1, which is exactly the right half
        # of this bond, and is still valid: site i+1 has not been touched yet.
        C, err_bond = evolve_bond(C, env_left.fetch(i + 1), E_right, -dt, krylov_opts)
        mps[i + 1] = absorb_forward(C, mps[i + 1])
        mps._center = i + 1

        err_max = max(err_max, err_site, err_bond)
        logger.debug("  site %*d / %d  krylov err = %.4e", w, i, L - 1, err_site)
        logger.debug("    bond %d krylov err = %.4e", i + 1, err_bond)

    # Rightmost site: no bond to its right, so no back-evolution.
    mps[L - 1], err_site = evolve_1site(
        mps[L - 1], mpo[L - 1], env_left.fetch(L - 1), env_right.fetch(L - 1),
        dt, krylov_opts,
    )

    err_max = max(err_max, err_site)
    logger.debug("  site %*d / %d  krylov err = %.4e", w, L - 1, L - 1, err_site)

    return err_max


def _backward_1s(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    krylov_opts: dict,
    dt: float,
) -> float:
    """Backward (right-to-left) half-sweep for 1-site TDVP.

    Mirror image of `_forward_1s`: at each site `i > 0` the right isometry is
    peeled off instead of the left one, and the bond tensor is absorbed into
    site `i-1`. Site `0` is evolved last.

    After this call:
    - `mps.center == 0`.
    - `env_right[i]` is populated for `i = 0, …, L-2`.
    """
    L = mps.L
    err_max = 0.0
    w = len(str(L - 1))

    for i in range(L - 1, 0, -1):
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i)

        M, err_site = evolve_1site(mps[i], mpo[i], E_left, E_right, dt, krylov_opts)

        itag = mps._bond_itag(i)
        C, mps[i] = split_site_backward(M, itag)
        env_right[i - 1] = step_right_env(E_right, mps[i], mpo[i])

        C, err_bond = evolve_bond(C, E_left, env_right.fetch(i - 1), -dt, krylov_opts)
        mps[i - 1] = absorb_backward(mps[i - 1], C)
        mps._center = i - 1

        err_max = max(err_max, err_site, err_bond)
        logger.debug("  site %*d / %d  krylov err = %.4e", w, i, L - 1, err_site)
        logger.debug("    bond %d krylov err = %.4e", i, err_bond)

    mps[0], err_site = evolve_1site(
        mps[0], mpo[0], env_left.fetch(0), env_right.fetch(0), dt, krylov_opts
    )

    err_max = max(err_max, err_site)
    logger.debug("  site %*d / %d  krylov err = %.4e", w, 0, L - 1, err_site)

    return err_max


# ---------------------------------------------------------------------------
# 2-site implementations
# ---------------------------------------------------------------------------

def _forward_2s(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    krylov_opts: dict,
    dt: float,
) -> Tuple[float, float]:
    """Forward (left-to-right) half-sweep for 2-site TDVP.

    At each bond `(i, i+1)`: evolve the 2-site tensor Θ forward by `dt`, split
    it so that site `i` is left-isometric, then evolve site `i+1` *backward* by
    `dt`. The back-evolution is a 1-site step rather than a zero-site one — the
    2-site forward step over-counts a whole site, not just a bond. The last
    bond is evolved without a back-evolution, since the chain ends there.

    After this call:
    - `mps.center == mps.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-2`.
    """
    L = mps.L
    err_max = 0.0
    dw = 0.0
    center_bond = L // 2 - 1
    pair_w = 2 * len(str(L - 1)) + 4

    for i in range(L - 1):
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i + 1)

        theta, err_pair = evolve_2site(
            mps[i], mps[i + 1], mpo[i], mpo[i + 1], E_left, E_right, dt, krylov_opts
        )
        err_max = max(err_max, err_pair)
        pair = ("(%d, %d)" % (i, i + 1)).center(pair_w)
        logger.debug("  site %s / %d  krylov err = %.4e", pair, L - 1, err_pair)

        # Measure the discarded weight once, at the mid-chain bond.
        if i == center_bond:
            dw = discarded_weight(theta, trunc)
            logger.debug("    discarded weight = %.4e", dw)

        itag = mps._bond_itag(i + 1)
        mps[i], mps[i + 1] = split_forward(theta, itag, trunc)
        mps._center = i + 1

        # Back-evolve site i+1, unless this was the last bond.
        if i < L - 2:
            env_left[i + 1] = step_left_env(E_left, mps[i], mpo[i])
            mps[i + 1], err_site = evolve_1site(
                mps[i + 1], mpo[i + 1], env_left.fetch(i + 1), E_right, -dt, krylov_opts
            )
            err_max = max(err_max, err_site)
            logger.debug("    back-evolve site %d krylov err = %.4e", i + 1, err_site)

    return err_max, dw


def _backward_2s(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    krylov_opts: dict,
    dt: float,
) -> Tuple[float, float]:
    """Backward (right-to-left) half-sweep for 2-site TDVP.

    Mirror image of `_forward_2s`: Θ is split so that site `i+1` is
    right-isometric, and the 1-site back-evolution acts on site `i`.

    After this call:
    - `mps.center == 0`.
    - `env_right[i]` is populated for `i = 1, …, L-2`.
    """
    L = mps.L
    err_max = 0.0
    dw = 0.0
    center_bond = L // 2 - 1
    pair_w = 2 * len(str(L - 1)) + 4

    for i in range(L - 2, -1, -1):
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i + 1)

        theta, err_pair = evolve_2site(
            mps[i], mps[i + 1], mpo[i], mpo[i + 1], E_left, E_right, dt, krylov_opts
        )
        err_max = max(err_max, err_pair)
        pair = ("(%d, %d)" % (i, i + 1)).center(pair_w)
        logger.debug("  site %s / %d  krylov err = %.4e", pair, L - 1, err_pair)

        if i == center_bond:
            dw = discarded_weight(theta, trunc)
            logger.debug("    discarded weight = %.4e", dw)

        itag = mps._bond_itag(i + 1)
        mps[i], mps[i + 1] = split_backward(theta, itag, trunc)
        mps._center = i

        # Back-evolve site i, unless this was the last bond.
        if i > 0:
            env_right[i] = step_right_env(E_right, mps[i + 1], mpo[i + 1])
            mps[i], err_site = evolve_1site(
                mps[i], mpo[i], E_left, env_right.fetch(i), -dt, krylov_opts
            )
            err_max = max(err_max, err_site)
            logger.debug("    back-evolve site %d krylov err = %.4e", i, err_site)

    return err_max, dw


# ---------------------------------------------------------------------------
# 1-site-plus implementations
# ---------------------------------------------------------------------------

def _forward_1sp(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    krylov_opts: dict,
    dt: float,
    k_expand: int,
    alpha: Optional[int],
) -> Tuple[float, float]:
    """Forward (left-to-right) half-sweep for 1-site-plus TDVP.

    `_forward_1s` with an expansion of bond `(i, i+1)` spliced in before each
    local evolution. At every site `i < L-1`:

    1. `expand_forward` widens the bond by up to `k_expand` complement
       directions, leaving the state unchanged and returning a right
       environment that is exact in the widened basis.
    2. The expanded site tensor is evolved forward by `dt` — a 1-site step,
       but over a tangent space that now reaches into the complement.
    3. The truncating split cuts the bond back to `max_bond`/`trunc_thresh`,
       keeping whichever directions the evolution actually populated.
    4. The exposed bond tensor is evolved *backward* by `dt` and absorbed into
       the expanded neighbor, which returns the stored bond to the truncated
       dimension and moves the center to `i+1`.

    The rightmost site has no neighbor to expand into and is evolved as in
    `'1s'`, without a back-evolution.

    After this call:
    - `mps.center == mps.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-1`.
    """
    L = mps.L
    err_max = 0.0
    dw = 0.0
    center_bond = L // 2 - 1
    w = len(str(L - 1))

    for i in range(L - 1):
        E_left = env_left.fetch(i)

        # Expand bond (i, i+1). env_right.fetch(i+1) spans sites i+2, …, L-1,
        # none of which this sweep has touched yet.
        M_i_exp, M_i1_exp, E_right_exp = expand_forward(
            mps[i], mps[i + 1],
            mpo[i], mpo[i + 1],
            E_left, env_right.fetch(i + 1),
            k_expand, alpha,
        )

        # Forward evolution of the expanded site tensor.
        M, err_site = evolve_1site(
            M_i_exp, mpo[i], E_left, E_right_exp, dt, krylov_opts
        )

        # Measure the discarded weight once, at the mid-chain bond.
        # `discarded_weight` bipartitions its argument into axes (0, 2) against
        # the rest, which for a site tensor is (ket_left, phys) against the
        # right bond — the cut the forward split is about to make.
        if i == center_bond:
            dw = discarded_weight(M, trunc)
            logger.debug("    discarded weight = %.4e", dw)

        # The expansion grew this bond, so the split truncates it back.
        itag = mps._bond_itag(i + 1)
        mps[i], C = split_site_forward(M, itag, trunc)
        env_left[i + 1] = step_left_env(E_left, mps[i], mpo[i])

        # Backward evolution of the bond tensor removes the double counting.
        # C is rectangular here: its left axis carries the truncated bond that
        # env_left[i+1] was just built over, its right axis the expanded bond
        # that E_right_exp lives on.
        C, err_bond = evolve_bond(
            C, env_left.fetch(i + 1), E_right_exp, -dt, krylov_opts
        )
        mps[i + 1] = absorb_forward(C, M_i1_exp)
        mps._center = i + 1

        err_max = max(err_max, err_site, err_bond)
        logger.debug("  site %*d / %d  krylov err = %.4e", w, i, L - 1, err_site)
        logger.debug("    bond %d krylov err = %.4e", i + 1, err_bond)

    # Rightmost site: no bond to its right, so nothing to expand and no
    # back-evolution.
    mps[L - 1], err_site = evolve_1site(
        mps[L - 1], mpo[L - 1], env_left.fetch(L - 1), env_right.fetch(L - 1),
        dt, krylov_opts,
    )

    err_max = max(err_max, err_site)
    logger.debug("  site %*d / %d  krylov err = %.4e", w, L - 1, L - 1, err_site)

    return err_max, dw


def _backward_1sp(
    mps: MPS,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    krylov_opts: dict,
    dt: float,
    k_expand: int,
    alpha: Optional[int],
) -> Tuple[float, float]:
    """Backward (right-to-left) half-sweep for 1-site-plus TDVP.

    Mirror image of `_forward_1sp`: bond `(i-1, i)` is expanded, the left
    isometry gives way to a right one, and the bond tensor is absorbed into
    site `i-1`. Site `0` is evolved last, without an expansion.

    After this call:
    - `mps.center == 0`.
    - `env_right[i]` is populated for `i = 0, …, L-2`.
    """
    L = mps.L
    err_max = 0.0
    dw = 0.0
    center_bond = L // 2 - 1
    w = len(str(L - 1))

    for i in range(L - 1, 0, -1):
        E_right = env_right.fetch(i)

        # Expand bond (i-1, i). env_left.fetch(i-1) spans sites 0, …, i-2,
        # none of which this sweep has touched yet.
        M_i_exp, M_im1_exp, E_left_exp = expand_backward(
            mps[i], mps[i - 1],
            mpo[i], mpo[i - 1],
            env_left.fetch(i - 1), E_right,
            k_expand, alpha,
        )

        # Forward evolution of the expanded site tensor.
        M, err_site = evolve_1site(
            M_i_exp, mpo[i], E_left_exp, E_right, dt, krylov_opts
        )

        # Measure the discarded weight once, at the mid-chain bond. The
        # backward split cuts the left bond off instead of the right one, so
        # the two bonds are swapped before handing the tensor over: that
        # presents the same bipartition from the other side, and hence the
        # same singular values, to a helper that always merges axes (0, 2).
        if i - 1 == center_bond:
            dw = discarded_weight(M.permute([1, 0, 2]), trunc)
            logger.debug("    discarded weight = %.4e", dw)

        # The expansion grew this bond, so the split truncates it back.
        itag = mps._bond_itag(i)
        C, mps[i] = split_site_backward(M, itag, trunc)
        env_right[i - 1] = step_right_env(E_right, mps[i], mpo[i])

        # Backward evolution of the bond tensor removes the double counting.
        # C is rectangular here: its right axis carries the truncated bond that
        # env_right[i-1] was just built over, its left axis the expanded bond
        # that E_left_exp lives on.
        C, err_bond = evolve_bond(
            C, E_left_exp, env_right.fetch(i - 1), -dt, krylov_opts
        )
        mps[i - 1] = absorb_backward(M_im1_exp, C)
        mps._center = i - 1

        err_max = max(err_max, err_site, err_bond)
        logger.debug("  site %*d / %d  krylov err = %.4e", w, i, L - 1, err_site)
        logger.debug("    bond %d krylov err = %.4e", i, err_bond)

    # Leftmost site: no bond to its left, so nothing to expand and no
    # back-evolution.
    mps[0], err_site = evolve_1site(
        mps[0], mpo[0], env_left.fetch(0), env_right.fetch(0), dt, krylov_opts
    )

    err_max = max(err_max, err_site)
    logger.debug("  site %*d / %d  krylov err = %.4e", w, 0, L - 1, err_site)

    return err_max, dw
