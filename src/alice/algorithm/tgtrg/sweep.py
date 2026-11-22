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


"""Half-sweep orchestration for 1-site, 2-site and 1-site-plus TgTRG.

This module is a thin coordination layer. `forward_sweep` and
`backward_sweep` are scheme-aware dispatchers that delegate to
`_forward_1s`, `_backward_1s`, `_forward_2s`, `_backward_2s`,
`_forward_1sp` or `_backward_1sp`; `cool_step` composes them into one
cooling step according to `opts.order`.

Each per-scheme implementation handles only the bookkeeping — which sites or
bonds to visit, where the orthogonality center is, which environment blocks
to refresh — and delegates every local computation to the pure functions in
`scheme_1s`, `scheme_2s` and `complement`.

The 1-site-plus half-sweeps reuse `scheme_1s` for every piece of local
arithmetic, a 1-site-sized update being the same arithmetic whatever space it
is carried out on. That reuse is where the kinship ends. `'1s'` cools on a
manifold of fixed rank, and its sweep is an exact splitting of one fixed
tangent-space projector; `'1sp'` assembles the space it projects onto afresh
at every bond out of the `ρ` it finds there, truncates each bond it has
expanded, and carries the resulting discarded weight. It is a scheme of the
`'2s'` kind whose local exponential happens to be 1-site sized, and the
expectations that make `'1s'` worth having — fixed rank, nothing discarded —
do not transfer to it. The single point where the two coincide is
`expand_k = 0`: the complement is then empty, the rank stops moving, and a
`'1sp'` sweep degenerates to a `'1s'` sweep.

A sweep *is* a cooling step
---------------------------
As in TDVP, and unlike DMRG or XTRG, a sweep here is not an iteration of a
variational search. It advances the inverse temperature by a fixed amount and
is never repeated at the same β, so there is no inner convergence loop and no
`e_tol` analogue. The sweeps take a `tau` argument and return error
diagnostics.

Integrator orders
-----------------
- `order=1` (Lie-Trotter): one half-sweep per step, with the full `tau`. The
  direction alternates between steps, since a half-sweep leaves the center at
  the far end of the chain.
- `order=2` (Lubich symmetric): a forward half-sweep with `tau/2` followed by
  a backward half-sweep with `tau/2`. The step is its own adjoint and the
  center returns to site 0 after every step. This is the default.

Alternating the direction of `order=1` steps makes any *pair* of them
symmetric, so an `order=1` run of `2n` steps at `tau/2` agrees to roundoff
with an `order=2` run of `n` steps at `tau`. What `order=2` buys is where the
accuracy lands: every recorded step ends a complete sweep pair with the
center back at site 0, so the free energy and every observable read off a
second-order accurate density matrix at every β on the grid. For a cooling
run that is close to the whole point — the trajectory *is* the output, not a
means to a final state — so `order=2` is the default and `order=1` is worth
choosing only to halve the work when the grid is sampled sparsely.

Magnitude bookkeeping
---------------------
`exp(-tau H_eff)` is not norm-preserving, and at the step sizes a cooling run
uses the magnitude leaves float64 range entirely. Every local evolution
therefore returns its scale factored into a log (see `krylov`), and the
sweeps accumulate those logs and hand the total to `NormalMPO.scale_by`. The
internal tensor train is then renormalized once per step, so `log_scale`
carries the entire magnitude of `ρ(β)` and the tensors stay O(1) no matter
how cold the run gets. The free energy is read from `log_scale` and
`log_trace`, never from `Tr[ρ]` itself.

Truncation scale
----------------
Every tensor a cooling sweep decomposes sits at norm `1`: the
renormalization above leaves the center there, contracting it with an
isometry preserves that, and the local propagators return `w` at the norm of
their input exactly. The relative `trunc_thresh` and the bare singular values
Nicole's `decomp` compares it against therefore share a scale. Only discarded
weight moves the norm, by `1e-11` on a converged run and by a part in ten
under a bond limit tight enough to cost the free energy its first digit.

Sweeps must be complete
-----------------------
A cooling sweep has to visit every site exactly once — a site skipped is a
site left behind in temperature — so the dispatchers require the center to
sit at the appropriate end of the chain and raise `ValueError` otherwise.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional, Tuple

from alice.network import MPO
from alice.network.thermal import NormalMPO

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
    # `tgtrg` imports this module, so `Options` can only be referenced under
    # TYPE_CHECKING; the future import keeps the annotation unevaluated.
    from .tgtrg import Options

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dispatcher functions
# ---------------------------------------------------------------------------

def cool_step(
    rho: NormalMPO,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    opts: Options,
    tau: float,
    forward: bool = True,
) -> Tuple[float, float]:
    """Cool the density matrix by one step of `tau` in inverse temperature.

    Composes half-sweeps according to `opts.order`, folds the accumulated
    log-magnitude into `rho.log_scale`, and renormalizes the internal tensor
    train so the next step starts from O(1) tensors.

    For `order=2` the `forward` flag is ignored: the symmetric step always
    runs a forward half-sweep and then a backward one, returning the center
    to site 0. For `order=1` a single half-sweep is performed in the
    requested direction, and the caller alternates `forward` between steps.

    Parameters
    ----------
    rho:
        Density matrix to cool in-place.
    mpo:
        Hamiltonian MPO.
    env_left:
        Left environment blocks.
    env_right:
        Right environment blocks.
    opts:
        Cooling run options (scheme, order, truncation, Krylov parameters).
    tau:
        Inverse-temperature increment for this step.
    forward:
        Direction of the half-sweep for `order=1`. Ignored for `order=2`.

    Returns
    -------
    float
        Largest Krylov error estimate over all local evolutions in the step.
    float
        Largest discarded weight at the mid-chain bond over the step (`'2s'`
        and `'1sp'`; `0.0` for `'1s'`, which never truncates).
    """
    if opts.order == 2:
        half = 0.5 * tau
        err_f, dw_f, log_f = forward_sweep(rho, mpo, env_left, env_right, opts, half)
        err_b, dw_b, log_b = backward_sweep(rho, mpo, env_left, env_right, opts, half)
        err, dw, log_delta = max(err_f, err_b), max(dw_f, dw_b), log_f + log_b
    elif opts.order == 1:
        sweep = forward_sweep if forward else backward_sweep
        err, dw, log_delta = sweep(rho, mpo, env_left, env_right, opts, tau)
    else:
        raise NotImplementedError(
            f"cool_step: unsupported integrator order {opts.order!r}"
        )

    rho.scale_by(log_delta + _renormalize(rho))

    return err, dw


def forward_sweep(
    rho: NormalMPO,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    opts: Options,
    tau: float,
) -> Tuple[float, float, float]:
    """Perform a left-to-right (forward) half-sweep over `tau`.

    Dispatches to the 1-site or 2-site implementation based on `opts.scheme`.

    Before this call `rho.center` must be `0`. After it:

    - `rho.center == rho.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-1` (`'1s'`, `'1sp'`) or
      `i = 1, …, L-2` (`'2s'`).

    Parameters
    ----------
    rho:
        Density matrix to cool in-place. Must have `center == 0`.
    mpo:
        Hamiltonian MPO.
    env_left:
        Left environment blocks. `env_left[0]` must be initialized.
    env_right:
        Right environment blocks. All slots must be initialized before the
        first call (populated by `build_right_envs`).
    opts:
        Cooling run options.
    tau:
        Inverse-temperature increment for this half-sweep: `tau` for
        `order=1`, `tau / 2` for `order=2`.

    Returns
    -------
    float
        Largest Krylov error estimate over the half-sweep.
    float
        Discarded weight at the mid-chain bond (`'2s'` and `'1sp'`; `0.0` for
        `'1s'`).
    float
        Accumulated log-magnitude removed by the local propagators, for the
        caller to fold into `rho.log_scale`.

    Raises
    ------
    ValueError
        If `rho.center` is not `0`, which would leave part of the chain
        uncooled.
    """
    if rho.center != 0:
        raise ValueError(
            f"forward_sweep requires the orthogonality center at site 0, "
            f"found {rho.center}; a cooling sweep must visit every site"
        )

    trunc, krylov_opts, cbe_opts = _unpack_opts(opts)
    if opts.scheme == '1s':
        err, log_delta = _forward_1s(rho, mpo, env_left, env_right, krylov_opts, tau)
        return err, 0.0, log_delta
    if opts.scheme == '2s':
        return _forward_2s(rho, mpo, env_left, env_right, trunc, krylov_opts, tau)
    if opts.scheme == '1sp':
        return _forward_1sp(rho, mpo, env_left, env_right, trunc, krylov_opts, tau,
                            **cbe_opts)
    raise NotImplementedError(f"forward_sweep: unknown scheme {opts.scheme!r}")


def backward_sweep(
    rho: NormalMPO,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    opts: Options,
    tau: float,
) -> Tuple[float, float, float]:
    """Perform a right-to-left (backward) half-sweep over `tau`.

    Dispatches to the 1-site or 2-site implementation based on `opts.scheme`.

    Before this call `rho.center` must be `rho.L - 1`. After it:

    - `rho.center == 0`.
    - `env_right[i]` is populated for `i = 0, …, L-2` (`'1s'`, `'1sp'`) or
      `i = 1, …, L-2` (`'2s'`).

    Parameters
    ----------
    rho:
        Density matrix to cool in-place. Must have `center == rho.L - 1`.
    mpo:
        Hamiltonian MPO.
    env_left:
        Left environment blocks. All slots must be populated from a prior
        `forward_sweep`.
    env_right:
        Right environment blocks. `env_right[rho.L - 1]` must be initialized.
    opts:
        Cooling run options.
    tau:
        Inverse-temperature increment for this half-sweep.

    Returns
    -------
    float
        Largest Krylov error estimate over the half-sweep.
    float
        Discarded weight at the mid-chain bond (`'2s'` and `'1sp'`; `0.0` for
        `'1s'`).
    float
        Accumulated log-magnitude removed by the local propagators.

    Raises
    ------
    ValueError
        If `rho.center` is not `rho.L - 1`.
    """
    if rho.center != rho.L - 1:
        raise ValueError(
            f"backward_sweep requires the orthogonality center at site "
            f"{rho.L - 1}, found {rho.center}; a cooling sweep must visit "
            f"every site"
        )

    trunc, krylov_opts, cbe_opts = _unpack_opts(opts)
    if opts.scheme == '1s':
        err, log_delta = _backward_1s(rho, mpo, env_left, env_right, krylov_opts, tau)
        return err, 0.0, log_delta
    if opts.scheme == '2s':
        return _backward_2s(rho, mpo, env_left, env_right, trunc, krylov_opts, tau)
    if opts.scheme == '1sp':
        return _backward_1sp(rho, mpo, env_left, env_right, trunc, krylov_opts, tau,
                             **cbe_opts)
    raise NotImplementedError(f"backward_sweep: unknown scheme {opts.scheme!r}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _unpack_opts(opts: Options) -> Tuple[Optional[dict], dict, dict]:
    """Build the truncation, Krylov and CBE keyword-args from `opts`.

    `trunc_thresh` passes straight through to Nicole's `thresh`: the sweeps
    hold every decomposed tensor at norm `1`, so the relative threshold and
    the bare singular values share a scale. See the module docstring.
    """
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


def _renormalize(rho: NormalMPO) -> float:
    """Divide the norm out of the center tensor and return its log.

    In canonical form the center tensor carries the whole Frobenius norm of
    the internal train, so this is a one-tensor operation. Keeping that norm
    at `1` is how a run reaches low temperature at all: the magnitude of
    `ρ(β)` grows without bound, and only `log_scale` can hold it.

    Parameters
    ----------
    rho:
        Density matrix with a valid `center`, modified in-place.

    Returns
    -------
    float
        `log` of the norm removed, to be added to `rho.log_scale`. Zero when
        the center tensor is already normalized or has no norm to speak of.
    """
    center = rho.center
    norm = rho[center].norm()
    if norm == 0.0 or not math.isfinite(norm):
        return 0.0
    rho[center] = rho[center] * (1.0 / norm)

    return math.log(norm)


# ---------------------------------------------------------------------------
# 1-site implementations
# ---------------------------------------------------------------------------

def _forward_1s(
    rho: NormalMPO,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    krylov_opts: dict,
    tau: float,
) -> Tuple[float, float]:
    """Forward (left-to-right) half-sweep for 1-site TgTRG.

    At each site `i < L-1`: cool the site tensor by `tau`, split off the left
    isometry, update the left environment, evolve the exposed bond tensor
    *backward* by `tau`, and absorb it into site `i+1`. The rightmost site is
    cooled last; it has no bond to its right to correct.

    After this call:

    - `rho.center == rho.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-1`.

    Returns
    -------
    float
        Largest Krylov error estimate over the half-sweep.
    float
        Accumulated log-magnitude removed by the local propagators.
    """
    L = rho.L
    err_max = 0.0
    log_delta = 0.0
    w = len(str(L - 1))

    for i in range(L - 1):
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i)

        # Cool the site tensor.
        M, log_site, err_site = evolve_1site(
            rho[i], mpo[i], E_left, E_right, tau, krylov_opts
        )

        # Peel off the isometry; the bond tensor is the double-counted piece.
        itag = rho._bond_itag(i + 1)
        rho[i], C = split_site_forward(M, itag)
        env_left[i + 1] = step_left_env(E_left, rho[i], mpo[i])

        # Backward evolution of the bond tensor removes the double counting.
        # env_right[i] spans sites i+1, …, L-1, which is exactly the right
        # half of this bond, and is still valid: site i+1 is untouched.
        C, log_bond, err_bond = evolve_bond(
            C, env_left.fetch(i + 1), E_right, -tau, krylov_opts
        )
        rho[i + 1] = absorb_forward(C, rho[i + 1])
        rho._center = i + 1

        err_max = max(err_max, err_site, err_bond)
        log_delta += log_site + log_bond
        logger.debug("  site %*d / %d  krylov err = %.4e", w, i, L - 1, err_site)
        logger.debug("    bond %d krylov err = %.4e", i + 1, err_bond)

    # Rightmost site: no bond to its right, so no back-evolution.
    rho[L - 1], log_site, err_site = evolve_1site(
        rho[L - 1], mpo[L - 1], env_left.fetch(L - 1), env_right.fetch(L - 1),
        tau, krylov_opts,
    )

    err_max = max(err_max, err_site)
    log_delta += log_site
    logger.debug("  site %*d / %d  krylov err = %.4e", w, L - 1, L - 1, err_site)

    return err_max, log_delta


def _backward_1s(
    rho: NormalMPO,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    krylov_opts: dict,
    tau: float,
) -> Tuple[float, float]:
    """Backward (right-to-left) half-sweep for 1-site TgTRG.

    Mirror image of `_forward_1s`: at each site `i > 0` the right isometry is
    peeled off instead of the left one, and the bond tensor is absorbed into
    site `i-1`. Site `0` is cooled last.

    After this call:

    - `rho.center == 0`.
    - `env_right[i]` is populated for `i = 0, …, L-2`.

    Returns
    -------
    float
        Largest Krylov error estimate over the half-sweep.
    float
        Accumulated log-magnitude removed by the local propagators.
    """
    L = rho.L
    err_max = 0.0
    log_delta = 0.0
    w = len(str(L - 1))

    for i in range(L - 1, 0, -1):
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i)

        M, log_site, err_site = evolve_1site(
            rho[i], mpo[i], E_left, E_right, tau, krylov_opts
        )

        itag = rho._bond_itag(i)
        C, rho[i] = split_site_backward(M, itag)
        env_right[i - 1] = step_right_env(E_right, rho[i], mpo[i])

        C, log_bond, err_bond = evolve_bond(
            C, E_left, env_right.fetch(i - 1), -tau, krylov_opts
        )
        rho[i - 1] = absorb_backward(rho[i - 1], C)
        rho._center = i - 1

        err_max = max(err_max, err_site, err_bond)
        log_delta += log_site + log_bond
        logger.debug("  site %*d / %d  krylov err = %.4e", w, i, L - 1, err_site)
        logger.debug("    bond %d krylov err = %.4e", i, err_bond)

    rho[0], log_site, err_site = evolve_1site(
        rho[0], mpo[0], env_left.fetch(0), env_right.fetch(0), tau, krylov_opts
    )

    err_max = max(err_max, err_site)
    log_delta += log_site
    logger.debug("  site %*d / %d  krylov err = %.4e", w, 0, L - 1, err_site)

    return err_max, log_delta


# ---------------------------------------------------------------------------
# 2-site implementations
# ---------------------------------------------------------------------------

def _forward_2s(
    rho: NormalMPO,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    krylov_opts: dict,
    tau: float,
) -> Tuple[float, float, float]:
    """Forward (left-to-right) half-sweep for 2-site TgTRG.

    At each bond `(i, i+1)`: cool the 2-site tensor Θ by `tau`, split it so
    that site `i` is left-isometric, then evolve site `i+1` *backward* by
    `tau`. The back-evolution is a 1-site step instead of a zero-site one —
    the 2-site forward step over-counts a whole site, not just a bond. The
    last bond takes no back-evolution, the chain ending there.

    After this call:

    - `rho.center == rho.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-2`.

    Returns
    -------
    float
        Largest Krylov error estimate over the half-sweep.
    float
        Discarded weight at the mid-chain bond.
    float
        Accumulated log-magnitude removed by the local propagators.
    """
    L = rho.L
    err_max = 0.0
    dw = 0.0
    log_delta = 0.0
    center_bond = L // 2 - 1
    pair_w = 2 * len(str(L - 1)) + 4

    for i in range(L - 1):
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i + 1)

        theta, log_pair, err_pair = evolve_2site(
            rho[i], rho[i + 1], mpo[i], mpo[i + 1], E_left, E_right, tau,
            krylov_opts,
        )
        err_max = max(err_max, err_pair)
        log_delta += log_pair
        pair = f"({i}, {i + 1})".center(pair_w)
        logger.debug("  site %s / %d  krylov err = %.4e", pair, L - 1, err_pair)

        if i == center_bond:
            dw = discarded_weight(theta, trunc)
            logger.debug("    discarded weight = %.4e", dw)

        itag = rho._bond_itag(i + 1)
        rho[i], rho[i + 1] = split_forward(theta, itag, trunc)
        rho._center = i + 1

        # Back-evolve site i+1, unless this was the last bond.
        if i < L - 2:
            env_left[i + 1] = step_left_env(E_left, rho[i], mpo[i])
            rho[i + 1], log_site, err_site = evolve_1site(
                rho[i + 1], mpo[i + 1], env_left.fetch(i + 1), E_right, -tau,
                krylov_opts,
            )
            err_max = max(err_max, err_site)
            log_delta += log_site
            logger.debug("    back-evolve site %d krylov err = %.4e", i + 1,
                         err_site)

    return err_max, dw, log_delta


def _backward_2s(
    rho: NormalMPO,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    krylov_opts: dict,
    tau: float,
) -> Tuple[float, float, float]:
    """Backward (right-to-left) half-sweep for 2-site TgTRG.

    Mirror image of `_forward_2s`: Θ is split so that site `i+1` is
    right-isometric, and the 1-site back-evolution acts on site `i`.

    After this call:

    - `rho.center == 0`.
    - `env_right[i]` is populated for `i = 1, …, L-2`.

    Returns
    -------
    float
        Largest Krylov error estimate over the half-sweep.
    float
        Discarded weight at the mid-chain bond.
    float
        Accumulated log-magnitude removed by the local propagators.
    """
    L = rho.L
    err_max = 0.0
    dw = 0.0
    log_delta = 0.0
    center_bond = L // 2 - 1
    pair_w = 2 * len(str(L - 1)) + 4

    for i in range(L - 2, -1, -1):
        E_left = env_left.fetch(i)
        E_right = env_right.fetch(i + 1)

        theta, log_pair, err_pair = evolve_2site(
            rho[i], rho[i + 1], mpo[i], mpo[i + 1], E_left, E_right, tau,
            krylov_opts,
        )
        err_max = max(err_max, err_pair)
        log_delta += log_pair
        pair = f"({i}, {i + 1})".center(pair_w)
        logger.debug("  site %s / %d  krylov err = %.4e", pair, L - 1, err_pair)

        if i == center_bond:
            dw = discarded_weight(theta, trunc)
            logger.debug("    discarded weight = %.4e", dw)

        itag = rho._bond_itag(i + 1)
        rho[i], rho[i + 1] = split_backward(theta, itag, trunc)
        rho._center = i

        # Back-evolve site i, unless this was the last bond.
        if i > 0:
            env_right[i] = step_right_env(E_right, rho[i + 1], mpo[i + 1])
            rho[i], log_site, err_site = evolve_1site(
                rho[i], mpo[i], E_left, env_right.fetch(i), -tau, krylov_opts
            )
            err_max = max(err_max, err_site)
            log_delta += log_site
            logger.debug("    back-evolve site %d krylov err = %.4e", i, err_site)

    return err_max, dw, log_delta


# ---------------------------------------------------------------------------
# 1-site-plus implementations
# ---------------------------------------------------------------------------

def _forward_1sp(
    rho: NormalMPO,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    krylov_opts: dict,
    tau: float,
    k_expand: int,
    alpha: Optional[int],
) -> Tuple[float, float, float]:
    """Forward (left-to-right) half-sweep for 1-site-plus TgTRG.

    At every site `i < L-1`:

    1. `expand_forward` expands bond `(i, i+1)` by up to `k_expand`
       complement directions, leaving `ρ` unchanged and returning a right
       environment that is exact in the expanded basis.
    2. The expanded site tensor is cooled by `tau`. The propagator is 1-site
       sized, but the space it acts in was built from the current `ρ` at this
       bond and is neither the fixed-rank tangent space of `'1s'` nor the
       same space the neighboring bonds will use.
    3. The truncating split cuts the bond back to `max_bond`/`trunc_thresh`,
       keeping whichever directions the cooling actually populated.
    4. The exposed bond tensor is evolved *backward* by `tau` and absorbed
       into the expanded neighbor, which returns the stored bond to the
       truncated dimension and moves the center to `i+1`. Unlike the
       zero-site step of `'1s'`, which removes the double counting of one
       bond space, this one spans two: `C` is rectangular, the truncated bond
       on one axis and the expanded bond on the other.

    The rightmost site has no neighbor to expand into, so it is cooled on the
    bond it already has and takes no back-evolution.

    After this call:

    - `rho.center == rho.L - 1`.
    - `env_left[i]` is populated for `i = 1, …, L-1`.

    Returns
    -------
    float
        Largest Krylov error estimate over the half-sweep.
    float
        Discarded weight at the mid-chain bond.
    float
        Accumulated log-magnitude removed by the local propagators.
    """
    L = rho.L
    err_max = 0.0
    dw = 0.0
    log_delta = 0.0
    center_bond = L // 2 - 1
    w = len(str(L - 1))

    for i in range(L - 1):
        E_left = env_left.fetch(i)

        # Expand bond (i, i+1). env_right.fetch(i+1) spans sites i+2, …, L-1,
        # none of which this sweep has touched yet.
        M_i_exp, M_i1_exp, E_right_exp = expand_forward(
            rho[i], rho[i + 1],
            mpo[i], mpo[i + 1],
            E_left, env_right.fetch(i + 1),
            k_expand, alpha,
        )

        # Cool the expanded site tensor.
        M, log_site, err_site = evolve_1site(
            M_i_exp, mpo[i], E_left, E_right_exp, tau, krylov_opts
        )

        # Measure the discarded weight once, at the mid-chain bond.
        # `discarded_weight` bipartitions its argument at axes (0, 2, 3),
        # which for a site tensor is (ket_left, phys_in, phys_out) against the
        # right bond — the cut the forward split is about to make.
        if i == center_bond:
            dw = discarded_weight(M, trunc)
            logger.debug("    discarded weight = %.4e", dw)

        # The expansion grew this bond, so the split truncates it back.
        itag = rho._bond_itag(i + 1)
        rho[i], C = split_site_forward(M, itag, trunc)
        env_left[i + 1] = step_left_env(E_left, rho[i], mpo[i])

        # Backward evolution of the bond tensor removes the double counting.
        # C is rectangular here: its left axis carries the truncated bond that
        # env_left[i+1] was just built over, its right axis the expanded bond
        # that E_right_exp lives on.
        C, log_bond, err_bond = evolve_bond(
            C, env_left.fetch(i + 1), E_right_exp, -tau, krylov_opts
        )
        rho[i + 1] = absorb_forward(C, M_i1_exp)
        rho._center = i + 1

        err_max = max(err_max, err_site, err_bond)
        log_delta += log_site + log_bond
        logger.debug("  site %*d / %d  krylov err = %.4e", w, i, L - 1, err_site)
        logger.debug("    bond %d krylov err = %.4e", i + 1, err_bond)

    # Rightmost site: no bond to its right, so nothing to expand and no
    # back-evolution.
    rho[L - 1], log_site, err_site = evolve_1site(
        rho[L - 1], mpo[L - 1], env_left.fetch(L - 1), env_right.fetch(L - 1),
        tau, krylov_opts,
    )

    err_max = max(err_max, err_site)
    log_delta += log_site
    logger.debug("  site %*d / %d  krylov err = %.4e", w, L - 1, L - 1, err_site)

    return err_max, dw, log_delta


def _backward_1sp(
    rho: NormalMPO,
    mpo: MPO,
    env_left: Environment,
    env_right: Environment,
    trunc: Optional[dict],
    krylov_opts: dict,
    tau: float,
    k_expand: int,
    alpha: Optional[int],
) -> Tuple[float, float, float]:
    """Backward (right-to-left) half-sweep for 1-site-plus TgTRG.

    Mirror image of `_forward_1sp`: bond `(i-1, i)` is expanded, the left
    isometry gives way to a right one, and the bond tensor is absorbed into
    site `i-1`. Site `0` is cooled last, without an expansion.

    After this call:

    - `rho.center == 0`.
    - `env_right[i]` is populated for `i = 0, …, L-2`.

    Returns
    -------
    float
        Largest Krylov error estimate over the half-sweep.
    float
        Discarded weight at the mid-chain bond.
    float
        Accumulated log-magnitude removed by the local propagators.
    """
    L = rho.L
    err_max = 0.0
    dw = 0.0
    log_delta = 0.0
    center_bond = L // 2 - 1
    w = len(str(L - 1))

    for i in range(L - 1, 0, -1):
        E_right = env_right.fetch(i)

        # Expand bond (i-1, i). env_left.fetch(i-1) spans sites 0, …, i-2,
        # none of which this sweep has touched yet.
        M_i_exp, M_im1_exp, E_left_exp = expand_backward(
            rho[i], rho[i - 1],
            mpo[i], mpo[i - 1],
            env_left.fetch(i - 1), E_right,
            k_expand, alpha,
        )

        # Cool the expanded site tensor.
        M, log_site, err_site = evolve_1site(
            M_i_exp, mpo[i], E_left_exp, E_right, tau, krylov_opts
        )

        # Measure the discarded weight once, at the mid-chain bond. The
        # backward split cuts the left bond off instead of the right one, so
        # the two bonds are swapped before handing the tensor over: that
        # presents the same bipartition from the other side, and hence the
        # same singular values, to a helper that always merges axes (0, 2, 3).
        if i - 1 == center_bond:
            dw = discarded_weight(M.permute([1, 0, 2, 3]), trunc)
            logger.debug("    discarded weight = %.4e", dw)

        # The expansion grew this bond, so the split truncates it back.
        itag = rho._bond_itag(i)
        C, rho[i] = split_site_backward(M, itag, trunc)
        env_right[i - 1] = step_right_env(E_right, rho[i], mpo[i])

        # Backward evolution of the bond tensor removes the double counting.
        # C is rectangular here: its right axis carries the truncated bond
        # that env_right[i-1] was just built over, its left axis the expanded
        # bond that E_left_exp lives on.
        C, log_bond, err_bond = evolve_bond(
            C, E_left_exp, env_right.fetch(i - 1), -tau, krylov_opts
        )
        rho[i - 1] = absorb_backward(M_im1_exp, C)
        rho._center = i - 1

        err_max = max(err_max, err_site, err_bond)
        log_delta += log_site + log_bond
        logger.debug("  site %*d / %d  krylov err = %.4e", w, i, L - 1, err_site)
        logger.debug("    bond %d krylov err = %.4e", i, err_bond)

    # Leftmost site: no bond to its left, so nothing to expand and no
    # back-evolution.
    rho[0], log_site, err_site = evolve_1site(
        rho[0], mpo[0], env_left.fetch(0), env_right.fetch(0), tau, krylov_opts
    )

    err_max = max(err_max, err_site)
    log_delta += log_site
    logger.debug("  site %*d / %d  krylov err = %.4e", w, 0, L - 1, err_site)

    return err_max, dw, log_delta
