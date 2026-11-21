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


"""Example: TgTRG finite-temperature thermodynamics of the 1D spinless free-fermion chain.

Cools ρ(β) = e^{-βH} along the tangent space of the MPO manifold for the
tight-binding Hamiltonian

    H = -t Σ_i (c†_i c_{i+1} + h.c.) - μ Σ_i n_i

on a spinless fermion chain of length L with open boundary conditions, then
compares log Z(β), the free energy, internal energy, specific heat, and
entropy against the exact grand-canonical solution at each cooling step.

Single-particle energies (OBC):

    ε_k = -2t cos(kπ / (L+1)),  k = 1, …, L.

Exact grand-canonical log Z (shifted by the chemical potential μ):

    log Z(β) = Σ_k log(1 + exp(-β (ε_k - μ))).

The default schedule is the stacked one: an exponential phase that doubles β
from τ₀ up to `--beta-switch`, followed by a linear phase in steps of
`--tau`. The exponential phase costs a handful of sweeps to get out of the
high-temperature regime, where the Taylor state lives; the linear phase then
takes the chain cold in controlled steps. Stacking matters because the
exponential grid's step size *is* β, and a step much past τ ΔE ~ 1 amplifies
roundoff through the backward bond update — see the `tgtrg` module docstring.
Pass `--beta-switch 0` for a purely exponential run, or `--lin-steps 0` for
the exponential phase alone.

Run from the repository root:

    uv run python examples/examples_tgtrg/tgtrg_spinless.py
    uv run python examples/examples_tgtrg/tgtrg_spinless.py --scheme 1s
    uv run python examples/examples_tgtrg/tgtrg_spinless.py --order 1
    uv run python examples/examples_tgtrg/tgtrg_spinless.py --beta-switch 0 --steps 11
    uv run python examples/examples_tgtrg/tgtrg_spinless.py --tau 0.25 --lin-steps 40
    uv run python examples/examples_tgtrg/tgtrg_spinless.py --symmetry Z2 --max-bond 32
    uv run python examples/examples_tgtrg/tgtrg_spinless.py --obs-exact
    uv run python examples/examples_tgtrg/tgtrg_spinless.py --mu 0.5
    uv run python examples/examples_tgtrg/tgtrg_spinless.py --env-cache-dir /tmp/env_cache
    uv run python examples/examples_tgtrg/tgtrg_spinless.py --checkpoint-dir /tmp/tgtrg_ckpt
    uv run python examples/examples_tgtrg/tgtrg_spinless.py --help
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import alice
from alice.network import build_hamiltonian, build_interaction
from alice.network.thermal import thermal_mpo
from alice.algorithm import tgtrg


# ---------------------------------------------------------------------------
# Exact reference
# ---------------------------------------------------------------------------

def single_particle_energies(L: int, t: float) -> List[float]:
    """Single-particle spectrum of the OBC tight-binding chain.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.

    Returns
    -------
    list of float
        `ε_k = -2t cos(kπ / (L+1))` for `k = 1, …, L`.
    """
    return [-2.0 * t * math.cos(k * math.pi / (L + 1)) for k in range(1, L + 1)]


def exact_log_z(L: int, t: float, mu: float, beta: float) -> float:
    """Exact grand-canonical log Z for a spinless OBC free-fermion chain.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.
    mu:
        Chemical potential.
    beta:
        Inverse temperature.

    Returns
    -------
    float
        `log Z(β) = Σ_k log(1 + exp(-β (ε_k - μ)))`.
    """
    return sum(math.log1p(math.exp(-beta * (e - mu)))
               for e in single_particle_energies(L, t))


def exact_energy(L: int, t: float, mu: float, beta: float) -> float:
    """Exact internal energy per site, `u = -d log Z / dβ / L`.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.
    mu:
        Chemical potential.
    beta:
        Inverse temperature.

    Returns
    -------
    float
        `u(β) = Σ_k x_k / (1 + exp(β x_k)) / L` with `x_k = ε_k - μ`.
    """
    xs = [e - mu for e in single_particle_energies(L, t)]
    return sum(x / (1.0 + math.exp(beta * x)) for x in xs) / L


def exact_specific_heat(L: int, t: float, mu: float, beta: float) -> float:
    """Exact specific heat per site, `c_V = -β² du/dβ`.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.
    mu:
        Chemical potential.
    beta:
        Inverse temperature.

    Returns
    -------
    float
        `c_V(β) = β² Σ_k x_k² / (4 cosh²(β x_k / 2)) / L`.
    """
    xs = [e - mu for e in single_particle_energies(L, t)]
    return beta ** 2 * sum(
        (x / (2.0 * math.cosh(0.5 * beta * x))) ** 2 for x in xs
    ) / L


def exact_entropy(L: int, t: float, mu: float, beta: float) -> float:
    """Exact entropy per site, `S = β (u - f)`.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.
    mu:
        Chemical potential.
    beta:
        Inverse temperature.

    Returns
    -------
    float
        `S(β) = β u(β) + log Z(β) / L`.
    """
    return (beta * exact_energy(L, t, mu, beta)
            + exact_log_z(L, t, mu, beta) / L)


# ---------------------------------------------------------------------------
# Main TgTRG function
# ---------------------------------------------------------------------------

def tgtrg_spinless(
    L: int = 8,
    t: float = 1.0,
    mu: float = 0.0,
    symmetry: str = 'U1',
    scheme: str = '2s',
    order: int = 2,
    tau0: float = 2 ** -12,
    beta_switch: float = 2.0,
    tau: float = 0.5,
    lin_steps: int = 20,
    steps: Optional[int] = None,
    taylor_order: int = 10,
    max_bond: Optional[int] = None,
    trunc_thresh: float = 1e-12,
    krylov_dim: int = 16,
    krylov_tol: float = 1e-12,
    obs_exact: bool = False,
    env_cache_dir: Optional[str] = None,
    env_async_io: bool = True,
    env_window: int = 2,
    checkpoint_dir: Optional[str] = None,
    save_artifacts: bool = True,
    save_artifacts_every: int = 1,
    verbose: bool = True,
) -> Tuple[tgtrg.Summary, tgtrg.Artifact]:
    """Cool the spinless free-fermion tight-binding chain with TgTRG.

    Runs the stacked schedule by default: an exponential phase up to
    `beta_switch`, then `lin_steps` linear steps of size `tau`. Each phase
    is one `tgtrg.run()` call, the second resuming from the `Artifact` the
    first returned, which is all a stacked schedule takes.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.
    mu:
        Chemical potential.
    symmetry:
        Symmetry exploited: `'U1'` (conserve particle number, default) or
        `'Z2'` (fermion parity only).
    scheme:
        Tangent-space update scheme: `'1s'` (1-site, fixed bond dimension)
        or `'2s'` (2-site, default, adaptive bond dimension).
    order:
        Integrator order per step: `1` for a single forward half-sweep, `2`
        (default) for the symmetric forward-then-backward pair.
    tau0:
        Initial inverse temperature τ₀. Should be small enough that the
        Taylor expansion of `e^{-τ₀ H}` converges.
    beta_switch:
        Inverse temperature at which the exponential phase hands over to
        the linear one. `0` skips the exponential phase entirely and cools
        linearly from τ₀.
    tau:
        Step size of the linear phase.
    lin_steps:
        Number of linear steps after the handover. `0` stops at
        `beta_switch`.
    steps:
        Explicit number of exponential steps, overriding `beta_switch`.
        `None` (default) derives it from `beta_switch`.
    taylor_order:
        Truncation order of the Taylor series for `ρ(τ₀) = e^{-τ₀ H}`.
    max_bond:
        Maximum bond dimension of ρ. `None` (default) means unlimited. For
        `'1s'` it fixes the rank by compacting the Taylor state, since the
        1-site scheme cannot change a bond dimension itself.
    trunc_thresh:
        Relative SVD truncation threshold (default: `1e-12`).
    krylov_dim:
        Maximum Krylov dimension per local propagator (default: 16).
    krylov_tol:
        Krylov convergence tolerance (default: `1e-12`).
    obs_exact:
        If `True`, measure `u = Tr[ρH] / (Tr[ρ] L)` directly at each step
        in addition to the finite-difference estimate. Off by default; the
        two columns together measure the grid's discretization error.
    env_cache_dir:
        Directory for environment block cache files. `None` (default) keeps
        all blocks in memory.
    env_async_io:
        Enable asynchronous disk I/O for environment caching (default
        `True`). Has no effect when `env_cache_dir` is `None`.
    env_window:
        Sliding-window size for in-memory environment blocks (default `2`).
        Has no effect when `env_cache_dir` is `None`.
    checkpoint_dir:
        Directory for `thermal.ckpt`, mid-run `tgtrg.ckpt`, and optional
        `artifacts/` archives. `None` (default) writes to the current
        working directory.
    save_artifacts:
        If `True` (default), archive per-step density matrices under
        `artifacts/step_XXXXX.ckpt`.
    save_artifacts_every:
        Archive stride in steps when `save_artifacts` is `True` (default:
        `1`, every step).
    verbose:
        Print configuration and the β-by-β thermodynamics table when
        `True`.

    Returns
    -------
    Summary
        Thermodynamic history over both phases.
    Artifact
        Final density matrix `ρ(β_max)`.

    Raises
    ------
    ValueError
        If `symmetry` is neither `'U1'` nor `'Z2'`, or if the schedule has
        no steps at all.
    """
    if symmetry not in ('U1', 'Z2'):
        raise ValueError(f"symmetry must be 'U1' or 'Z2', got {symmetry!r}")

    # The exponential grid lands exactly on β = 2^n τ₀, so the handover
    # point picks out a step count, not an arbitrary β.
    if steps is None:
        steps = (0 if beta_switch <= tau0
                 else max(0, math.ceil(math.log2(beta_switch / tau0))))
    if steps == 0 and lin_steps == 0:
        raise ValueError(
            "the schedule is empty: give a beta_switch above tau0, a "
            "positive steps, or a positive lin_steps"
        )

    # -----------------------------------------------------------------------
    # Build the Hamiltonian MPO
    # -----------------------------------------------------------------------
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': L, 'bcx': 'OBC', 'n2x': True},
        'model': {
            'category': 'fermionic',
            'label': 'FreeFermion',
            'symmetry': symmetry,
            't': t,
            'mu': mu,
        },
    }
    interactions, spc, geo = build_interaction(cfg)
    H = build_hamiltonian(interactions, geo.L, spc)

    beta_exp = tau0 * 2 ** steps
    beta_max = beta_exp + tau * lin_steps
    max_bond_str = str(max_bond) if max_bond is not None else 'unlimited'
    if verbose:
        print(f"Chain length   : {geo.L}")
        print(f"Symmetry       : {symmetry}")
        print(f"Hopping t      : {t}")
        print(f"Chemical pot μ : {mu}")
        print(f"Scheme         : {scheme}   order = {order}")
        print(f"MPO bond dims  : {H.bond_dims}")
        print(f"tau_0          : {tau0:.6g}   taylor_order = {taylor_order}")
        print(f"Exponential    : {steps} steps   -> beta = {beta_exp:.6g}")
        print(f"Linear         : {lin_steps} steps of tau = {tau:.6g}   "
              f"-> beta = {beta_max:.6g}")
        print(f"Max bond dim   : {max_bond_str}   "
              f"trunc_thresh = {trunc_thresh:.2e}")
        print(f"Krylov         : dim = {krylov_dim}   tol = {krylov_tol:.2e}")
        if obs_exact:
            print("Observables    : finite differences + direct Tr[ρH]")
        if env_cache_dir is not None:
            print(f"Env cache dir  : {env_cache_dir}  "
                  f"(window={env_window}, async={env_async_io})")
        if checkpoint_dir is not None:
            print(f"Checkpoint dir : {checkpoint_dir}")
        if save_artifacts:
            print(f"Artifacts      : every {save_artifacts_every} step(s)")
        print()

    shared = dict(
        scheme=scheme,
        order=order,
        tau_0=tau0,
        taylor_order=taylor_order,
        max_bond=max_bond,
        trunc_thresh=trunc_thresh,
        krylov_dim=krylov_dim,
        krylov_tol=krylov_tol,
        obs_exact=obs_exact,
        env_cache_dir=env_cache_dir,
        env_async_io=env_async_io,
        env_window=env_window,
        checkpoint_dir=checkpoint_dir,
        save_artifacts=save_artifacts,
        save_artifacts_every=save_artifacts_every,
    )

    # -----------------------------------------------------------------------
    # Phase 1: exponential cooling out of the high-temperature regime
    # -----------------------------------------------------------------------
    opts = tgtrg.Options(grid='exp', n_steps=steps, **shared)
    rho0 = thermal_mpo(H, tau0, taylor_order, spc)
    state = tgtrg.Artifact(rho=rho0, beta=tau0, step=0)
    summary, artifact = tgtrg.run(state, H, opts)

    # -----------------------------------------------------------------------
    # Phase 2: linear cooling to the target temperature
    # -----------------------------------------------------------------------
    if lin_steps > 0:
        if verbose:
            print()
        opts = tgtrg.Options(grid='lin', tau=tau,
                             n_steps=steps + lin_steps, **shared)
        summary, artifact = tgtrg.run(artifact, H, opts)

    # -----------------------------------------------------------------------
    # Print thermodynamics comparison table
    # -----------------------------------------------------------------------
    if verbose:
        print()
        header = (
            f"{'beta':>10}  {'T':>8}  {'log_Z':>14}  {'log_Z_exact':>14}  "
            f"{'rel_err':>9}  {'f':>10}  {'f_exact':>10}  "
            f"{'u':>10}  {'u_exact':>10}  {'c_V':>9}  {'c_V_exact':>9}  "
            f"{'S':>9}  {'S_exact':>9}  {'D':>5}"
        )
        if obs_exact:
            header += f"  {'u_direct':>10}"
        print(header)
        print("-" * len(header))

        for n, beta in enumerate(summary.betas):
            lz_exact = exact_log_z(L, t, mu, beta)
            rel_err = abs(summary.log_z[n] - lz_exact) / abs(lz_exact)
            row = (
                f"{beta:10.4f}  {1.0 / beta:8.4f}  {summary.log_z[n]:14.8f}  "
                f"{lz_exact:14.8f}  {rel_err:9.2e}  "
                f"{summary.free_energies[n]:10.6f}  "
                f"{-lz_exact / (beta * L):10.6f}  "
                f"{summary.energies[n]:10.6f}  "
                f"{exact_energy(L, t, mu, beta):10.6f}  "
                f"{summary.specific_heats[n]:9.5f}  "
                f"{exact_specific_heat(L, t, mu, beta):9.5f}  "
                f"{summary.entropies[n]:9.5f}  "
                f"{exact_entropy(L, t, mu, beta):9.5f}  "
                f"{summary.max_bonds[n]:5d}"
            )
            if obs_exact:
                row += f"  {summary.exact_energies[n]:10.6f}"
            print(row)

        print()
        print(f"Final β = {summary.betas[-1]:.4g}  "
              f"(T = {1 / summary.betas[-1]:.4g})")
        lz_cold = exact_log_z(L, t, mu, summary.betas[-1])
        print(f"log Z relative error at the cold end: "
              f"{abs(summary.log_z[-1] - lz_cold) / abs(lz_cold):.2e}")
        print(f"Largest Krylov error estimate: "
              f"{max(summary.krylov_errors, default=0.0):.2e}")
        print(f"Largest discarded weight: "
              f"{max(summary.discarded_weights, default=0.0):.2e}")
        print()
        print("TgTRG computes log Z itself, and f follows from it directly, "
              "so both track the exact solution to roundoff.")
        print("u, c_V, and S come from finite differences on the β grid and "
              "are only as good as its spacing — c_V, a second derivative, "
              "worst of all, and especially where the spacing jumps at the "
              "phase boundary. Refine with a smaller --tau, or measure "
              "directly with --obs-exact.")

    return summary, artifact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "TgTRG finite-temperature thermodynamics for the 1D spinless "
            "free-fermion chain.\n\n"
            "Examples:\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinless.py\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinless.py --scheme 1s\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinless.py --order 1\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinless.py --beta-switch 0 --steps 11\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinless.py --tau 0.25 --lin-steps 40\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinless.py --symmetry Z2 --max-bond 32\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinless.py --obs-exact\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinless.py --mu 0.5\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinless.py --env-cache-dir /tmp/env_cache\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinless.py --checkpoint-dir /tmp/tgtrg_ckpt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--L', type=int, default=8, metavar='N',
        help='chain length (default: 8)',
    )
    p.add_argument(
        '--t', type=float, default=1.0, metavar='T',
        help='nearest-neighbor hopping amplitude (default: 1.0)',
    )
    p.add_argument(
        '--mu', type=float, default=0.0, metavar='MU',
        help='chemical potential (default: 0.0)',
    )
    p.add_argument(
        '--symmetry', choices=['U1', 'Z2'], default='U1',
        help='symmetry group exploited (default: U1)',
    )
    p.add_argument(
        '--scheme', choices=['1s', '2s'], default='2s',
        help='tangent-space update scheme: 1s (fixed bond) or 2s (adaptive, default)',
    )
    p.add_argument(
        '--order', type=int, choices=[1, 2], default=2,
        help='integrator order per step: 1 (one half-sweep) or 2 (symmetric, default)',
    )
    p.add_argument(
        '--tau0', type=float, default=2 ** -12, metavar='TAU',
        help='initial inverse temperature tau_0 (default: 2**-12)',
    )
    p.add_argument(
        '--beta-switch', type=float, default=2.0, metavar='BETA',
        help=(
            'inverse temperature at which the exponential phase hands over to '
            'the linear one; 0 cools linearly from tau_0 (default: 2.0)'
        ),
    )
    p.add_argument(
        '--tau', type=float, default=0.5, metavar='TAU',
        help='step size of the linear phase (default: 0.5)',
    )
    p.add_argument(
        '--lin-steps', type=int, default=20, metavar='N',
        help='number of linear steps after the handover (default: 20)',
    )
    p.add_argument(
        '--steps', type=int, default=None, metavar='N',
        help='explicit number of exponential steps, overriding --beta-switch',
    )
    p.add_argument(
        '--taylor-order', type=int, default=10, metavar='K',
        help='Taylor series truncation order for the initial rho(tau_0) (default: 10)',
    )
    p.add_argument(
        '--max-bond', type=int, default=None, metavar='D',
        help='maximum bond dimension (default: unlimited)',
    )
    p.add_argument(
        '--trunc-thresh', type=float, default=1e-12, metavar='THR',
        help='relative SVD truncation threshold (default: 1e-12)',
    )
    p.add_argument(
        '--krylov-dim', type=int, default=16, metavar='M',
        help='maximum Krylov dimension per local propagator (default: 16)',
    )
    p.add_argument(
        '--krylov-tol', type=float, default=1e-12, metavar='TOL',
        help='Krylov convergence tolerance (default: 1e-12)',
    )
    p.add_argument(
        '--obs-exact', action='store_true',
        help='also measure u = Tr[rho H] / (Tr[rho] L) directly at each step',
    )
    p.add_argument(
        '--env-cache-dir', default=None, metavar='PATH',
        help=(
            'directory for environment block cache files; enables disk-spilling '
            'so only --env-window blocks per direction are kept in memory '
            '(default: disabled, all blocks kept in memory)'
        ),
    )
    p.add_argument(
        '--env-window', type=int, default=2, metavar='W',
        help='sliding-window size for in-memory environment blocks (default: 2)',
    )
    p.add_argument(
        '--no-env-async-io', action='store_true',
        help='disable asynchronous environment block I/O (default: async enabled)',
    )
    p.add_argument(
        '--checkpoint-dir', default=None, metavar='PATH',
        help='directory for thermal.ckpt / tgtrg.ckpt / artifacts (default: cwd)',
    )
    p.add_argument(
        '--no-save-artifacts', action='store_true',
        help='disable per-step Artifact archives under artifacts/',
    )
    p.add_argument(
        '--save-artifacts-every', type=int, default=1, metavar='K',
        help='archive stride in steps when saving artifacts (default: 1)',
    )
    p.add_argument(
        '--quiet', action='store_true',
        help='suppress configuration and results output',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    alice.configure_logging()

    tgtrg_spinless(
        L=args.L,
        t=args.t,
        mu=args.mu,
        symmetry=args.symmetry,
        scheme=args.scheme,
        order=args.order,
        tau0=args.tau0,
        beta_switch=args.beta_switch,
        tau=args.tau,
        lin_steps=args.lin_steps,
        steps=args.steps,
        taylor_order=args.taylor_order,
        max_bond=args.max_bond,
        trunc_thresh=args.trunc_thresh,
        krylov_dim=args.krylov_dim,
        krylov_tol=args.krylov_tol,
        obs_exact=args.obs_exact,
        env_cache_dir=args.env_cache_dir,
        env_async_io=not args.no_env_async_io,
        env_window=args.env_window,
        checkpoint_dir=args.checkpoint_dir,
        save_artifacts=not args.no_save_artifacts,
        save_artifacts_every=args.save_artifacts_every,
        verbose=not args.quiet,
    )
