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


"""Example: TgTRG finite-temperature thermodynamics of the 1D Hubbard chain.

Cools ρ(β) = e^{-βH} along the tangent space of the MPO manifold for the
Hubbard Hamiltonian

    H = -t Σ_{<i,j>,σ} (c†_{i,σ} c_{j,σ} + h.c.) + U Σ_i n_{i,↑} n_{i,↓}
        - (μ + U/2) Σ_i n_i

on a chain of length L with open boundary conditions, then reports log Z(β),
the free energy, internal energy, and entropy at each cooling step.

At U=0 the spin-up and spin-down channels decouple, and each channel is
identical to a spinless free-fermion chain with the same μ, so:

    log Z_spinful(β) = 2 × log Z_spinless(β),

which the run is compared against. For U != 0 there is no closed-form log Z,
so only the TgTRG thermodynamics are printed, alongside the U=0 reference at
the same μ for context.

As in the spinless example, the default schedule stacks an exponential phase
up to `--beta-switch` onto a linear phase of `--tau` steps; see
`tgtrg_spinless.py` and the `tgtrg` module docstring for why.

Run from the repository root:

    uv run python examples/examples_tgtrg/tgtrg_spinful.py
    uv run python examples/examples_tgtrg/tgtrg_spinful.py --scheme 1s
    uv run python examples/examples_tgtrg/tgtrg_spinful.py --order 1
    uv run python examples/examples_tgtrg/tgtrg_spinful.py --beta-switch 0 --steps 10
    uv run python examples/examples_tgtrg/tgtrg_spinful.py --tau 0.25 --lin-steps 24
    uv run python examples/examples_tgtrg/tgtrg_spinful.py --symmetry U1,U1 --max-bond 64
    uv run python examples/examples_tgtrg/tgtrg_spinful.py --U 4.0
    uv run python examples/examples_tgtrg/tgtrg_spinful.py --obs-exact
    uv run python examples/examples_tgtrg/tgtrg_spinful.py --env-cache-dir /tmp/env_cache
    uv run python examples/examples_tgtrg/tgtrg_spinful.py --checkpoint-dir /tmp/tgtrg_ckpt
    uv run python examples/examples_tgtrg/tgtrg_spinful.py --help
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import alice
from alice.network import build_hamiltonian, build_interaction
from alice.network.thermal import thermal_mpo
from alice.algorithm import tgtrg


# ---------------------------------------------------------------------------
# Exact reference (U=0 only)
# ---------------------------------------------------------------------------

def exact_log_z_spinless(L: int, t: float, mu: float, beta: float) -> float:
    """Exact grand-canonical log Z for a spinless OBC free-fermion chain."""
    eps = [-2.0 * t * math.cos(k * math.pi / (L + 1)) for k in range(1, L + 1)]
    return sum(math.log1p(math.exp(-beta * (e - mu))) for e in eps)


def exact_log_z_spinful(L: int, t: float, mu: float, beta: float) -> float:
    """Exact grand-canonical log Z for the U=0 Hubbard chain (two decoupled spin channels)."""
    return 2.0 * exact_log_z_spinless(L, t, mu, beta)


def exact_energy_spinful(L: int, t: float, mu: float, beta: float) -> float:
    """Exact internal energy per site of the U=0 Hubbard chain.

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
        `u(β) = 2 Σ_k x_k / (1 + exp(β x_k)) / L` with `x_k = ε_k - μ`.
    """
    xs = [-2.0 * t * math.cos(k * math.pi / (L + 1)) - mu
          for k in range(1, L + 1)]
    return 2.0 * sum(x / (1.0 + math.exp(beta * x)) for x in xs) / L


# ---------------------------------------------------------------------------
# Main TgTRG function
# ---------------------------------------------------------------------------

def tgtrg_spinful(
    L: int = 8,
    t: float = 1.0,
    U: float = 0.0,
    mu: float = 0.0,
    symmetry: str = 'Z2,SU2',
    scheme: str = '2s',
    order: int = 2,
    tau0: float = 2 ** -12,
    beta_switch: float = 2.0,
    tau: float = 0.5,
    lin_steps: int = 12,
    steps: Optional[int] = None,
    taylor_order: int = 10,
    max_bond: Optional[int] = None,
    trunc_thresh: float = 1e-12,
    krylov_dim: int = 16,
    krylov_tol: float = 1e-12,
    expand_k: int = 4,
    expand_alpha: Optional[int] = None,
    obs_exact: bool = False,
    env_cache_dir: Optional[str] = None,
    env_async_io: bool = True,
    env_window: int = 2,
    checkpoint_dir: Optional[str] = None,
    save_artifacts: bool = True,
    save_artifacts_every: int = 1,
    verbose: bool = True,
) -> Tuple[tgtrg.Summary, tgtrg.Artifact]:
    """Cool the 1D Hubbard chain with TgTRG.

    Runs the stacked schedule by default: an exponential phase up to
    `beta_switch`, then `lin_steps` linear steps of size `tau`. Each phase
    is one `tgtrg.run()` call, the second resuming from the `Artifact` the
    first returned.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.
    U:
        On-site Coulomb repulsion. At `U = 0` the model reduces to two
        decoupled spinless free-fermion channels, which have an exact
        grand-canonical solution used for comparison.
    mu:
        Chemical potential relative to half-filling. The shift `U/2` is
        applied automatically to enforce particle-hole symmetry at `mu = 0`.
    symmetry:
        Band symmetry exploited. Accepted values:

        * `'U1,U1'`  — particle number × spin-z
        * `'Z2,U1'`  — fermion parity × spin-z
        * `'U1,SU2'` — particle number × full SU(2) spin rotation
        * `'Z2,SU2'` — fermion parity × full SU(2) spin rotation (default)
    scheme:
        Tangent-space update scheme: `'1s'` (1-site, fixed bond dimension),
        `'2s'` (2-site, default, adaptive bond dimension), or `'1sp'`
        (1-site-plus, adaptive at close to 1-site cost).
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
    expand_k:
        Complement directions added per bond per local update under
        `'1sp'` (default: 4). Ignored by the other schemes.
    expand_alpha:
        Internal bond of the cheap factorization inside the `'1sp'`
        expansion. `None` (default) keeps the full bond.
    obs_exact:
        If `True`, measure `u = Tr[ρH] / (Tr[ρ] L)` directly at each step
        in addition to the finite-difference estimate. Off by default. The
        direct measurement is the one to trust for an interacting run,
        where there is no analytic `u` to compare the grid against.
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
        If `symmetry` is not one of the four accepted strings, or if the
        schedule has no steps at all.
    """
    _VALID_SYMMETRIES = ('U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2')
    if symmetry not in _VALID_SYMMETRIES:
        raise ValueError(
            f"symmetry must be one of {_VALID_SYMMETRIES}, got {symmetry!r}"
        )

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
    # Build the Hubbard Hamiltonian MPO
    # -----------------------------------------------------------------------
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': L, 'bcx': 'OBC', 'n2x': True},
        'model': {
            'category': 'conductor',
            'label': 'Hubbard',
            'symmetry': symmetry,
            't': t,
            'U': U,
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
        print(f"Hubbard U      : {U}")
        print(f"Chemical pot μ : {mu}  (effective μ_eff = {mu + U / 2:.4f})")
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
        expand_k=expand_k,
        expand_alpha=expand_alpha,
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
    # Print thermodynamics table
    # -----------------------------------------------------------------------
    if verbose:
        print()
        if U != 0.0:
            print("Note: exact grand-canonical log Z is only known at U=0; the "
                  "reference columns below are the U=0 solution at the same μ, "
                  "shown for context only.")
            print()

        header = (
            f"{'beta':>10}  {'T':>8}  {'log_Z':>14}  "
            f"{'log_Z_U=0':>14}  {'rel_err':>9}  {'f':>10}  "
            f"{'u':>10}  {'u_U=0':>10}  {'S':>9}  {'D':>5}"
        )
        if obs_exact:
            header += f"  {'u_direct':>10}"
        print(header)
        print("-" * len(header))

        for n, beta in enumerate(summary.betas):
            lz_ref = exact_log_z_spinful(L, t, mu, beta)
            rel_err = abs(summary.log_z[n] - lz_ref) / abs(lz_ref)
            row = (
                f"{beta:10.4f}  {1.0 / beta:8.4f}  {summary.log_z[n]:14.8f}  "
                f"{lz_ref:14.8f}  {rel_err:9.2e}  "
                f"{summary.free_energies[n]:10.6f}  "
                f"{summary.energies[n]:10.6f}  "
                f"{exact_energy_spinful(L, t, mu, beta):10.6f}  "
                f"{summary.entropies[n]:9.5f}  "
                f"{summary.max_bonds[n]:5d}"
            )
            if obs_exact:
                row += f"  {summary.exact_energies[n]:10.6f}"
            print(row)

        print()
        print(f"Final β = {summary.betas[-1]:.4g}  "
              f"(T = {1 / summary.betas[-1]:.4g})")
        print(f"Largest Krylov error estimate: "
              f"{max(summary.krylov_errors, default=0.0):.2e}")
        print(f"Largest discarded weight: "
              f"{max(summary.discarded_weights, default=0.0):.2e}")

    return summary, artifact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "TgTRG finite-temperature thermodynamics for the 1D Hubbard chain.\n\n"
            "Examples:\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinful.py\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinful.py --scheme 1s\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinful.py --order 1\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinful.py --beta-switch 0 --steps 10\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinful.py --tau 0.25 --lin-steps 24\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinful.py --symmetry U1,U1 --max-bond 64\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinful.py --U 4.0\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinful.py --obs-exact\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinful.py --env-cache-dir /tmp/env_cache\n"
            "  uv run python examples/examples_tgtrg/tgtrg_spinful.py --checkpoint-dir /tmp/tgtrg_ckpt"
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
        '--U', type=float, default=0.0, metavar='U',
        help='on-site Coulomb repulsion (default: 0.0, the exactly solvable point)',
    )
    p.add_argument(
        '--mu', type=float, default=0.0, metavar='MU',
        help='chemical potential relative to half-filling (default: 0.0)',
    )
    p.add_argument(
        '--symmetry', choices=['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'],
        default='Z2,SU2',
        help='band symmetry exploited (default: Z2,SU2)',
    )
    p.add_argument(
        '--scheme', choices=['1s', '2s', '1sp'], default='2s',
        help='tangent-space update scheme: 1s (fixed bond), 2s (adaptive, default), '
             'or 1sp (adaptive at close to 1-site cost)',
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
        '--lin-steps', type=int, default=12, metavar='N',
        help='number of linear steps after the handover (default: 12)',
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
        '--expand-k', type=int, default=4, metavar='K',
        help='complement directions per bond for the 1sp scheme (default: 4)',
    )
    p.add_argument(
        '--expand-alpha', type=int, default=None, metavar='A',
        help='internal bond of the cheap factorization inside the 1sp expansion '
             '(default: full)',
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

    tgtrg_spinful(
        L=args.L,
        t=args.t,
        U=args.U,
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
        expand_k=args.expand_k,
        expand_alpha=args.expand_alpha,
        obs_exact=args.obs_exact,
        env_cache_dir=args.env_cache_dir,
        env_async_io=not args.no_env_async_io,
        env_window=args.env_window,
        checkpoint_dir=args.checkpoint_dir,
        save_artifacts=not args.no_save_artifacts,
        save_artifacts_every=args.save_artifacts_every,
        verbose=not args.quiet,
    )
