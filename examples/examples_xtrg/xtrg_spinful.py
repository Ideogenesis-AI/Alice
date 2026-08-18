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


"""Example: XTRG finite-temperature thermodynamics of the 1D Hubbard chain.

Computes ρ(β) = e^{-βH} via XTRG for the Hubbard Hamiltonian

    H = -t Σ_{<i,j>,σ} (c†_{i,σ} c_{j,σ} + h.c.) + U Σ_i n_{i,↑} n_{i,↓}
        - (μ + U/2) Σ_i n_i

for a chain of length L with open boundary conditions, then compares log
Z(β), the free energy, internal energy, and entropy against the exact
grand-canonical solution at each cooling step.

At U=0 the spin-up and spin-down channels decouple, and each channel is
identical to a spinless free-fermion chain with the same μ, so:

    log Z_spinful(β) = 2 × log Z_spinless(β).

For U != 0 there is no closed-form log Z, so only the XTRG thermodynamics are
printed, alongside the U=0 reference at the same μ for context.

Run from the repository root:

    uv run python examples/examples_xtrg/xtrg_spinful.py
    uv run python examples/examples_xtrg/xtrg_spinful.py --scheme 1s
    uv run python examples/examples_xtrg/xtrg_spinful.py --scheme 1sp
    uv run python examples/examples_xtrg/xtrg_spinful.py --scheme 1sp --expand-k 8
    uv run python examples/examples_xtrg/xtrg_spinful.py --steps 8
    uv run python examples/examples_xtrg/xtrg_spinful.py --symmetry U1,U1 --max-bond 64
    uv run python examples/examples_xtrg/xtrg_spinful.py --U 4.0
    uv run python examples/examples_xtrg/xtrg_spinful.py --env-cache-dir /tmp/env_cache
    uv run python examples/examples_xtrg/xtrg_spinful.py --checkpoint-dir /tmp/xtrg_ckpt
    uv run python examples/examples_xtrg/xtrg_spinful.py --help
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import alice
from alice.network import build_hamiltonian, build_interaction
from alice.network.thermal import thermal_mpo
from alice.algorithm import xtrg


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


# ---------------------------------------------------------------------------
# Main XTRG function
# ---------------------------------------------------------------------------

def xtrg_spinful(
    L: int = 8,
    t: float = 1.0,
    U: float = 0.0,
    mu: float = 0.0,
    symmetry: str = 'Z2,SU2',
    scheme: str = '2s',
    tau0: float = 2 ** -12,
    steps: int = 10,
    taylor_order: int = 10,
    max_bond: Optional[int] = None,
    sweeps: int = 4,
    trunc_thresh: float = 1e-15,
    expand_k: int = 4,
    expand_alpha: Optional[int] = None,
    env_cache_dir: Optional[str] = None,
    env_async_io: bool = True,
    env_window: int = 2,
    checkpoint_dir: Optional[str] = None,
    save_artifacts: bool = True,
    save_artifacts_since: int = 0,
    verbose: bool = True,
) -> Tuple[xtrg.Summary, xtrg.Artifact]:
    """Run XTRG for the 1D Hubbard chain.

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
        XTRG update scheme per squaring step: `'1s'` (1-site), `'2s'`
        (2-site, default), or `'1sp'` (1-site-plus / controlled bond
        expansion). The 1-site-plus scheme grows the bond dimension cheaply
        via a complement isometry before each 1-site update, targeting
        near-2-site accuracy at closer-to-1-site cost.
    tau0:
        Initial inverse temperature τ₀. Should be small enough that the
        Taylor expansion of `e^{-τ₀ H}` converges; doubled at each cooling
        step.
    steps:
        Number of cooling steps. The final inverse temperature is
        `β_max = 2^steps × τ0`.
    taylor_order:
        Truncation order of the Taylor series for `ρ(τ0) = e^{-τ0 H}`.
    max_bond:
        Maximum bond dimension of the compressed ρ. `None` (default) means
        unlimited.
    sweeps:
        Number of full variational sweeps per cooling step.
    trunc_thresh:
        SVD truncation threshold (default: `1e-15`).
    expand_k:
        Maximum number of complement vectors added per bond end in the
        `'1sp'` scheme (default: 4). Ignored for `'1s'` and `'2s'`.
    expand_alpha:
        Truncated internal bond dimension used to cheaply compress each
        factor-MPO's connector bond in the `'1sp'` complement computation.
        `None` (default) keeps the full bond. Ignored for `'1s'` and `'2s'`.
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
        Directory for `thermal.ckpt`, mid-run `xtrg.ckpt`, and optional
        `artifacts/` archives. `None` (default) writes to the current
        working directory.
    save_artifacts:
        If `True` (default), archive per-step density matrices under
        `artifacts/step_XX.ckpt`.
    save_artifacts_since:
        First step index (inclusive) to archive when `save_artifacts` is
        `True` (default: `0`, after Taylor init).
    verbose:
        Print configuration and the β-by-β thermodynamics table when `True`.

    Returns
    -------
    Summary
        Thermodynamic history at each cooling step.
    Artifact
        Final density matrix `ρ(β_max)`.
    """
    _VALID_SYMMETRIES = ('U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2')
    if symmetry not in _VALID_SYMMETRIES:
        raise ValueError(
            f"symmetry must be one of {_VALID_SYMMETRIES}, got {symmetry!r}"
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

    max_bond_str = str(max_bond) if max_bond is not None else 'unlimited'
    if verbose:
        print(f"Chain length   : {geo.L}")
        print(f"Symmetry       : {symmetry}")
        print(f"Hopping t      : {t}")
        print(f"Hubbard U      : {U}")
        print(f"Chemical pot μ : {mu}  (effective μ_eff = {mu + U / 2:.4f})")
        print(f"Scheme         : {scheme}")
        if scheme in ('1sp', '1-site-plus', 'one-site-plus'):
            alpha_str = str(expand_alpha) if expand_alpha is not None else 'full'
            print(f"  expand_k     : {expand_k}")
            print(f"  expand_alpha : {alpha_str}")
        print(f"MPO bond dims  : {H.bond_dims}")
        print(f"tau_0          : {tau0:.6g}   taylor_order = {taylor_order}")
        print(f"Cooling steps  : {steps}   beta_max = {tau0 * 2 ** steps:.6g}")
        print(f"Max bond dim   : {max_bond_str}   sweeps/step = {sweeps}   "
              f"trunc_thresh = {trunc_thresh:.2e}")
        if env_cache_dir is not None:
            print(f"Env cache dir  : {env_cache_dir}  (window={env_window}, async={env_async_io})")
        if checkpoint_dir is not None:
            print(f"Checkpoint dir : {checkpoint_dir}")
        if save_artifacts:
            print(f"Artifacts      : since step {save_artifacts_since}")
        print()

    # -----------------------------------------------------------------------
    # XTRG run
    # -----------------------------------------------------------------------
    opts = xtrg.Options(
        scheme=scheme,
        tau_0=tau0,
        n_steps=steps,
        taylor_order=taylor_order,
        max_bond=max_bond,
        trunc_thresh=trunc_thresh,
        n_sweeps=sweeps,
        expand_k=expand_k,
        expand_alpha=expand_alpha,
        env_cache_dir=env_cache_dir,
        env_async_io=env_async_io,
        env_window=env_window,
        checkpoint_dir=checkpoint_dir,
        save_artifacts=save_artifacts,
        save_artifacts_since=save_artifacts_since,
    )
    rho0 = thermal_mpo(H, opts.tau_0, opts.taylor_order, spc)
    summary, artifact = xtrg.run(xtrg.Artifact(rho=rho0, beta=opts.tau_0, step=0), opts)

    # -----------------------------------------------------------------------
    # Print thermodynamics table
    # -----------------------------------------------------------------------
    if verbose:
        print()
        if U == 0.0:
            header = (
                f"{'beta':>10}  {'T':>8}  {'log_Z_xtrg':>14}  "
                f"{'log_Z_exact':>14}  {'rel_err':>10}  {'f':>10}  "
                f"{'u':>10}  {'S':>10}"
            )
            print(header)
            print("-" * len(header))
            for n, beta in enumerate(summary.betas):
                T = 1.0 / beta
                lz_exact = exact_log_z_spinful(L, t, mu, beta)
                rel_err = abs(summary.log_z[n] - lz_exact) / abs(lz_exact)
                f = summary.free_energies[n]
                u = summary.energies[n]
                S = summary.entropies[n]
                print(
                    f"{beta:10.4f}  {T:8.4f}  {summary.log_z[n]:14.8f}  "
                    f"{lz_exact:14.8f}  {rel_err:10.2e}  {f:10.6f}  "
                    f"{u:10.6f}  {S:10.6f}"
                )
        else:
            print("Note: exact grand-canonical log Z is only known at U=0; "
                  "no closed-form reference exists for this interacting run.")
            print()
            header = (
                f"{'beta':>10}  {'T':>8}  {'log_Z_xtrg':>14}  "
                f"{'f':>10}  {'u':>10}  {'S':>10}"
            )
            print(header)
            print("-" * len(header))
            for n, beta in enumerate(summary.betas):
                T = 1.0 / beta
                f = summary.free_energies[n]
                u = summary.energies[n]
                S = summary.entropies[n]
                print(
                    f"{beta:10.4f}  {T:8.4f}  {summary.log_z[n]:14.8f}  "
                    f"{f:10.6f}  {u:10.6f}  {S:10.6f}"
                )
            print()
            lz_ref = exact_log_z_spinful(L, t, mu, summary.betas[-1])
            print(f"U=0 reference log Z at β={summary.betas[-1]:.4g} (same μ): {lz_ref:.8f}")

        print()
        print(f"Final β = {summary.betas[-1]:.4g}  (T = {1 / summary.betas[-1]:.4g})")

    return summary, artifact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "XTRG finite-temperature thermodynamics for the 1D Hubbard chain.\n\n"
            "Examples:\n"
            "  uv run python examples/examples_xtrg/xtrg_spinful.py\n"
            "  uv run python examples/examples_xtrg/xtrg_spinful.py --scheme 1s\n"
            "  uv run python examples/examples_xtrg/xtrg_spinful.py --scheme 1sp\n"
            "  uv run python examples/examples_xtrg/xtrg_spinful.py --scheme 1sp --expand-k 8\n"
            "  uv run python examples/examples_xtrg/xtrg_spinful.py --steps 8\n"
            "  uv run python examples/examples_xtrg/xtrg_spinful.py --symmetry U1,U1 --max-bond 64\n"
            "  uv run python examples/examples_xtrg/xtrg_spinful.py --U 4.0\n"
            "  uv run python examples/examples_xtrg/xtrg_spinful.py --env-cache-dir /tmp/env_cache\n"
            "  uv run python examples/examples_xtrg/xtrg_spinful.py --checkpoint-dir /tmp/xtrg_ckpt"
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
        help='on-site Coulomb repulsion (default: 0.0)',
    )
    p.add_argument(
        '--mu', type=float, default=0.0, metavar='MU',
        help='chemical potential relative to half-filling (default: 0.0)',
    )
    p.add_argument(
        '--symmetry',
        choices=['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'],
        default='Z2,SU2',
        help='band symmetry exploited (default: Z2,SU2)',
    )
    p.add_argument(
        '--scheme', choices=['1s', '2s', '1sp'], default='2s',
        help='XTRG update scheme: 1s (1-site), 2s (2-site, default), or 1sp (1-site-plus / CBE)',
    )
    p.add_argument(
        '--tau0', type=float, default=2 ** -12, metavar='TAU',
        help='initial inverse temperature tau_0 (default: 2**-12)',
    )
    p.add_argument(
        '--steps', type=int, default=10, metavar='N',
        help='number of cooling steps (default: 10)',
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
        '--sweeps', type=int, default=4, metavar='K',
        help='variational sweeps per cooling step (default: 4)',
    )
    p.add_argument(
        '--trunc-thresh', type=float, default=1e-15, metavar='THR',
        help='SVD truncation threshold (default: 1e-15)',
    )
    p.add_argument(
        '--expand-k', type=int, default=4, metavar='K',
        help='complement vectors per bond end for the 1sp scheme (default: 4)',
    )
    p.add_argument(
        '--expand-alpha', type=int, default=None, metavar='A',
        help=(
            'truncated connector-bond dimension for the 1sp complement '
            'computation; None (default) keeps the full bond'
        ),
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
        help='directory for thermal.ckpt / xtrg.ckpt / artifacts (default: cwd)',
    )
    p.add_argument(
        '--no-save-artifacts', action='store_true',
        help='disable per-step Artifact archives under artifacts/',
    )
    p.add_argument(
        '--save-artifacts-since', type=int, default=0, metavar='K',
        help='first step index to archive when saving artifacts (default: 0)',
    )
    p.add_argument(
        '--quiet', action='store_true',
        help='suppress configuration and results output',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    alice.configure_logging()

    xtrg_spinful(
        L=args.L,
        t=args.t,
        U=args.U,
        mu=args.mu,
        symmetry=args.symmetry,
        scheme=args.scheme,
        tau0=args.tau0,
        steps=args.steps,
        taylor_order=args.taylor_order,
        max_bond=args.max_bond,
        sweeps=args.sweeps,
        trunc_thresh=args.trunc_thresh,
        expand_k=args.expand_k,
        expand_alpha=args.expand_alpha,
        env_cache_dir=args.env_cache_dir,
        env_async_io=not args.no_env_async_io,
        env_window=args.env_window,
        checkpoint_dir=args.checkpoint_dir,
        save_artifacts=not args.no_save_artifacts,
        save_artifacts_since=args.save_artifacts_since,
        verbose=not args.quiet,
    )
