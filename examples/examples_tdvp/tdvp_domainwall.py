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


"""Example: TDVP domain-wall melting on a free-fermion chain.

Quenches the sharp domain wall

    |ψ(0)⟩ = |1 1 … 1 0 0 … 0⟩

under the tight-binding Hamiltonian

    H = -t Σ_j (c†_j c_{j+1} + h.c.)

on an open chain of length L, and compares both the site occupations and the
entanglement entropy against exact results at every reported time.

This is the genuinely many-body counterpart to `tdvp_freefermion.py`. Half the
chain is occupied, entanglement grows as the wall melts, and the bond
dimension has to grow with it — so unlike the single-particle example, this
one actually exercises truncation, and `--bond-dim` matters. Use `--bond-scan`
to watch the error fall as the bond dimension is raised.

The exact reference is still closed form. A product state is a Slater
determinant, and a quadratic Hamiltonian keeps it one, so the entire
many-body state is carried by the `L x L` correlation matrix

    M(t) = U(t) diag(occ) U(t)†,   U(t) = exp(-i h t).

Its diagonal gives the occupations. Its restriction to sites `0 … ℓ-1` gives
the entanglement entropy through the Gaussian formula

    S(ℓ) = -Σ [ν log ν + (1-ν) log(1-ν)]

over the eigenvalues `ν` of that block. The entropy is the sharper of the two
checks: occupations are a one-body quantity that a truncated state can get
roughly right, whereas the entropy is a direct statement about the Schmidt
spectrum the truncation is throwing away.

Two further things this example shows:

- **The light cone.** Correlations spread at the maximum group velocity
  `v_max = 2t`, so the melted region grows as `x0 ± 2 t`, which is visible in
  the rendered density profile.
- **The 1-site trap.** A product state has every bond at dimension 1, and the
  1-site scheme cannot grow a bond. Running `--scheme 1s` from this initial
  state therefore does not evolve at all — silently, since the norm and the
  energy stay perfectly conserved. This is why `Options.scheme` defaults to
  `'2s'`, and why `run()` warns when a `1s` run starts from such a state.
- **The way out of it.** `--scheme 1sp` widens each bond before updating it,
  so the rank adapts as it does under `'2s'` while every local exponential
  stays 1-site sized. It melts the wall from the same product state, truncates
  and reports discarded weight like `'2s'`, and `--bond-scan` accepts it
  alongside. See `tdvp_scheme_comparison.py` for the three schemes measured
  against each other on exactly this quench.

Run from the repository root:

    uv run python examples/examples_tdvp/tdvp_domainwall.py
    uv run python examples/examples_tdvp/tdvp_domainwall.py --L 24 --time 4
    uv run python examples/examples_tdvp/tdvp_domainwall.py --bond-dim 256
    uv run python examples/examples_tdvp/tdvp_domainwall.py --bond-scan
    uv run python examples/examples_tdvp/tdvp_domainwall.py --scheme 1s
    uv run python examples/examples_tdvp/tdvp_domainwall.py --scheme 1sp --expand-k 8
    uv run python examples/examples_tdvp/tdvp_domainwall.py --checkpoint-dir /tmp/tdvp_ckpt
    uv run python examples/examples_tdvp/tdvp_domainwall.py --help
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import alice
from alice import build_hamiltonian, build_interaction, init_mps, observe, tdvp
from alice.physics.system import build_fermionic
from freefermion_exact import (  # pyright: ignore[reportMissingImports]
    correlation_matrix,
    density_row,
    error_table,
    exact_entropy,
    mps_entropy,
    mps_occupations,
    occupation_mpos,
)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _build_system(L: int, t: float):
    """Build the MPO, the occupation observables, and the physical space.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Hopping amplitude.

    Returns
    -------
    tuple
        `(mpo, n_obs, spc, ops)`.
    """
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': L, 'bcx': 'OBC', 'n2x': True},
        'model': {
            'category': 'fermionic',
            'label': 'FreeFermion',
            'symmetry': 'U1',
            't': t,
        },
    }
    interactions, spc, geo = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, geo.L, spc)
    _, ops = build_fermionic('U1')
    return mpo, occupation_mpos(geo.L, spc, ops), spc, ops


def _domain_wall_state(L: int, spc, ops):
    """Build the sharp domain-wall product state, left half filled.

    Parameters
    ----------
    L:
        Chain length.
    spc:
        Physical space from `build_interaction`.
    ops:
        Operator dictionary from `build_fermionic`.

    Returns
    -------
    tuple
        `(mps, occ)` where `occ` is the initial occupation list.
    """
    occ = [1] * (L // 2) + [0] * (L - L // 2)
    mps = init_mps(L, spc, ops, bond_dim=1, config=occ,
                   dtype=torch.complex128)
    mps.canonical(0)
    mps.normalize()
    return mps, occ


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def tdvp_domainwall(
    L: int = 20,
    t: float = 1.0,
    time: float = 3.0,
    dt: float = 0.05,
    scheme: str = '2s',
    order: int = 2,
    bond_dim: int = 128,
    trunc_thresh: float = 1e-12,
    expand_k: int = 4,
    expand_alpha: Optional[int] = None,
    krylov_dim: int = 16,
    snapshots: int = 6,
    checkpoint_dir: Optional[str] = None,
    env_cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[tdvp.Summary, tdvp.Artifact]:
    """Melt a domain wall and compare against the exact free-fermion result.

    Parameters
    ----------
    L:
        Chain length. The wall sits between sites `L/2 - 1` and `L/2`.
    t:
        Nearest-neighbor hopping amplitude. The light cone spreads at `2t`.
    time:
        Total evolution time. Keep `2 t · time` below `L / 2` to stop the
        melting front from reaching the chain ends.
    dt:
        Time step.
    scheme:
        `'2s'` (default), `'1s'`, or `'1sp'`. The 1-site scheme cannot grow a
        bond, so from this product state it does not evolve; `'1sp'` can, and
        does so at 1-site cost. See the module docstring.
    order:
        Integrator order, `1` or `2`.
    bond_dim:
        Maximum bond dimension. This is the quantity that limits accuracy
        here, since the entanglement grows without bound as the wall melts.
    trunc_thresh:
        SVD truncation threshold.
    expand_k:
        Complement directions added per bond per local update, for `'1sp'`.
        Ignored by the other schemes. This caps how fast the bond profile can
        follow the growing entanglement, so too small a value shows up as an
        error floor that more `bond_dim` does not lift.
    expand_alpha:
        Internal bond of the cheap factorization inside the expansion, for
        `'1sp'`. `None` keeps the full bond.
    krylov_dim:
        Maximum Krylov subspace dimension for the local propagators.
    snapshots:
        Number of intermediate times at which to report.
    checkpoint_dir:
        Directory for `tdvp.ckpt` and `evolution.ckpt`. `None` uses the
        current working directory.
    env_cache_dir:
        Directory for spilling environment blocks to disk. `None` keeps them
        in memory.
    verbose:
        Print the profile table and the error summaries.

    Returns
    -------
    Summary
        Trajectory history from the final evolution segment.
    Artifact
        Final state with its time and step index.
    """
    mpo, n_obs, spc, ops = _build_system(L, t)
    mps, occ = _domain_wall_state(L, spc, ops)

    if verbose:
        print(f"Chain length   : {L}")
        print(f"Hopping t      : {t}")
        print(f"Initial state  : |{''.join(str(o) for o in occ)}>")
        print(f"Particles      : {sum(occ)}  (half filling)")
        print(f"Light cone     : x0 +/- 2t  =>  front reaches "
              f"+/-{2 * t * time:g} sites by t = {time:g}")
        print(f"Scheme         : {scheme}   order = {order}")
        if scheme in ('1sp', '1-site-plus', 'one-site-plus'):
            alpha_str = str(expand_alpha) if expand_alpha is not None else 'full'
            print(f"  expand_k     : {expand_k}   expand_alpha : {alpha_str}")
        print(f"Time step      : {dt}   total time = {time}")
        print(f"Bond dim       : {bond_dim}   trunc thresh = {trunc_thresh:.1e}")
        print()
        if scheme == '1s':
            print("NOTE: every bond of a product state has dimension 1 and the")
            print("      1-site scheme cannot grow one, so this run will not")
            print("      evolve. Norm and energy stay conserved regardless.")
            print("      Use '2s' or '1sp' to actually melt the wall.")
            print()

    n_total = max(1, round(time / dt))
    stride = max(1, n_total // max(1, snapshots))

    opts = tdvp.Options(
        scheme=scheme,
        dt=dt,
        n_steps=n_total,
        order=order,
        max_bond=bond_dim,
        trunc_thresh=trunc_thresh,
        expand_k=expand_k,
        expand_alpha=expand_alpha,
        krylov_dim=krylov_dim,
        checkpoint_dir=checkpoint_dir,
        env_cache_dir=env_cache_dir,
        save_artifacts=False,
    )

    if verbose:
        print(f"{'t':>6}  {'chi':>4}  {'n_j err':>9}  {'S err':>9}  "
              f"{'dw':>9}  {'N-Nex':>9}  {'E':>10}  density profile")
        print('-' * (72 + L))

    artifact = tdvp.Artifact(state=mps, time=0.0, step=0)
    summary = None
    step = 0
    while step < n_total:
        step = min(n_total, step + stride)
        summary, artifact = tdvp.run(artifact, mpo,
                                     dataclasses.replace(opts, n_steps=step))

        num = mps_occupations(artifact.state, n_obs)
        M = correlation_matrix(occ, t, artifact.time)
        exact = np.real(np.diag(M))
        s_err = max(
            abs(mps_entropy(artifact.state, ell) - exact_entropy(M, ell))
            for ell in range(1, L)
        )
        if verbose:
            print(f"{artifact.time:>6.2f}  {max(artifact.state.bond_dims):>4d}  "
                  f"{np.max(np.abs(num - exact)):>9.2e}  {s_err:>9.2e}  "
                  f"{summary.discarded_weights[-1]:>9.2e}  "
                  f"{num.sum() - sum(occ):>+9.1e}  "
                  f"{observe(artifact.state, mpo):>+10.2e}  "
                  f"|{density_row(num)}|")

    # -----------------------------------------------------------------------
    # Final comparison against the exact correlation matrix
    # -----------------------------------------------------------------------
    if verbose:
        num = mps_occupations(artifact.state, n_obs)
        M = correlation_matrix(occ, t, artifact.time)
        exact = np.real(np.diag(M))

        print()
        print(f"Occupations at t = {artifact.time:g}  (across the melted region)")
        lo, hi = max(0, L // 2 - 5), min(L, L // 2 + 5)
        table, max_err = error_table('n_j (TDVP)', num, exact, range(lo, hi))
        print(table)
        print()
        print(f"Max |n_j - exact| over all sites : {max_err:.3e}")
        print()

        print(f"Entanglement entropy at t = {artifact.time:g}")
        print(f"{'cut':>5}  {'S (TDVP)':>12}  {'S (exact)':>12}  "
              f"{'error':>11}  {'chi':>5}")
        print('-' * 54)
        worst = 0.0
        for ell in range(1, L):
            s_num = mps_entropy(artifact.state, ell)
            s_ex = exact_entropy(M, ell)
            worst = max(worst, abs(s_num - s_ex))
            print(f"{ell:>5}  {s_num:>12.8f}  {s_ex:>12.8f}  "
                  f"{s_num - s_ex:>+11.2e}  "
                  f"{artifact.state[ell].indices[0].dim:>5d}")
        print()
        print(f"Max |S - exact| over all cuts : {worst:.3e}")
        print()

        print(f"Particle number : {num.sum():.12f}  (exact {sum(occ)})")
        print(f"Energy          : {observe(artifact.state, mpo):+.3e}  "
              f"(exact 0, since the wall has no hopping energy)")
        print(f"Norm            : {artifact.state.norm():.12f}")
        print(f"Bond dims       : {artifact.state.bond_dims}")
        print(f"Max discarded weight over the run : "
              f"{max(summary.discarded_weights):.3e}")

    return summary, artifact


# ---------------------------------------------------------------------------
# Bond-dimension convergence scan
# ---------------------------------------------------------------------------

def bond_scan(
    L: int = 14,
    t: float = 1.0,
    time: float = 2.0,
    dt: float = 0.05,
    scheme: str = '2s',
    order: int = 2,
    expand_k: int = 4,
    expand_alpha: Optional[int] = None,
    bonds: Tuple[int, ...] = (4, 8, 16, 32, 64),
) -> None:
    """Report the error against the exact result as the bond dimension grows.

    The two error sources in this quench are truncation and the integrator's
    `O(dt²)`, and they dominate in different regimes. At small `max_bond`
    truncation dominates and raising it helps sharply; once the bond is large
    enough to hold the entanglement the melting has generated, the error
    stops improving and sits on a floor set by `dt`. The scan holds `dt`
    fixed to expose that floor, then halves `dt` at the largest bond
    dimension to confirm that the floor is indeed the time step and not a
    limitation of the state.

    Under `'1sp'` there is a third source, since `expand_k` caps how fast the
    bond profile can follow the entanglement. A scan whose plateau does not
    drop when `dt` is halved is pointing at the expansion rather than the
    integrator; raise `expand_k` to tell the two apart.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Hopping amplitude.
    time:
        Total evolution time, held fixed across the scan.
    dt:
        Time step, held fixed across the scan.
    scheme:
        `'2s'` (default) or `'1sp'`. `'1s'` would hold every bond at 1 and
        make the scan meaningless.
    order:
        Integrator order.
    expand_k:
        Complement directions added per bond per local update, for `'1sp'`.
    expand_alpha:
        Internal bond of the cheap factorization inside the expansion, for
        `'1sp'`. `None` keeps the full bond.
    bonds:
        Maximum bond dimensions to try.

    Raises
    ------
    ValueError
        If `scheme` is `'1s'`, which holds every bond at 1 whatever `bonds`
        says.
    """
    if scheme in ('1s', '1-site', 'one-site'):
        raise ValueError(
            "bond_scan needs a scheme whose bonds can grow; '1s' holds every "
            "bond of this product state at dimension 1 regardless of "
            "max_bond, so every row would be identical. Use '2s' or '1sp'"
        )

    mpo, n_obs, spc, ops = _build_system(L, t)
    _, occ = _domain_wall_state(L, spc, ops)

    M = correlation_matrix(occ, t, time)
    exact = np.real(np.diag(M))
    s_exact = [exact_entropy(M, ell) for ell in range(1, L)]

    print(f"Bond-dimension convergence  (L = {L}, t_final = {time}, "
          f"dt = {dt}, scheme = {scheme}, order = {order})")
    print("Reference: exact free-fermion correlation matrix")
    print()
    print(f"{'max_bond':>9}  {'chi':>4}  {'max |n_j - exact|':>19}  "
          f"{'max |S - exact|':>17}  {'max dw':>10}")
    print('-' * 70)

    def _run(max_bond: int, step: float):
        """Evolve from a fresh domain wall and return the two errors."""
        mps, _ = _domain_wall_state(L, spc, ops)
        summary, artifact = tdvp.run(
            tdvp.Artifact(state=mps, time=0.0, step=0), mpo,
            tdvp.Options(scheme=scheme, dt=step, n_steps=round(time / step),
                         order=order, max_bond=max_bond, expand_k=expand_k,
                         expand_alpha=expand_alpha, save_artifacts=False),
        )
        num = mps_occupations(artifact.state, n_obs)
        s_err = max(abs(mps_entropy(artifact.state, ell) - s_exact[ell - 1])
                    for ell in range(1, L))
        return (max(artifact.state.bond_dims),
                float(np.max(np.abs(num - exact))), s_err,
                max(summary.discarded_weights))

    for max_bond in bonds:
        chi, n_err, s_err, dw = _run(max_bond, dt)
        print(f"{max_bond:>9}  {chi:>4d}  {n_err:>19.3e}  {s_err:>17.3e}  "
              f"{dw:>10.2e}")

    # Halving dt at the largest bond dimension separates the two error
    # sources: if the plateau above is the time step, this drops by ~4.
    chi, n_err, s_err, dw = _run(bonds[-1], dt / 2)
    print()
    print(f"Same run at the largest bond dimension with dt = {dt / 2:g}:")
    print(f"{bonds[-1]:>9}  {chi:>4d}  {n_err:>19.3e}  {s_err:>17.3e}  "
          f"{dw:>10.2e}")
    print()
    print("The error falls steeply with the bond dimension while truncation")
    print("dominates, then flattens onto a plateau. The plateau is the")
    print("integrator's O(dt^2): more bond dimension buys nothing there, and")
    print("only a smaller time step helps, as the last row shows.")
    if scheme in ('1sp', '1-site-plus', 'one-site-plus'):
        print()
        print(f"If the last row did not improve, expand_k = {expand_k} is the")
        print("binding constraint rather than dt: the bonds never reached the")
        print("rank the melting asked for. Raise it and run the scan again.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "TDVP domain-wall melting on a free-fermion chain, compared against\n"
            "the exact correlation matrix (occupations and entanglement entropy).\n\n"
            "Examples:\n"
            "  uv run python examples/examples_tdvp/tdvp_domainwall.py\n"
            "  uv run python examples/examples_tdvp/tdvp_domainwall.py --L 24 --time 4\n"
            "  uv run python examples/examples_tdvp/tdvp_domainwall.py --bond-dim 256\n"
            "  uv run python examples/examples_tdvp/tdvp_domainwall.py --bond-scan\n"
            "  uv run python examples/examples_tdvp/tdvp_domainwall.py --scheme 1s\n"
            "  uv run python examples/examples_tdvp/tdvp_domainwall.py --scheme 1sp --expand-k 8\n"
            "  uv run python examples/examples_tdvp/tdvp_domainwall.py --checkpoint-dir /tmp/tdvp_ckpt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--L', type=int, default=20, metavar='N',
                   help='chain length (default: 20)')
    p.add_argument('--t', type=float, default=1.0, metavar='T',
                   help='nearest-neighbor hopping amplitude (default: 1.0)')
    p.add_argument('--time', type=float, default=3.0, metavar='T',
                   help='total evolution time (default: 3.0)')
    p.add_argument('--dt', type=float, default=0.05, metavar='DT',
                   help='time step (default: 0.05)')
    p.add_argument('--scheme', choices=['1s', '2s', '1sp'], default='2s',
                   help='TDVP update scheme: 2s (2-site, default), 1s '
                        '(1-site, frozen from a product state since it cannot '
                        'grow bonds), or 1sp (1-site-plus / CBE)')
    p.add_argument('--order', type=int, choices=[1, 2], default=2,
                   help='integrator order (default: 2)')
    p.add_argument('--bond-dim', type=int, default=128, metavar='D',
                   help='maximum bond dimension (default: 128)')
    p.add_argument('--trunc-thresh', type=float, default=1e-12, metavar='THR',
                   help='SVD truncation threshold (default: 1e-12)')
    p.add_argument('--expand-k', type=int, default=4, metavar='K',
                   help="complement directions per bond per update, for '1sp' "
                        '(default: 4)')
    p.add_argument('--expand-alpha', type=int, default=None, metavar='A',
                   help='internal bond of the cheap factorization inside the '
                        "'1sp' expansion (default: full bond)")
    p.add_argument('--krylov-dim', type=int, default=16, metavar='K',
                   help='maximum Krylov subspace dimension (default: 16)')
    p.add_argument('--snapshots', type=int, default=6, metavar='K',
                   help='number of intermediate reports (default: 6)')
    p.add_argument('--checkpoint-dir', default=None, metavar='PATH',
                   help='directory for tdvp.ckpt and evolution.ckpt '
                        '(default: current working directory)')
    p.add_argument('--env-cache-dir', default=None, metavar='PATH',
                   help='directory for spilling environment blocks to disk '
                        '(default: keep in memory)')
    p.add_argument('--bond-scan', action='store_true',
                   help='run the bond-dimension convergence scan instead of a '
                        'single evolution; defaults to a shorter chain and '
                        'time since it performs six full evolutions, and '
                        'requires a scheme whose bonds can grow')
    p.add_argument('--quiet', action='store_true',
                   help='suppress per-run output')
    p.add_argument('--quiet-log', action='store_true',
                   help='silence the per-step INFO log from the TDVP driver, '
                        'leaving only this script output')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    alice.configure_logging()
    if args.quiet_log:
        logging.getLogger('alice').setLevel(logging.WARNING)

    if args.bond_scan:
        bond_scan(
            L=args.L if args.L != 20 else 14,
            t=args.t,
            time=min(args.time, 2.0),
            dt=args.dt,
            scheme=args.scheme,
            order=args.order,
            expand_k=args.expand_k,
            expand_alpha=args.expand_alpha,
        )
    else:
        tdvp_domainwall(
            L=args.L,
            t=args.t,
            time=args.time,
            dt=args.dt,
            scheme=args.scheme,
            order=args.order,
            bond_dim=args.bond_dim,
            trunc_thresh=args.trunc_thresh,
            expand_k=args.expand_k,
            expand_alpha=args.expand_alpha,
            krylov_dim=args.krylov_dim,
            snapshots=args.snapshots,
            checkpoint_dir=args.checkpoint_dir,
            env_cache_dir=args.env_cache_dir,
            verbose=not args.quiet,
        )
