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


"""Example: TDVP propagation of a wavepacket on a free-fermion chain.

Evolves a Gaussian wavepacket under the tight-binding Hamiltonian

    H = -t Σ_j (c†_j c_{j+1} + h.c.)

on an open chain of length L, and compares the site occupations against the
exact single-particle propagator `exp(-i h t)` at every reported time.

Why this is a useful check and not just a demonstration: the initial state is
a *superposition* with momentum `k0`, so the relative phases of the amplitudes
matter. The packet therefore travels, at the lattice group velocity

    v = dε/dk = 2 t sin(k0),   ε(k) = -2 t cos(k),

and it travels to the *right* for `k0 > 0` only because the propagator is
`exp(-i H t)` rather than `exp(+i H t)`. A sign error in the integrator would
still conserve the norm and the energy, and would still reproduce the exact
occupations of a *localized* particle (whose profile is symmetric in time),
but it would send this packet the wrong way. The reported center of mass is
what catches that.

The one-particle sector of a chain is spanned by L states, so the exact state
has bond dimension 2 at every bond for all time. The evolution is therefore
essentially unaffected by `--bond-dim`, and the residual error is purely the
integrator's `O(dt²)`. That makes this the example to run when checking the
time-step convergence; use `--dt-scan` to see the error fall as `dt²`.

Because the bonds are already 2-dimensional, the `1s` scheme works here just
as well as `2s` — unlike a quench from a product state, where `1s` is frozen.
See `tdvp_domainwall.py` for that contrast.

All three schemes are therefore available, and `1sp` is the undemanding case
for it: a bond that is already exact has nothing in its complement above the
truncation threshold, so the expansion finds no direction worth keeping and
the bond stays at 2. That makes this the place to confirm that the expansion
adds no error when none is needed — the `1sp` numbers must land on the `1s`
ones. `tdvp_scheme_comparison.py` is where the expansion has work to do.

Run from the repository root:

    uv run python examples/examples_tdvp/tdvp_freefermion.py
    uv run python examples/examples_tdvp/tdvp_freefermion.py --L 40 --time 8
    uv run python examples/examples_tdvp/tdvp_freefermion.py --scheme 1s
    uv run python examples/examples_tdvp/tdvp_freefermion.py --scheme 1sp --expand-k 4
    uv run python examples/examples_tdvp/tdvp_freefermion.py --k0 0.0
    uv run python examples/examples_tdvp/tdvp_freefermion.py --order 1
    uv run python examples/examples_tdvp/tdvp_freefermion.py --dt-scan
    uv run python examples/examples_tdvp/tdvp_freefermion.py --checkpoint-dir /tmp/tdvp_ckpt
    uv run python examples/examples_tdvp/tdvp_freefermion.py --help
"""

from __future__ import annotations

import dataclasses
import logging
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import alice
from alice import build_hamiltonian, build_interaction, observe, tdvp
from alice.physics.system import build_fermionic
from freefermion_exact import (  # pyright: ignore[reportMissingImports]
    density_row,
    error_table,
    exact_single_particle,
    gaussian_wavepacket,
    hopping_matrix,
    mps_occupations,
    occupation_mpos,
    wavepacket_mps,
)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def tdvp_freefermion(
    L: int = 24,
    t: float = 1.0,
    x0: Optional[float] = None,
    k0: float = math.pi / 4,
    sigma: float = 2.0,
    time: float = 4.0,
    dt: float = 0.05,
    scheme: str = '2s',
    order: int = 2,
    bond_dim: int = 32,
    trunc_thresh: float = 1e-12,
    expand_k: int = 4,
    expand_alpha: Optional[int] = None,
    krylov_dim: int = 16,
    snapshots: int = 8,
    checkpoint_dir: Optional[str] = None,
    env_cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[tdvp.Summary, tdvp.Artifact]:
    """Propagate a Gaussian wavepacket and compare against the exact result.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.
    x0:
        Initial center of the packet. Defaults to `L / 4`, which leaves room
        for a rightward packet to travel before it reflects off the far edge.
    k0:
        Mean momentum of the packet. The group velocity is `2 t sin(k0)`.
    sigma:
        Initial width of the packet, in sites.
    time:
        Total evolution time.
    dt:
        Time step. The error against the exact propagator scales as `dt²`.
    scheme:
        `'2s'`, `'1s'`, or `'1sp'`. All three work here, since the exact state
        needs only bond dimension 2 and the initial state already has it.
    order:
        Integrator order, `1` or `2`.
    bond_dim:
        Maximum bond dimension. Has essentially no effect in the
        single-particle sector, where bond dimension 2 is exact.
    trunc_thresh:
        SVD truncation threshold for the `'2s'` and `'1sp'` schemes.
    expand_k:
        Complement directions added per bond per local update, for `'1sp'`.
        Ignored by the other schemes. Nothing survives the truncation here, so
        this only changes how much work the expansion does.
    expand_alpha:
        Internal bond of the cheap factorization inside the expansion, for
        `'1sp'`. `None` keeps the full bond.
    krylov_dim:
        Maximum Krylov subspace dimension for the local propagators.
    snapshots:
        Number of intermediate times at which to report the profile.
    checkpoint_dir:
        Directory for `tdvp.ckpt` and `evolution.ckpt`. `None` uses the
        current working directory.
    env_cache_dir:
        Directory for spilling environment blocks to disk. `None` keeps them
        in memory.
    verbose:
        Print the profile table and the error summary.

    Returns
    -------
    Summary
        Trajectory history from the final evolution segment.
    Artifact
        Final state with its time and step index.
    """
    if x0 is None:
        x0 = L / 4.0

    # -----------------------------------------------------------------------
    # Hamiltonian, observables, and the initial wavepacket
    # -----------------------------------------------------------------------
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

    phys, ops = build_fermionic('U1')
    n_obs = occupation_mpos(geo.L, spc, ops)

    psi0 = gaussian_wavepacket(geo.L, x0, k0, sigma)
    mps = wavepacket_mps(psi0, phys, ops)

    sites = np.arange(geo.L)
    velocity = 2.0 * t * math.sin(k0)
    energy_exact = float(np.real(psi0.conj() @ hopping_matrix(geo.L, t) @ psi0))

    if verbose:
        print(f"Chain length   : {geo.L}")
        print(f"Hopping t      : {t}")
        print(f"Packet         : x0 = {x0:g}, k0 = {k0:.6f}, sigma = {sigma:g}")
        print(f"Group velocity : {velocity:+.6f}  "
              f"(packet moves {'right' if velocity > 0 else 'left' if velocity < 0 else 'nowhere'})")
        print(f"Scheme         : {scheme}   order = {order}")
        if scheme in ('1sp', '1-site-plus', 'one-site-plus'):
            alpha_str = str(expand_alpha) if expand_alpha is not None else 'full'
            print(f"  expand_k     : {expand_k}   expand_alpha : {alpha_str}")
        print(f"Time step      : {dt}   total time = {time}")
        print(f"Bond dim       : {bond_dim}")
        print(f"Initial MPS bond dims: {mps.bond_dims}")
        print()
        # The state construction is independent of the evolution, so verify it
        # before spending any time evolving.
        occ0 = mps_occupations(mps, n_obs)
        print("Initial state check")
        print(f"  n_j vs |psi_j|^2 : {np.max(np.abs(occ0 - np.abs(psi0) ** 2)):.3e}")
        print(f"  energy           : {observe(mps, mpo):+.12f}  "
              f"(exact {energy_exact:+.12f})")
        print(f"  particle number  : {occ0.sum():.12f}")
        print()

    # -----------------------------------------------------------------------
    # Evolve, reporting at `snapshots` intermediate times
    # -----------------------------------------------------------------------
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
        print(f"{'t':>7}  {'<x>':>8}  {'exact':>8}  {'n_j err':>10}  "
              f"{'E drift':>10}  {'N-1':>9}  density profile")
        print('-' * (60 + geo.L))

    # Each segment resumes from the previous one, which is exactly how `run`
    # is meant to be used for a long trajectory: the history accumulates in
    # evolution.ckpt and the state in tdvp.ckpt.
    artifact = tdvp.Artifact(state=mps, time=0.0, step=0)
    summary = None
    step = 0
    while step < n_total:
        step = min(n_total, step + stride)
        summary, artifact = tdvp.run(artifact, mpo,
                                     dataclasses.replace(opts, n_steps=step))

        occ = mps_occupations(artifact.state, n_obs)
        exact = exact_single_particle(psi0, t, artifact.time)
        if verbose:
            print(f"{artifact.time:>7.3f}  {occ @ sites:>8.4f}  "
                  f"{exact @ sites:>8.4f}  "
                  f"{np.max(np.abs(occ - exact)):>10.2e}  "
                  f"{observe(artifact.state, mpo) - energy_exact:>+10.2e}  "
                  f"{occ.sum() - 1.0:>+9.1e}  |{density_row(occ, vmax=0.3)}|")

    # -----------------------------------------------------------------------
    # Final comparison against the exact propagator
    # -----------------------------------------------------------------------
    if verbose:
        occ = mps_occupations(artifact.state, n_obs)
        exact = exact_single_particle(psi0, t, artifact.time)
        peak = int(np.argmax(exact))
        window = [j for j in range(max(0, peak - 4), min(geo.L, peak + 5))]

        print()
        print(f"Occupations at t = {artifact.time:g}  (around the packet peak)")
        table, max_err = error_table('n_j (TDVP)', occ, exact, window)
        print(table)
        print()
        print(f"Max |n_j - exact| over all sites : {max_err:.3e}")
        print(f"Center of mass  TDVP : {occ @ sites:.6f}")
        print(f"Center of mass  exact: {exact @ sites:.6f}")
        print(f"Ballistic estimate x0 + v t : {x0 + velocity * artifact.time:.6f}")
        print("  (the estimate is approximate: a packet of finite width "
              "spreads as it moves)")
        print()
        print(f"Energy  : {observe(artifact.state, mpo):+.12f}  "
              f"(exact {energy_exact:+.12f}, conserved)")
        print(f"Norm    : {artifact.state.norm():.12f}")
        print(f"Bond dims : {artifact.state.bond_dims}")
        print(f"Max Krylov error over the run : "
              f"{max(summary.krylov_errors):.2e}")

    return summary, artifact


# ---------------------------------------------------------------------------
# Time-step convergence scan
# ---------------------------------------------------------------------------

def dt_scan(
    L: int = 16,
    t: float = 1.0,
    k0: float = math.pi / 4,
    sigma: float = 2.0,
    time: float = 2.0,
    scheme: str = '2s',
    order: int = 2,
    bond_dim: int = 32,
    expand_k: int = 4,
    expand_alpha: Optional[int] = None,
    steps: Tuple[int, ...] = (10, 20, 40, 80),
) -> None:
    """Report the error against the exact propagator as `dt` is refined.

    The reference is analytic rather than a finer TDVP run, so a systematic
    error could not hide behind self-consistency. The error must fall by a
    factor of four per halving of `dt`.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Hopping amplitude.
    k0:
        Mean momentum of the packet.
    sigma:
        Packet width.
    time:
        Total evolution time, held fixed as the step count varies.
    scheme:
        `'1s'`, `'2s'`, or `'1sp'`.
    order:
        Integrator order.
    bond_dim:
        Maximum bond dimension.
    expand_k:
        Complement directions added per bond per local update, for `'1sp'`.
    expand_alpha:
        Internal bond of the cheap factorization inside the expansion, for
        `'1sp'`. `None` keeps the full bond.
    steps:
        Step counts to use. Each entry doubles the resolution of the last.
    """
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': L, 'bcx': 'OBC', 'n2x': True},
        'model': {'category': 'fermionic', 'label': 'FreeFermion',
                  'symmetry': 'U1', 't': t},
    }
    interactions, spc, geo = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, geo.L, spc)
    phys, ops = build_fermionic('U1')
    n_obs = occupation_mpos(geo.L, spc, ops)

    psi0 = gaussian_wavepacket(geo.L, L / 4.0, k0, sigma)
    exact = exact_single_particle(psi0, t, time)

    print(f"Time-step convergence  (L = {geo.L}, t_final = {time}, "
          f"scheme = {scheme}, order = {order})")
    print("Reference: exact single-particle propagator exp(-i h t)")
    print()
    print(f"{'steps':>7}  {'dt':>9}  {'max |n_j - exact|':>19}  {'ratio':>7}")
    print('-' * 50)

    previous = None
    for n_steps in steps:
        mps = wavepacket_mps(psi0, phys, ops)
        _, artifact = tdvp.run(
            tdvp.Artifact(state=mps, time=0.0, step=0), mpo,
            tdvp.Options(scheme=scheme, dt=time / n_steps, n_steps=n_steps,
                         order=order, max_bond=bond_dim, expand_k=expand_k,
                         expand_alpha=expand_alpha, save_artifacts=False),
        )
        err = float(np.max(np.abs(mps_occupations(artifact.state, n_obs) - exact)))
        ratio = '' if previous is None else f"{previous / err:>7.2f}"
        print(f"{n_steps:>7}  {time / n_steps:>9.5f}  {err:>19.3e}  {ratio}")
        previous = err

    print()
    print("A second-order integrator gives ratios near 4.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "TDVP propagation of a Gaussian wavepacket on a free-fermion chain,\n"
            "compared against the exact single-particle propagator.\n\n"
            "Examples:\n"
            "  uv run python examples/examples_tdvp/tdvp_freefermion.py\n"
            "  uv run python examples/examples_tdvp/tdvp_freefermion.py --L 40 --time 8\n"
            "  uv run python examples/examples_tdvp/tdvp_freefermion.py --scheme 1s\n"
            "  uv run python examples/examples_tdvp/tdvp_freefermion.py --scheme 1sp --expand-k 4\n"
            "  uv run python examples/examples_tdvp/tdvp_freefermion.py --k0 0.0\n"
            "  uv run python examples/examples_tdvp/tdvp_freefermion.py --order 1\n"
            "  uv run python examples/examples_tdvp/tdvp_freefermion.py --dt-scan\n"
            "  uv run python examples/examples_tdvp/tdvp_freefermion.py --checkpoint-dir /tmp/tdvp_ckpt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--L', type=int, default=24, metavar='N',
                   help='chain length (default: 24)')
    p.add_argument('--t', type=float, default=1.0, metavar='T',
                   help='nearest-neighbor hopping amplitude (default: 1.0)')
    p.add_argument('--x0', type=float, default=None, metavar='X',
                   help='initial packet center (default: L/4)')
    p.add_argument('--k0', type=float, default=math.pi / 4, metavar='K',
                   help='mean packet momentum; group velocity is 2t sin(k0) '
                        '(default: pi/4)')
    p.add_argument('--sigma', type=float, default=2.0, metavar='S',
                   help='initial packet width in sites (default: 2.0)')
    p.add_argument('--time', type=float, default=4.0, metavar='T',
                   help='total evolution time (default: 4.0)')
    p.add_argument('--dt', type=float, default=0.05, metavar='DT',
                   help='time step (default: 0.05)')
    p.add_argument('--scheme', choices=['1s', '2s', '1sp'], default='2s',
                   help='TDVP update scheme: 2s (2-site, default), 1s '
                        '(1-site), or 1sp (1-site-plus / CBE)')
    p.add_argument('--order', type=int, choices=[1, 2], default=2,
                   help='integrator order (default: 2)')
    p.add_argument('--bond-dim', type=int, default=32, metavar='D',
                   help='maximum bond dimension (default: 32)')
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
    p.add_argument('--snapshots', type=int, default=8, metavar='K',
                   help='number of intermediate reports (default: 8)')
    p.add_argument('--checkpoint-dir', default=None, metavar='PATH',
                   help='directory for tdvp.ckpt and evolution.ckpt '
                        '(default: current working directory)')
    p.add_argument('--env-cache-dir', default=None, metavar='PATH',
                   help='directory for spilling environment blocks to disk '
                        '(default: keep in memory)')
    p.add_argument('--dt-scan', action='store_true',
                   help='run the time-step convergence scan instead of a '
                        'single evolution')
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

    if args.dt_scan:
        dt_scan(
            L=args.L if args.L != 24 else 16,
            t=args.t,
            k0=args.k0,
            sigma=args.sigma,
            time=min(args.time, 2.0),
            scheme=args.scheme,
            order=args.order,
            bond_dim=args.bond_dim,
            expand_k=args.expand_k,
            expand_alpha=args.expand_alpha,
        )
    else:
        tdvp_freefermion(
            L=args.L,
            t=args.t,
            x0=args.x0,
            k0=args.k0,
            sigma=args.sigma,
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
