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


"""Example: VUMPS ground state of the 1D spinless free-fermion chain.

Runs VUMPS to find the ground state of the tight-binding Hamiltonian

    H = -t Σ_i (c†_i c_{i+1} + h.c.)

directly in the thermodynamic limit, on an infinite chain built by tiling a
unit cell of `N` sites. The exact energy per site at half filling is
E/N = -2t/π ≈ -0.6366 t, with no finite-size correction to subtract.

Half filling is not a setting here, it is what a unit cell is. The cell has
to carry no net charge for the bond charges to come back to themselves after
`N` sites, so a uniform state at any other filling is not representable on a
cell of this length, and `init_imps` refuses to build one. A doped state
needs a longer cell, which fixes the filling at a rational number of its own.

Two initializations are supported via --init:

- `random` (default): random unit cell of width `bond_dim`, each bond
  carrying the charges reachable at its position.
- `product`: deterministic cell of width one, an alternating occupied and
  empty pattern. Only the two-site scheme can widen it, so pair it with
  `--scheme 2s`.

Run from the repository root:

    uv run python examples/examples_vumps/vumps_freefermion.py --cell 2 --bond-dim 16
    uv run python examples/examples_vumps/vumps_freefermion.py --cell 2 --bond-dim 32 --n-iters 60
    uv run python examples/examples_vumps/vumps_freefermion.py --cell 4 --bond-dim 16 --history
    uv run python examples/examples_vumps/vumps_freefermion.py --cell 2 --bond-dim 16 --init product
    uv run python examples/examples_vumps/vumps_freefermion.py --cell 2 --bond-dim 16 --scheme 1s
    uv run python examples/examples_vumps/vumps_freefermion.py --help
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

import alice
from alice import build_ihamiltonian, build_interaction, iMPS, init_imps
from alice import vumps
from nicole import load_space


# ---------------------------------------------------------------------------
# Main VUMPS function
# ---------------------------------------------------------------------------

def vumps_freefermion(
    N: int = 2,
    bond_dim: int = 16,
    n_iters: int = 50,
    t: float = 1.0,
    symmetry: str = 'U1',
    scheme: str = '2s',
    grad_tol: float = 1e-10,
    davidson_tol: float = 1e-12,
    env_tol: float = 1e-8,
    env_maxiter: int = 200,
    trunc_thresh: float = 1e-15,
    init: str = 'random',
    seed: int = 42,
    env_cache_dir: str = None,
    env_async_io: bool = True,
    env_window: int = 2,
    verbose: bool = True,
) -> Tuple[vumps.Summary, iMPS]:
    """Run VUMPS for the spinless free-fermion chain at half filling.

    Parameters
    ----------
    N:
        Number of sites in the unit cell, even so that the cell can be
        charge neutral.
    bond_dim:
        Largest bond dimension a two-site split may keep, and the width of
        the initial cell when `init='random'`.
    n_iters:
        Maximum number of VUMPS iterations. Each one solves every position
        of the cell against one set of environments.
    t:
        Nearest-neighbor hopping amplitude.
    symmetry:
        Symmetry exploited: `'U1'` (conserve particle number) or `'Z2'`
        (conserve parity only).
    scheme:
        Update scheme: `'2s'` (two-site, default) or `'1s'` (one-site).
    grad_tol:
        Convergence criterion on the gradient norm `|AC - AL C|`.
    davidson_tol:
        Residual norm tolerance for the Davidson eigensolver. Keep it well
        below `grad_tol`, which can never fall past it.
    env_tol:
        Convergence tolerance for the environment fixed point. The chain
        is gapless, so reaching the library default of 1e-12 takes more
        cells than `env_maxiter` allows, and the energy is the same
        either way.
    env_maxiter:
        Maximum number of unit cells absorbed by the environment solver.
    trunc_thresh:
        Smallest singular value a two-site split may keep.
    init:
        Initial cell: `'random'` (default) or `'product'`, which starts at
        width one and needs `scheme='2s'` to grow.
    seed:
        Base random seed used when `init='random'`.
    env_cache_dir:
        Root directory for environment block cache files. `None` (default)
        keeps every block in memory.
    env_async_io:
        Enable asynchronous environment block I/O (default `True`). Has no
        effect when `env_cache_dir` is `None`.
    env_window:
        Number of right environment blocks kept in memory at once (default
        `2`). Has no effect when `env_cache_dir` is `None`.
    verbose:
        Print the setup and the final summary when `True`.

    Returns
    -------
    Summary
        VUMPS output: energy per site, convergence flag, histories.
    iMPS
        Optimized unit cell, gauge fixed.
    """
    if symmetry not in ('U1', 'Z2'):
        raise ValueError(f"symmetry must be 'U1' or 'Z2', got {symmetry!r}")
    if init not in ('random', 'product'):
        raise ValueError(f"init must be 'random' or 'product', got {init!r}")

    # -----------------------------------------------------------------------
    # Build the Hamiltonian iMPO over the unit cell
    # -----------------------------------------------------------------------
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': N, 'bcx': 'IBC', 'n2x': True},
        'model': {
            'category': 'fermionic',
            'label': 'FreeFermion',
            'symmetry': symmetry,
            't': t,
        },
    }
    interactions, spc, geo = build_interaction(cfg)
    impo = build_ihamiltonian(interactions, geo.L, spc)

    # -----------------------------------------------------------------------
    # Build the initial unit cell
    # -----------------------------------------------------------------------
    Spc, Op = load_space('Ferm', symmetry)
    imps = init_imps(
        geo.L, Spc, Op,
        bond_dim=1 if init == 'product' else bond_dim,
        seed=seed,
    )

    if verbose:
        print(f"Unit cell      : {geo.L} sites (infinite chain, half filling)")
        print(f"Symmetry       : {symmetry}  t = {t}")
        print(f"Initialization : {init}")
        print(f"Scheme         : {scheme}")
        print(f"iMPO bond dims : {impo.bond_dims}")
        print(f"Bond dim       : {bond_dim}")
        print(f"Initial iMPS bond dims: {imps.bond_dims}")
        print(f"Max iterations : {n_iters}  grad_tol = {grad_tol:.2e}  env_tol = {env_tol:.2e}")
        if env_cache_dir is not None:
            print(f"Env cache dir  : {env_cache_dir}  (window={env_window}, async={env_async_io})")
        print()

    # -----------------------------------------------------------------------
    # VUMPS run
    # -----------------------------------------------------------------------
    opts = vumps.Options(
        scheme=scheme,
        n_iters=n_iters,
        max_bond=bond_dim,
        trunc_thresh=trunc_thresh,
        grad_tol=grad_tol,
        davidson_tol=davidson_tol,
        env_tol=env_tol,
        env_maxiter=env_maxiter,
        env_cache_dir=env_cache_dir,
        env_async_io=env_async_io,
        env_window=env_window,
    )
    summary = vumps.run(imps, impo, opts)

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    if verbose:
        status = "converged" if summary.converged else "not converged"
        print(f"Iterations     : {summary.n_iters}  ({status})")
        print(f"Energy per site: {summary.energy:.10f}")
        print(f"Gradient       : {summary.gradients[-1]:.4e}")
        print(f"iMPS bond dims : {summary.bond_dims}")

        print()
        E_exact = -2.0 * t / np.pi
        print(f"Exact E/N      : {E_exact:.10f}")
        print(f"VUMPS E/N      : {summary.energy:.10f}")
        print(f"Difference     : {summary.energy - E_exact:+.4e}")

    return summary, summary.state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "VUMPS ground-state energy per site for the infinite free-fermion chain.\n\n"
            "Examples:\n"
            "  uv run python examples/examples_vumps/vumps_freefermion.py --cell 2 --bond-dim 16\n"
            "  uv run python examples/examples_vumps/vumps_freefermion.py --cell 2 --bond-dim 32 --n-iters 60\n"
            "  uv run python examples/examples_vumps/vumps_freefermion.py --cell 4 --bond-dim 16 --history\n"
            "  uv run python examples/examples_vumps/vumps_freefermion.py --cell 2 --bond-dim 16 --init product\n"
            "  uv run python examples/examples_vumps/vumps_freefermion.py --cell 2 --bond-dim 16 --scheme 1s\n"
            "  uv run python examples/examples_vumps/vumps_freefermion.py --cell 2 --bond-dim 16 --seed 123"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--cell', type=int, default=2, metavar='N',
        help='number of sites in the unit cell, even at half filling (default: 2)',
    )
    p.add_argument(
        '--bond-dim', type=int, default=16, metavar='D',
        help='largest bond dimension kept (default: 16)',
    )
    p.add_argument(
        '--n-iters', type=int, default=50, metavar='K',
        help='maximum number of VUMPS iterations (default: 50)',
    )
    p.add_argument(
        '--t', type=float, default=1.0, metavar='T',
        help='nearest-neighbor hopping amplitude (default: 1.0)',
    )
    p.add_argument(
        '--symmetry', choices=['U1', 'Z2'], default='U1',
        help='symmetry group exploited (default: U1)',
    )
    p.add_argument(
        '--scheme', choices=['1s', '2s'], default='2s',
        help='update scheme: 2s (two-site, default) or 1s (one-site)',
    )
    p.add_argument(
        '--grad-tol', type=float, default=1e-10, metavar='TOL',
        help='convergence criterion on the gradient norm (default: 1e-10)',
    )
    p.add_argument(
        '--davidson-tol', type=float, default=1e-12, metavar='TOL',
        help='Davidson eigensolver residual tolerance (default: 1e-12)',
    )
    p.add_argument(
        '--env-tol', type=float, default=1e-8, metavar='TOL',
        help='environment fixed-point tolerance (default: 1e-8)',
    )
    p.add_argument(
        '--env-maxiter', type=int, default=200, metavar='K',
        help='maximum unit cells absorbed by the environment solver (default: 200)',
    )
    p.add_argument(
        '--trunc-thresh', type=float, default=1e-15, metavar='THR',
        help='SVD truncation threshold for two-site splits (default: 1e-15)',
    )
    p.add_argument(
        '--init', choices=['random', 'product'], default='random',
        help=(
            'initial unit cell: random (default) or product, which starts at '
            'width one and needs --scheme 2s to grow'
        ),
    )
    p.add_argument(
        '--seed', type=int, default=42, metavar='S',
        help='base random seed used when --init random (default: 42)',
    )
    p.add_argument(
        '--env-cache-dir', default=None, metavar='PATH',
        help=(
            'root directory for environment block cache files; enables '
            'disk-spilling so only --env-window right blocks are kept in '
            'memory (default: disabled, all blocks kept in memory)'
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
        '--history', action='store_true',
        help='print an iteration-by-iteration convergence table after the run',
    )
    p.add_argument(
        '--quiet', action='store_true',
        help='suppress per-run output (overridden by --history)',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    alice.configure_logging()

    summary, _ = vumps_freefermion(
        N=args.cell,
        bond_dim=args.bond_dim,
        n_iters=args.n_iters,
        t=args.t,
        symmetry=args.symmetry,
        scheme=args.scheme,
        grad_tol=args.grad_tol,
        davidson_tol=args.davidson_tol,
        env_tol=args.env_tol,
        env_maxiter=args.env_maxiter,
        trunc_thresh=args.trunc_thresh,
        init=args.init,
        seed=args.seed,
        env_cache_dir=args.env_cache_dir,
        env_async_io=not args.no_env_async_io,
        env_window=args.env_window,
        verbose=not args.quiet,
    )

    if args.history:
        print()
        print(f"{'Iter':>6}  {'Energy/site':>18}  {'Delta E':>14}  {'Gradient':>12}")
        print('-' * 58)
        energies = summary.energies
        for k, E in enumerate(energies):
            dE = E - energies[k - 1] if k > 0 else float('nan')
            print(f"{k + 1:>6}  {E:>18.10f}  {dE:>+14.4e}  {summary.gradients[k]:>12.4e}")
        print(f"\nFinal energy per site : {energies[-1]:.10f}")
        print(f"Exact  energy per site: {-2.0 / np.pi:.10f}")
