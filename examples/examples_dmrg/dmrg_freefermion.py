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


"""Example: DMRG ground state of the 1D spinless free-fermion chain.

Runs DMRG to find the ground state of the tight-binding Hamiltonian

    H = -t Σ_i (c†_i c_{i+1} + h.c.)

for a spinless fermion chain of length L with open boundary conditions. The
algorithm performs alternating left and right half-sweeps, optimizing each
site tensor (1-site / 1-site-plus) or bond tensor (2-site) with the Davidson
eigensolver, until the energy converges.

At half-filling (⌊L/2⌋ particles) the exact ground-state energy per site in
the thermodynamic limit is E/N = -2t/π ≈ -0.6366 t. The finite-chain exact
energy is computed analytically from the single-particle spectrum

    ε_k = -2t cos(kπ / (L+1)),  k = 1, …, ⌊L/2⌋.

Both initializations offered via --init come from `init_mps`:

- `random` (default): random MPS with group-derived bond sectors.
- `product`: deterministic bond-dimension-1 product state.

Neither carries the bond sectors of the ground state, which matters for the
update scheme. The 1-site scheme optimizes one site tensor at a time and so
leaves the bond sectors of the initial MPS untouched for the whole run: it
converges inside that restricted variational space and settles visibly above
the exact energy. Compare `--scheme 1s` against `--scheme 2s` or
`--scheme 1sp`, both of which let the sectors adapt and reach the exact
energy from either initialization.

Run from the repository root:

    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32
    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32 --scheme 2s
    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32 --scheme 1sp
    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32 --scheme 1sp --expand-k 8
    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 50 --bond-dim 64 --history
    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --init product --scheme 2s
    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32 --env-cache-dir /tmp/env_cache
    uv run python examples/examples_dmrg/dmrg_freefermion.py --help
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

import alice
from alice import MPS, build_hamiltonian, build_interaction, init_mps
from alice import dmrg
from nicole import load_space


def _exact_halffilling_energy(L: int, t: float = 1.0) -> float:
    """Exact ground-state energy of an L-site open tight-binding chain at half-filling.

    Fills the ⌊L/2⌋ lowest levels of the analytic single-particle spectrum
    ε_k = -2t cos(kπ / (L+1)), k = 1, …, L.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Hopping amplitude.

    Returns
    -------
    float
        Exact ground-state energy at half-filling (⌊L/2⌋ particles).
    """
    n_particles = L // 2
    if n_particles == 0:
        return 0.0
    k = np.arange(1, n_particles + 1)
    return float(-2 * t * np.sum(np.cos(k * np.pi / (L + 1))))


# ---------------------------------------------------------------------------
# Main DMRG function
# ---------------------------------------------------------------------------

def dmrg_freefermion(
    L: int = 20,
    bond_dim: int = 64,
    n_sweeps: int = 20,
    t: float = 1.0,
    symmetry: str = 'U1',
    scheme: str = '1s',
    e_tol: float = 1e-8,
    davidson_tol: float = 1e-10,
    trunc_thresh: float = 1e-15,
    init: str = 'random',
    seed: int = 42,
    expand_k: int = 4,
    expand_alpha: int = None,
    env_cache_dir: str = None,
    env_async_io: bool = True,
    env_window: int = 2,
    verbose: bool = True,
) -> Tuple[dmrg.Summary, MPS]:
    """Run DMRG for the spinless free-fermion tight-binding chain.

    Parameters
    ----------
    L:
        Chain length.
    bond_dim:
        Maximum bond dimension of the DMRG run. For `init='random'` it is also
        the budget distributed across the particle-number sectors of the
        initial MPS; `init='product'` always starts at bond dimension 1.
    n_sweeps:
        Maximum number of full sweeps (forward + backward half-sweep each).
    t:
        Nearest-neighbor hopping amplitude.
    symmetry:
        Symmetry exploited: `'U1'` (conserve particle number) or `'Z2'`
        (fermion parity only).
    scheme:
        DMRG update scheme: `'1s'` (1-site, default), `'2s'` (2-site), or
        `'1sp'` (1-site-plus / controlled bond expansion). The 2-site scheme
        optimizes a bond tensor at each step and uses SVD truncation to control
        the bond dimension. The 1-site-plus scheme grows the bond dimension
        cheaply via a complement isometry before each 1-site update.
    e_tol:
        Energy convergence threshold: stop when `|E_new - E_old| < e_tol`.
    davidson_tol:
        Residual norm tolerance for the Davidson eigensolver.
    trunc_thresh:
        SVD truncation threshold: singular values below this fraction of the
        largest singular value are discarded (default: 1e-15). Applied at
        every canonical step in the `'2s'` and `'1sp'` schemes.
    init:
        Initial MPS strategy, both built by `init_mps`: `'random'` (random
        tensors with group-derived bond sectors, default) or `'product'`
        (deterministic bond-dimension-1 product state). Neither reproduces
        the bond sectors of the ground state, so `scheme='1s'` converges
        above the exact energy; use `'2s'` or `'1sp'` to let them adapt.
    seed:
        Base random seed used when `init='random'`.
    expand_k:
        Maximum number of complement vectors added per bond end in the `'1sp'`
        scheme (default: 4). Ignored for `'1s'` and `'2s'`.
    expand_alpha:
        Internal bond dimension for the truncated 2-site tensor used in the
        `'1sp'` complement computation. `None` (default) keeps the full bond.
        Ignored for `'1s'` and `'2s'`.
    env_cache_dir:
        Directory for environment block cache files. When set, environment
        blocks are spilled to disk and only `env_window` blocks per direction
        are kept in memory. `None` (default) keeps all blocks in memory.
    env_async_io:
        Enable asynchronous disk I/O for environment caching (default `True`).
        Has no effect when `env_cache_dir` is `None`.
    env_window:
        Sliding-window size for in-memory environment blocks (default `2`).
        Has no effect when `env_cache_dir` is `None`.
    verbose:
        Print sweep-by-sweep progress and final summary when `True`.

    Returns
    -------
    Summary
        DMRG output: final energy, convergence flag, energy history, bond dims.
    MPS
        Optimized ground-state MPS.
    """
    if symmetry not in ('U1', 'Z2'):
        raise ValueError(f"symmetry must be 'U1' or 'Z2', got {symmetry!r}")
    if init not in ('random', 'product'):
        raise ValueError(f"init must be 'random' or 'product', got {init!r}")

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
        },
    }
    interactions, spc, geo = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, geo.L, spc)

    # -----------------------------------------------------------------------
    # Build the initial MPS
    # -----------------------------------------------------------------------
    Spc, Op = load_space('Ferm', symmetry)
    mps = init_mps(geo.L, Spc, Op, bond_dim=1 if init == 'product' else bond_dim, seed=seed)

    if verbose:
        print(f"Chain length   : {geo.L}")
        print(f"Symmetry       : {symmetry}")
        print(f"Hopping t      : {t}")
        print(f"Initialization : {init}")
        print(f"Scheme         : {scheme}")
        if scheme in ('1sp', '1-site-plus', 'one-site-plus'):
            alpha_str = str(expand_alpha) if expand_alpha is not None else 'full'
            print(f"  expand_k     : {expand_k}  expand_alpha : {alpha_str}")
        print(f"MPO bond dims  : {mpo.bond_dims}")
        print(f"Bond dim       : {bond_dim}")
        print(f"Initial MPS bond dims: {mps.bond_dims}")
        print(f"Max sweeps     : {n_sweeps}  e_tol = {e_tol:.2e}  trunc_thresh = {trunc_thresh:.2e}")
        if env_cache_dir is not None:
            print(f"Env cache dir  : {env_cache_dir}  (window={env_window}, async={env_async_io})")
        print()

    # -----------------------------------------------------------------------
    # DMRG run
    # -----------------------------------------------------------------------
    opts = dmrg.Options(
        scheme=scheme,
        n_sweeps=n_sweeps,
        max_bond=bond_dim,
        trunc_thresh=trunc_thresh,
        e_tol=e_tol,
        davidson_tol=davidson_tol,
        expand_k=expand_k,
        expand_alpha=expand_alpha,
        env_cache_dir=env_cache_dir,
        env_async_io=env_async_io,
        env_window=env_window,
    )
    summary = dmrg.run(mps, mpo, opts)

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    if verbose:
        status = "converged" if summary.converged else "not converged"
        print(f"Sweeps performed : {summary.n_sweeps}  ({status})")
        print(f"Ground-state E   : {summary.energy:.10f}")
        print(f"Energy per site  : {summary.energy / geo.L:.10f}")
        print(f"MPS bond dims    : {summary.bond_dims}")

        print()
        E_exact_N = _exact_halffilling_energy(geo.L, t)
        E_exact_inf = -2.0 * t / np.pi
        print(f"Exact E (finite, half-filling) : {E_exact_N:.10f}")
        print(f"DMRG  E                        : {summary.energy:.10f}")
        delta = summary.energy - E_exact_N
        print(f"Difference                     : {delta:+.4e}  (truncation error)")
        print()
        print(f"Exact E/N (N→∞, half-filling)  : {E_exact_inf:.10f}")
        print(f"DMRG  E/N                      : {summary.energy / geo.L:.10f}")

    return summary, mps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "DMRG ground-state energy for the 1D spinless free-fermion chain.\n\n"
            "Examples:\n"
            "  uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32\n"
            "  uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32 --scheme 2s\n"
            "  uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32 --scheme 1sp\n"
            "  uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32 --scheme 1sp --expand-k 8\n"
            "  uv run python examples/examples_dmrg/dmrg_freefermion.py --L 50 --bond-dim 64 --history\n"
            "  uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --init product --scheme 2s\n"
            "  uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32 --env-cache-dir /tmp/env_cache\n"
            "  uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --init random --seed 123"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--L', type=int, default=20, metavar='N',
        help='chain length (default: 20)',
    )
    p.add_argument(
        '--bond-dim', type=int, default=32, metavar='D',
        help='bond dimension (default: 32)',
    )
    p.add_argument(
        '--n-sweeps', type=int, default=20, metavar='K',
        help='maximum number of full DMRG sweeps (default: 20)',
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
        '--scheme', choices=['1s', '2s', '1sp'], default='1s',
        help='DMRG update scheme: 1s (1-site, default), 2s (2-site), or 1sp (1-site-plus / CBE)',
    )
    p.add_argument(
        '--expand-k', type=int, default=4, metavar='K',
        help='complement vectors per bond end for the 1sp scheme (default: 4)',
    )
    p.add_argument(
        '--expand-alpha', type=int, default=None, metavar='A',
        help=(
            'truncated internal bond dimension for the 1sp complement computation; '
            'None (default) keeps the full bond'
        ),
    )
    p.add_argument(
        '--e-tol', type=float, default=1e-8, metavar='TOL',
        help='energy convergence threshold (default: 1e-8)',
    )
    p.add_argument(
        '--davidson-tol', type=float, default=1e-10, metavar='TOL',
        help='Davidson eigensolver residual tolerance (default: 1e-10)',
    )
    p.add_argument(
        '--trunc-thresh', type=float, default=1e-15, metavar='THR',
        help='SVD truncation threshold (default: 1e-15)',
    )
    p.add_argument(
        '--init', choices=['random', 'product'], default='random',
        help=(
            'MPS initialization strategy: random (random tensors with '
            'group-derived bond sectors, default) or product (bond-dimension-1 '
            'product state)'
        ),
    )
    p.add_argument(
        '--seed', type=int, default=42, metavar='S',
        help='base random seed used when --init random (default: 42)',
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
        '--history', action='store_true',
        help='print a sweep-by-sweep energy convergence table after the run',
    )
    p.add_argument(
        '--quiet', action='store_true',
        help='suppress per-run output (overridden by --history)',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    alice.configure_logging()

    summary, _ = dmrg_freefermion(
        L=args.L,
        bond_dim=args.bond_dim,
        n_sweeps=args.n_sweeps,
        t=args.t,
        symmetry=args.symmetry,
        scheme=args.scheme,
        e_tol=args.e_tol,
        davidson_tol=args.davidson_tol,
        trunc_thresh=args.trunc_thresh,
        init=args.init,
        seed=args.seed,
        expand_k=args.expand_k,
        expand_alpha=args.expand_alpha,
        env_cache_dir=args.env_cache_dir,
        env_async_io=not args.no_env_async_io,
        env_window=args.env_window,
        verbose=not args.quiet,
    )

    if args.history:
        print()
        print(f"{'Sweep':>6}  {'Energy':>18}  {'Delta E':>14}")
        print('-' * 44)
        energies = summary.energies
        for k, E in enumerate(energies):
            dE = E - energies[k - 1] if k > 0 else float('nan')
            print(f"{k + 1:>6}  {E:>18.10f}  {dE:>+14.4e}")
        E_exact_N = _exact_halffilling_energy(args.L, args.t)
        print(f"\nFinal energy      : {energies[-1]:.10f}")
        print(f"Exact energy      : {E_exact_N:.10f}  (finite chain, half-filling)")
        print(f"Exact E/N (N→∞)   : {-2.0 * args.t / np.pi:.10f}  (thermodynamic limit)")
