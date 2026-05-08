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
algorithm performs alternating left and right half-sweeps, optimising each
site tensor (1-site / 1-site-plus) or bond tensor (2-site) with the Davidson
eigensolver, until the energy converges.

At half-filling (⌊L/2⌋ particles) the exact ground-state energy per site in
the thermodynamic limit is E/N = -2t/π ≈ -0.6366 t. The finite-chain exact
energy is computed analytically from the single-particle spectrum

    ε_k = -2t cos(kπ / (L+1)),  k = 1, …, ⌊L/2⌋.

Two initializations are supported via --init:

- `iter_diag` (default): iterative diagonalization, growing the chain
  site by site and keeping `bond_dim` states. Provides a good starting
  point and typically converges in one or two sweeps.
- `random`: random MPS with `bond_dim // n_sectors` states per particle-
  number sector (U1 symmetry only). Simpler to construct but requires more
  sweeps to converge.

Run from the repository root:

    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32
    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32 --scheme 2s
    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32 --scheme 1sp
    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32 --scheme 1sp --expand-k 8
    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 50 --bond-dim 64 --history
    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --init random --bond-dim 32
    uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --bond-dim 32 --env-cache-dir /tmp/env_cache
    uv run python examples/examples_dmrg/dmrg_freefermion.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tests" / "diagonaliztn"))

from fermionic import iter_diag_ferm, exact_halffilling_energy  # pyright: ignore[reportMissingImports]

import alice
from alice import MPS, build_hamiltonian, build_interaction
from alice import dmrg
from nicole import Direction, Tensor, load_space
from nicole.index import Index, Sector


# ---------------------------------------------------------------------------
# Initial MPS construction
# ---------------------------------------------------------------------------

def _random_mps(
    L: int,
    bond_dim: int,
    symmetry: str = 'U1',
    seed: int = 42,
) -> MPS:
    """Build a random MPS for the spinless free-fermion chain.

    Bond charge sectors cover particle numbers 0…⌊L/2⌋ + 4 (capped at L),
    distributing `bond_dim` states evenly across sectors.  Only U1 symmetry
    is supported; use `iter_diag` for Z2.

    Parameters
    ----------
    L:
        Chain length.
    bond_dim:
        Total number of states distributed across bond charge sectors.
    symmetry:
        Must be `'U1'` (conserve particle number). `'Z2'` is not supported
        for random initialization; use `init='iter_diag'` instead.
    seed:
        Base random seed for reproducibility.

    Returns
    -------
    MPS
        Right-canonical MPS (`center == 0`).

    Raises
    ------
    ValueError
        If `symmetry` is not `'U1'`.
    """
    if symmetry != 'U1':
        raise ValueError(
            f"random MPS initialization only supports 'U1' symmetry, got {symmetry!r}; "
            "use --init iter_diag for Z2"
        )

    Spc, Op = load_space('Ferm', symmetry)
    vac = Op["vac"]

    # Cover particle-number sectors from 0 to half-filling + a small buffer.
    Nmax = min(L // 2 + 4, L)
    bond_charges = tuple(range(0, Nmax + 1))

    dim_per_sector = max(1, bond_dim // len(bond_charges))
    bulk = Index(
        direction=Direction.IN,
        group=Spc.group,
        sectors=tuple(Sector(charge=q, dim=dim_per_sector) for q in bond_charges),
    )

    tensors = []
    for i in range(L):
        l_idx = vac if i == 0 else bulk
        r_idx = (vac if i == L - 1 else bulk).flip()
        T = Tensor.random(
            [l_idx, r_idx, Spc],
            seed=seed + i,
            itags=[f'A{i:02d}', f'A{i + 1:02d}', f's{i:02d}'],
        )
        tensors.append(T)

    mps = MPS(tensors, center=None)
    mps.canonical(0)
    return mps


def _iter_diag_mps(
    L: int,
    bond_dim: int,
    t: float = 1.0,
    symmetry: str = 'U1',
) -> MPS:
    """Build an initial MPS from iterative diagonalization of the free-fermion chain.

    Uses `iter_diag_ferm` (fermionic.py) to grow the chain site by site with
    isometry-based tensors, keeping at most `bond_dim` states at each step.
    The result is placed in right-canonical form with `center == 0` before
    being returned.

    Parameters
    ----------
    L:
        Chain length.
    bond_dim:
        Maximum number of states kept per truncation step.
    t:
        Nearest-neighbor hopping amplitude forwarded to `iter_diag_ferm`.
    symmetry:
        `'U1'` (conserve particle number) or `'Z2'` (fermion parity).

    Returns
    -------
    MPS
        Right-canonical MPS (`center == 0`).
    """
    _, _, tensors = iter_diag_ferm(
        N=L, Nkeep=bond_dim, t=t, symmetry=symmetry, verbose=False,
    )
    mps = MPS(tensors, center=None)
    mps.canonical(0)
    return mps


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
    init: str = 'iter_diag',
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
        Maximum bond dimension. For `init='iter_diag'` this is the number of
        states kept at each iterative step. For `init='random'` the total is
        distributed evenly across particle-number sectors.
    n_sweeps:
        Maximum number of full sweeps (forward + backward half-sweep each).
    t:
        Nearest-neighbor hopping amplitude.
    symmetry:
        Symmetry exploited: `'U1'` (conserve particle number) or `'Z2'`
        (fermion parity only). Random initialization is only available for
        `'U1'`; use `init='iter_diag'` with `'Z2'`.
    scheme:
        DMRG update scheme: `'1s'` (1-site, default), `'2s'` (2-site), or
        `'1sp'` (1-site-plus / controlled bond expansion). The 2-site scheme
        optimises a bond tensor at each step and uses SVD truncation to control
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
        Initial MPS strategy: `'iter_diag'` (iterative diagonalization,
        default) or `'random'` (random tensors, U1 only).
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
        Optimised ground-state MPS.
    """
    if symmetry not in ('U1', 'Z2'):
        raise ValueError(f"symmetry must be 'U1' or 'Z2', got {symmetry!r}")
    if init not in ('iter_diag', 'random'):
        raise ValueError(f"init must be 'iter_diag' or 'random', got {init!r}")

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
    if init == 'iter_diag':
        mps = _iter_diag_mps(geo.L, bond_dim=bond_dim, t=t, symmetry=symmetry)
    else:
        mps = _random_mps(geo.L, bond_dim=bond_dim, symmetry=symmetry, seed=seed)

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
        E_exact_N = exact_halffilling_energy(geo.L, t)
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
            "  uv run python examples/examples_dmrg/dmrg_freefermion.py --L 20 --init random --bond-dim 32\n"
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
        '--init', choices=['iter_diag', 'random'], default='iter_diag',
        help=(
            'MPS initialization strategy: iter_diag (iterative diagonalization,'
            ' default) or random (random tensors, U1 only)'
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
        E_exact_N = exact_halffilling_energy(args.L, args.t)
        print(f"\nFinal energy      : {energies[-1]:.10f}")
        print(f"Exact energy      : {E_exact_N:.10f}  (finite chain, half-filling)")
        print(f"Exact E/N (N→∞)   : {-2.0 * args.t / np.pi:.10f}  (thermodynamic limit)")
