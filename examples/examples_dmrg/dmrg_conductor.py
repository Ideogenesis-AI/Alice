# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Alice library.
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


"""Example: DMRG ground state of the 1D Hubbard chain.

Runs DMRG to find the ground state of the Hubbard Hamiltonian

    H = -t Σ_{<i,j>,σ} (c†_{i,σ} c_{j,σ} + h.c.) + U Σ_i n_{i,↑} n_{i,↓}
        - (U/2) Σ_i n_i

for a chain of length L with open boundary conditions. The chemical potential
shift -U/2 restores particle-hole symmetry at half-filling (μ = 0). The
algorithm performs alternating left and right half-sweeps, optimising each
site tensor (1-site / 1-site-plus) or bond tensor (2-site) with the Davidson
eigensolver, until the energy converges.

The initial MPS is obtained from iterative diagonalization of the U = 0
(non-interacting) limit, which provides an excellent starting point for the
interacting system. The free-fermion initial energy per site at half-filling
converges to

    E₀/N = -4t/π ≈ -1.2732 t  (thermodynamic limit, U = 0).

Adding U > 0 increases the ground-state energy. For the infinite chain at
half-filling, the exact Hubbard energy per site can be obtained from the
Bethe Ansatz but requires numerical integration.

Run from the repository root:

    uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64
    uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64 --scheme 2s
    uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64 --scheme 1sp
    uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64 --scheme 1sp --expand-k 8
    uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64 --U 4.0
    uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64 --history
    uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64 --env-cache-dir /tmp/env_cache
    uv run python examples/examples_dmrg/dmrg_conductor.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tests" / "diagonaliztn"))

from conductor import iter_diag_band, exact_halffilling_energy_band  # pyright: ignore[reportMissingImports]

import alice
from alice import MPS, build_hamiltonian, build_interaction
from alice import dmrg


# ---------------------------------------------------------------------------
# Initial MPS construction
# ---------------------------------------------------------------------------

def _iter_diag_mps(
    L: int,
    bond_dim: int,
    t: float = 1.0,
    symmetry: str = 'U1,U1',
) -> MPS:
    """Build an initial MPS from iterative diagonalization of the free spinful chain.

    Uses `iter_diag_band` (conductor.py) to diagonalize the non-interacting
    (U = 0) spinful tight-binding chain site by site, keeping at most
    `bond_dim` states (multiplets for SU2) at each step. This free-fermion
    MPS is an excellent starting point for the interacting Hubbard problem.
    The result is placed in right-canonical form with `center == 0`.

    Parameters
    ----------
    L:
        Chain length.
    bond_dim:
        Maximum number of states kept per truncation step. For symmetries with
        SU2 spin rotation (`'U1,SU2'` or `'Z2,SU2'`) this counts SU(2)
        multiplets.
    t:
        Nearest-neighbor hopping amplitude forwarded to `iter_diag_band`.
    symmetry:
        Band symmetry. Accepted values: `'U1,U1'` (default), `'Z2,U1'`,
        `'U1,SU2'`, or `'Z2,SU2'`.

    Returns
    -------
    MPS
        Right-canonical MPS (`center == 0`).
    """
    _, _, tensors = iter_diag_band(
        N=L, Nkeep=bond_dim, t=t, symmetry=symmetry, verbose=False,
    )
    mps = MPS(tensors, center=None)
    mps.canonical(0)
    return mps


# ---------------------------------------------------------------------------
# Main DMRG function
# ---------------------------------------------------------------------------

def dmrg_conductor(
    L: int = 20,
    bond_dim: int = 64,
    n_sweeps: int = 20,
    t: float = 1.0,
    U: float = 0.0,
    mu: float = 0.0,
    symmetry: str = 'U1,U1',
    scheme: str = '1s',
    e_tol: float = 1e-8,
    davidson_tol: float = 1e-10,
    trunc_thresh: float = 1e-15,
    expand_k: int = 4,
    expand_alpha: int = None,
    env_cache_dir: str = None,
    env_async_io: bool = True,
    env_window: int = 2,
    verbose: bool = True,
) -> Tuple[dmrg.Summary, MPS]:
    """Run DMRG for the 1D Hubbard chain.

    The initial MPS is always obtained from iterative diagonalization of the
    non-interacting (U = 0) spinful tight-binding chain. This free-fermion
    state provides a good starting point for the interacting problem and
    typically allows convergence in a few sweeps.

    Parameters
    ----------
    L:
        Chain length.
    bond_dim:
        Maximum bond dimension. Also controls the number of states kept during
        iterative diagonalization of the initial MPS.
    n_sweeps:
        Maximum number of full sweeps (forward + backward half-sweep each).
    t:
        Nearest-neighbor hopping amplitude.
    U:
        On-site Coulomb repulsion. At `U = 0` the model reduces to the spinful
        tight-binding chain.
    mu:
        Chemical potential relative to half-filling. The shift -U/2 is applied
        automatically to enforce particle-hole symmetry at `mu = 0`.
    symmetry:
        Band symmetry exploited. Accepted values:

        * `'U1,U1'`  — particle number × spin-z (default)
        * `'Z2,U1'`  — fermion parity × spin-z
        * `'U1,SU2'` — particle number × full SU(2) spin rotation
        * `'Z2,SU2'` — fermion parity × full SU(2) spin rotation

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
    _VALID_SYMMETRIES = ('U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2')
    if symmetry not in _VALID_SYMMETRIES:
        raise ValueError(
            f"symmetry must be one of {_VALID_SYMMETRIES}, got {symmetry!r}"
        )

    # -----------------------------------------------------------------------
    # Build the Hamiltonian MPO
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
    interactions, spc, L_built = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, L_built, spc)

    # -----------------------------------------------------------------------
    # Build the initial MPS from iterative diagonalization (U = 0 limit)
    # -----------------------------------------------------------------------
    mps = _iter_diag_mps(L_built, bond_dim=bond_dim, t=t, symmetry=symmetry)

    if verbose:
        print(f"Chain length   : {L_built}")
        print(f"Symmetry       : {symmetry}")
        print(f"Hopping t      : {t}")
        print(f"Hubbard U      : {U}")
        print(f"Chemical pot μ : {mu}  (effective μ_eff = {mu + U / 2:.4f})")
        print(f"Initialization : iter_diag (U=0 free-fermion MPS)")
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
        print(f"Energy per site  : {summary.energy / L_built:.10f}")
        print(f"MPS bond dims    : {summary.bond_dims}")

        print()
        # The U=0 free-fermion exact energy serves as a lower bound and a
        # reference for how much the Hubbard interaction raises the energy.
        E_free_N   = exact_halffilling_energy_band(L_built, t)
        E_free_inf = -4.0 * t / np.pi
        print(f"Reference (U=0, finite, half-filling) : {E_free_N:.10f}")
        print(f"Reference E/N  (U=0, N→∞, half-fill) : {E_free_inf:.10f}")
        if U != 0.0:
            delta = summary.energy - E_free_N
            print(f"Energy shift from U                   : {delta:+.4e}  (U = {U})")

    return summary, mps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "DMRG ground-state energy for the 1D Hubbard chain.\n\n"
            "Examples:\n"
            "  uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64\n"
            "  uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64 --scheme 2s\n"
            "  uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64 --scheme 1sp\n"
            "  uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64 --scheme 1sp --expand-k 8\n"
            "  uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64 --U 4.0\n"
            "  uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64 --history\n"
            "  uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --bond-dim 64 --env-cache-dir /tmp/env_cache\n"
            "  uv run python examples/examples_dmrg/dmrg_conductor.py --L 20 --symmetry U1,SU2 --bond-dim 64"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--L', type=int, default=20, metavar='N',
        help='chain length (default: 20)',
    )
    p.add_argument(
        '--bond-dim', type=int, default=64, metavar='D',
        help='bond dimension (default: 64)',
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
        '--U', type=float, default=0.0, metavar='U',
        help='on-site Coulomb repulsion (default: 4.0)',
    )
    p.add_argument(
        '--mu', type=float, default=0.0, metavar='MU',
        help='chemical potential relative to half-filling (default: 0.0)',
    )
    p.add_argument(
        '--symmetry',
        choices=['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'],
        default='U1,U1',
        help='band symmetry exploited (default: U1,U1)',
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

    summary, _ = dmrg_conductor(
        L=args.L,
        bond_dim=args.bond_dim,
        n_sweeps=args.n_sweeps,
        t=args.t,
        U=args.U,
        mu=args.mu,
        symmetry=args.symmetry,
        scheme=args.scheme,
        e_tol=args.e_tol,
        davidson_tol=args.davidson_tol,
        trunc_thresh=args.trunc_thresh,
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
        E_free_N = exact_halffilling_energy_band(args.L, args.t)
        print(f"\nFinal energy          : {energies[-1]:.10f}")
        print(f"Ref. (U=0, half-fill) : {E_free_N:.10f}")
        print(f"Ref. E/N (N→∞, U=0)  : {-4.0 * args.t / np.pi:.10f}  (thermodynamic limit)")
