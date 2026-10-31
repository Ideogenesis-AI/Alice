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


"""Example: VUMPS on the Majumdar-Ghosh point of the J1-J2 spin chain.

Runs VUMPS to find the ground state of

    H = J1 Σ_i S_i · S_{i+1} + J2 Σ_i S_i · S_{i+2}

at the Majumdar-Ghosh point J2 = J1/2, where the ground state is known in
closed form: a product of nearest-neighbor singlets, with energy per site
exactly -3 J1 / 8 = -0.375.

This is the one example here whose exact ground state lies inside the
variational class. The dimer covering is a uniform MPS of bond dimension
two on a cell of two sites — one spin-1/2 multiplet per bond, so bond
dimension one when SU(2) is exploited — and VUMPS reproduces it to machine
precision rather than converging towards it. The gradient falls to
`grad_tol` instead of settling on the floor that a bond cap imposes in the
Heisenberg, free-fermion, and Hubbard examples, which is what convergence
looks like when nothing is being truncated.

The state is twofold degenerate: the dimers can cover the odd bonds or the
even ones, and a cell of two sites silently picks whichever the initial
state leans towards. Both have the same energy. That degeneracy is also why
the environment solver can exhaust its budget here at bond dimensions well
above the two the state needs — see the lessons page for the measurements.

Two symmetries are available via --symmetry. SU(2) reaches the exact energy
to around 1e-14; U(1) floors at about 9e-9, which is the precision of
Alice's spin tables rather than anything the algorithm did. Detuning --J2
away from 0.5 leaves the Majumdar-Ghosh point, where the dimer state is no
longer exact and the reference below no longer applies.

Run from the repository root:

    uv run python examples/examples_vumps/vumps_mgdimer.py
    uv run python examples/examples_vumps/vumps_mgdimer.py --symmetry U1
    uv run python examples/examples_vumps/vumps_mgdimer.py --init product
    uv run python examples/examples_vumps/vumps_mgdimer.py --cell 4 --history
    uv run python examples/examples_vumps/vumps_mgdimer.py --bond-dim 8
    uv run python examples/examples_vumps/vumps_mgdimer.py --help
"""

from __future__ import annotations

from typing import Tuple

import alice
from alice import build_ihamiltonian, build_interaction, iMPS, init_imps
from alice import vumps
from nicole import load_space


# ---------------------------------------------------------------------------
# Main VUMPS function
# ---------------------------------------------------------------------------

def vumps_mgdimer(
    N: int = 2,
    bond_dim: int = 2,
    n_iters: int = 30,
    J: float = 1.0,
    J2: float = 0.5,
    symmetry: str = 'SU2',
    scheme: str = '2s',
    grad_tol: float = 1e-10,
    davidson_tol: float = 1e-12,
    env_tol: float = 1e-12,
    env_maxiter: int = 200,
    trunc_thresh: float = 1e-15,
    init: str = 'random',
    seed: int = 42,
    env_cache_dir: str = None,
    env_async_io: bool = True,
    env_window: int = 2,
    verbose: bool = True,
) -> Tuple[vumps.Summary, iMPS]:
    """Run VUMPS for the J1-J2 chain, exact at the Majumdar-Ghosh point.

    Parameters
    ----------
    N:
        Number of sites in the unit cell, even so that the cell can be
        charge neutral and hold a whole number of dimers.
    bond_dim:
        Largest bond dimension a two-site split may keep. Two states, or
        one multiplet under SU(2), are enough for the exact ground state.
    n_iters:
        Maximum number of VUMPS iterations. Each one solves every position
        of the cell against one set of environments.
    J:
        Nearest-neighbor coupling `J1`.
    J2:
        Next-nearest-neighbor coupling. The Majumdar-Ghosh point is
        `J2 = J / 2`, the only value at which the dimer state is exact.
    symmetry:
        Symmetry exploited: `'SU2'` (default) or `'U1'`. SU(2) counts spin
        multiplets, so the exact state needs only `bond_dim=1` there.
    scheme:
        Update scheme: `'2s'` (two-site, default) or `'1s'` (one-site).
    grad_tol:
        Convergence criterion on the gradient norm `|AC - AL C|`. Nothing
        is truncated at the exact point, so this is reached rather than
        approached.
    davidson_tol:
        Residual norm tolerance for the Davidson eigensolver. Keep it well
        below `grad_tol`, which can never fall past it.
    env_tol:
        Convergence tolerance for the environment fixed point. The dimer
        state has zero correlation length beyond one bond, so the library
        default is kept here rather than loosened as on the critical
        chains.
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
    if symmetry not in ('U1', 'SU2'):
        raise ValueError(f"symmetry must be 'U1' or 'SU2', got {symmetry!r}")
    if init not in ('random', 'product'):
        raise ValueError(f"init must be 'random' or 'product', got {init!r}")

    at_mg_point = abs(J2 - J / 2.0) < 1e-12

    # -----------------------------------------------------------------------
    # Build the Hamiltonian iMPO over the unit cell
    # -----------------------------------------------------------------------
    cfg = {
        'geometry': {
            'lattice': 'chain', 'lx': N, 'bcx': 'IBC',
            'n2x': True, 'n3x': True,
        },
        'model': {
            'category': 'bosonic',
            'label': 'Heisenberg',
            'symmetry': symmetry,
            'spin': 0.5,
            'J': J,
            'Jp': J2,
        },
    }
    interactions, spc, geo = build_interaction(cfg)
    impo = build_ihamiltonian(interactions, geo.L, spc)

    # -----------------------------------------------------------------------
    # Build the initial unit cell
    # -----------------------------------------------------------------------
    Spc, Op = load_space('Spin', symmetry, {'J': 0.5})
    imps = init_imps(
        geo.L, Spc, Op,
        bond_dim=1 if init == 'product' else bond_dim,
        seed=seed,
    )

    if verbose:
        point = "Majumdar-Ghosh point" if at_mg_point else "detuned from J2 = J/2"
        print(f"Unit cell      : {geo.L} sites (infinite chain)")
        print(f"Couplings      : J1 = {J}  J2 = {J2}  ({point})")
        print(f"Symmetry       : {symmetry}")
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
        print(f"Energy per site: {summary.energy:.12f}")
        print(f"Gradient       : {summary.gradients[-1]:.4e}")
        print(f"iMPS bond dims : {summary.bond_dims}")

        print()
        if at_mg_point:
            E_exact = -3.0 * J / 8.0
            print(f"Exact E/N      : {E_exact:.12f}  (dimer product state)")
            print(f"VUMPS E/N      : {summary.energy:.12f}")
            print(f"Difference     : {summary.energy - E_exact:+.4e}")
        else:
            print(f"No closed form away from J2 = {J / 2.0}; nothing to compare against.")

    return summary, summary.state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "VUMPS at the Majumdar-Ghosh point of the J1-J2 spin-1/2 chain,\n"
            "where the exact ground state is a product of singlet dimers.\n\n"
            "Examples:\n"
            "  uv run python examples/examples_vumps/vumps_mgdimer.py\n"
            "  uv run python examples/examples_vumps/vumps_mgdimer.py --symmetry U1\n"
            "  uv run python examples/examples_vumps/vumps_mgdimer.py --init product\n"
            "  uv run python examples/examples_vumps/vumps_mgdimer.py --cell 4 --history\n"
            "  uv run python examples/examples_vumps/vumps_mgdimer.py --bond-dim 8"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--cell', type=int, default=2, metavar='N',
        help='number of sites in the unit cell, even (default: 2)',
    )
    p.add_argument(
        '--bond-dim', type=int, default=2, metavar='D',
        help='largest bond dimension kept; two is already exact (default: 2)',
    )
    p.add_argument(
        '--n-iters', type=int, default=30, metavar='K',
        help='maximum number of VUMPS iterations (default: 30)',
    )
    p.add_argument(
        '--J', type=float, default=1.0, metavar='J1',
        help='nearest-neighbor coupling (default: 1.0)',
    )
    p.add_argument(
        '--J2', type=float, default=0.5, metavar='J2',
        help='next-nearest-neighbor coupling; J/2 is the exact point (default: 0.5)',
    )
    p.add_argument(
        '--symmetry', choices=['SU2', 'U1'], default='SU2',
        help='symmetry group exploited (default: SU2)',
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
        '--env-tol', type=float, default=1e-12, metavar='TOL',
        help='environment fixed-point tolerance (default: 1e-12)',
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

    summary, _ = vumps_mgdimer(
        N=args.cell,
        bond_dim=args.bond_dim,
        n_iters=args.n_iters,
        J=args.J,
        J2=args.J2,
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
            print(f"{k + 1:>6}  {E:>18.12f}  {dE:>+14.4e}  {summary.gradients[k]:>12.4e}")
        print(f"\nFinal energy per site : {energies[-1]:.12f}")
        print(f"Exact  energy per site: {-3.0 * args.J / 8.0:.12f}")
