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


"""Example: VUMPS ground state of the 1D Hubbard chain at half filling.

Runs VUMPS to find the ground state of

    H = -t Σ_<ij>σ (c†_iσ c_jσ + h.c.) + U Σ_i n_i↑ n_i↓ - (U/2) Σ_i n_i

directly in the thermodynamic limit, on an infinite chain built by tiling a
unit cell of `N` sites. The chain is Bethe-ansatz integrable, so the answer
is known exactly at any `U` through the Lieb-Wu solution: at half filling
the ground-state energy per site is the single integral

    e(U) = -4t ∫_0^∞ dω J_0(ω) J_1(ω) / (ω (1 + exp(ω U / 2t)))

evaluated here by quadrature in `lieb_wu_energy`. At `U = 0` it collapses to
-4t/π ≈ -1.2732 t, twice the spinless chain, and the chain is a metal; at any
`U > 0` it is a Mott insulator with a charge gap.

The -(U/2) Σ n term is the half-filling chemical potential that Alice's
Hubbard builder applies to keep the model particle-hole symmetric. It
contributes exactly -U/2 per site at <n> = 1, so the energy VUMPS reports
sits that far below the Lieb-Wu value, and the comparison below shifts the
reference to match.

Half filling is not a setting here, it is what a unit cell is. The cell has
to carry no net charge for the bond charges to come back to themselves after
`N` sites, which for a band means one electron per site on average.

All four band symmetries are available via --symmetry. The two SU(2) groups
count spin multiplets rather than states, so a given `--bond-dim` buys more
of the Hilbert space there, and `Z2,SU2` is both the coarsest and the
fastest of the four.

Two initializations are supported via --init:

- `product` (default): deterministic cell of width one, one electron per
  site, widened by the two-site scheme over the first few iterations.
- `random`: random unit cell of width `bond_dim`, each bond carrying the
  charges reachable at its position.

The product start is the default here, unlike in the spin and spinless
examples. With four states per site a random cell of a given width spreads
its weight over many more sectors than the ground state uses, and the
two-site splits spend their iterations undoing that; growing from width one
lets them choose the sectors instead. At `--bond-dim 16` the product start
converges to within 2.6e-4 of the exact energy while the random start
stalls above 1.2e-3, and it gets there in fewer iterations.

Run from the repository root:

    uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 16
    uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 16 --U 0
    uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 32 --n-iters 30
    uv run python examples/examples_vumps/vumps_hubbard.py --symmetry U1,SU2 --bond-dim 16
    uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 16 --init random
    uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 16 --history
    uv run python examples/examples_vumps/vumps_hubbard.py --help
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

import alice
from alice import build_ihamiltonian, build_interaction, iMPS, init_imps
from alice import vumps
from nicole import load_space


_VALID_SYMMETRIES = ('U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2')


# ---------------------------------------------------------------------------
# Exact reference: the Lieb-Wu integral
# ---------------------------------------------------------------------------

def _bessel_j(
    n: int,
    x: np.ndarray,
    n_phase: int = 1024,
    chunk: int = 4096,
) -> np.ndarray:
    """Bessel function of the first kind on a grid, without SciPy.

    Evaluates `J_n(x) = (1/2π) ∫_0^2π cos(n θ - x sin θ) dθ` by the
    trapezoidal rule, which converges exponentially on a periodic
    integrand. `n_phase` points resolve arguments well past the range this
    example needs.

    Parameters
    ----------
    n:
        Order of the Bessel function.
    x:
        Arguments, strictly positive.
    n_phase:
        Phase samples per period.
    chunk:
        Arguments evaluated per block, to bound peak memory.

    Returns
    -------
    numpy.ndarray
        `J_n(x)`, same shape as `x`.
    """
    phase = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    sin_phase = np.sin(phase)
    out = np.empty_like(x)
    for i in range(0, x.size, chunk):
        xi = x[i:i + chunk]
        out[i:i + chunk] = np.cos(
            n * phase[None, :] - xi[:, None] * sin_phase[None, :]
        ).mean(axis=1)
    return out


def lieb_wu_energy(U: float, t: float = 1.0, d_omega: float = 0.002) -> float:
    """Exact ground-state energy per site of the half-filled Hubbard chain.

    Evaluates the Lieb-Wu integral by quadrature. The integrand decays like
    `exp(-ω U / 2t)`, so the cutoff is placed where that factor is
    negligible; at `U = 0` the integral has the closed form `-4t/π` and is
    returned directly.

    Parameters
    ----------
    U:
        On-site Coulomb repulsion. Non-negative.
    t:
        Nearest-neighbor hopping amplitude.
    d_omega:
        Quadrature spacing. The default gives about six digits.

    Returns
    -------
    float
        Energy per site, excluding the -U/2 chemical-potential shift that
        Alice's Hubbard builder applies.
    """
    if U < 0.0:
        raise ValueError(f"U must be non-negative, got {U}")
    if U == 0.0:
        return -4.0 * t / np.pi

    omega_max = min(40.0 * (2.0 * t / U), 2000.0)
    omega = np.arange(0, int(omega_max / d_omega) + 1) * d_omega
    decay = 1.0 / (1.0 + np.exp(np.clip(omega * U / (2.0 * t), 0.0, 700.0)))

    # The integrand is finite at the origin: J_1(ω) ~ ω/2 cancels the 1/ω.
    f = np.empty_like(omega)
    f[0] = 0.25
    f[1:] = (
        _bessel_j(0, omega[1:]) * _bessel_j(1, omega[1:])
        * decay[1:] / omega[1:]
    )
    return float(-4.0 * t * np.trapezoid(f, omega))


# ---------------------------------------------------------------------------
# Main VUMPS function
# ---------------------------------------------------------------------------

def vumps_hubbard(
    N: int = 2,
    bond_dim: int = 16,
    n_iters: int = 20,
    t: float = 1.0,
    U: float = 4.0,
    symmetry: str = 'Z2,SU2',
    scheme: str = '2s',
    grad_tol: float = 1e-10,
    davidson_tol: float = 1e-12,
    env_tol: float = 1e-8,
    env_maxiter: int = 200,
    trunc_thresh: float = 1e-15,
    init: str = 'product',
    seed: int = 42,
    env_cache_dir: str = None,
    env_async_io: bool = True,
    env_window: int = 2,
    verbose: bool = True,
) -> Tuple[vumps.Summary, iMPS]:
    """Run VUMPS for the Hubbard chain at half filling.

    Parameters
    ----------
    N:
        Number of sites in the unit cell, even so that the cell can be
        charge neutral.
    bond_dim:
        Largest bond dimension a two-site split may keep, and the width of
        the initial cell when `init='random'`. Counted in multiplets under
        the SU(2) symmetries.
    n_iters:
        Maximum number of VUMPS iterations. Each one solves every position
        of the cell against one set of environments.
    t:
        Nearest-neighbor hopping amplitude.
    U:
        On-site Coulomb repulsion. `U = 0` is the metallic spinful
        tight-binding chain; any `U > 0` opens a Mott gap.
    symmetry:
        Band symmetry exploited:

        * `'U1,U1'` — particle number × spin-z
        * `'Z2,U1'` — fermion parity × spin-z
        * `'U1,SU2'` — particle number × full SU(2) spin rotation
        * `'Z2,SU2'` — fermion parity × full SU(2) spin rotation (default,
          and the cheapest per iteration)

    scheme:
        Update scheme: `'2s'` (two-site, default) or `'1s'` (one-site).
    grad_tol:
        Convergence criterion on the gradient norm `|AC - AL C|`.
    davidson_tol:
        Residual norm tolerance for the Davidson eigensolver. Keep it well
        below `grad_tol`, which can never fall past it.
    env_tol:
        Convergence tolerance for the environment fixed point. The default
        is loose enough for the metallic case at `U = 0`, where the
        correlation length is cut off by the bond dimension rather than by
        a gap; at `U > 0` the library default of 1e-12 is within reach.
    env_maxiter:
        Maximum number of unit cells absorbed by the environment solver.
    trunc_thresh:
        Smallest singular value a two-site split may keep.
    init:
        Initial cell: `'product'` (default), which starts at width one and
        needs `scheme='2s'` to grow, or `'random'`.
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
    if symmetry not in _VALID_SYMMETRIES:
        raise ValueError(
            f"symmetry must be one of {_VALID_SYMMETRIES}, got {symmetry!r}"
        )
    if init not in ('random', 'product'):
        raise ValueError(f"init must be 'random' or 'product', got {init!r}")

    # -----------------------------------------------------------------------
    # Build the Hamiltonian iMPO over the unit cell
    # -----------------------------------------------------------------------
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': N, 'bcx': 'IBC', 'n2x': True},
        'model': {
            'category': 'conductor',
            'label': 'Hubbard',
            'symmetry': symmetry,
            't': t,
            'U': U,
            'mu': 0.0,
        },
    }
    interactions, spc, geo = build_interaction(cfg)
    impo = build_ihamiltonian(interactions, geo.L, spc)

    # -----------------------------------------------------------------------
    # Build the initial unit cell
    # -----------------------------------------------------------------------
    Spc, Op = load_space('Band', symmetry)
    imps = init_imps(
        geo.L, Spc, Op,
        bond_dim=1 if init == 'product' else bond_dim,
        seed=seed,
    )

    if verbose:
        print(f"Unit cell      : {geo.L} sites (infinite chain, half filling)")
        print(f"Symmetry       : {symmetry}  t = {t}  U = {U}")
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
        E_exact = lieb_wu_energy(U, t) - U / 2.0
        print(f"Lieb-Wu E/N    : {lieb_wu_energy(U, t):.10f}  (before the -U/2 shift)")
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
            "VUMPS ground-state energy per site for the infinite Hubbard chain\n"
            "at half filling, against the Lieb-Wu exact solution.\n\n"
            "Examples:\n"
            "  uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 16\n"
            "  uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 16 --U 0\n"
            "  uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 32 --n-iters 30\n"
            "  uv run python examples/examples_vumps/vumps_hubbard.py --symmetry U1,SU2 --bond-dim 16\n"
            "  uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 16 --history"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--cell', type=int, default=2, metavar='N',
        help='number of sites in the unit cell, even at half filling (default: 2)',
    )
    p.add_argument(
        '--bond-dim', type=int, default=16, metavar='D',
        help='largest bond dimension kept, in multiplets under SU(2) (default: 16)',
    )
    p.add_argument(
        '--n-iters', type=int, default=20, metavar='K',
        help='maximum number of VUMPS iterations (default: 20)',
    )
    p.add_argument(
        '--t', type=float, default=1.0, metavar='T',
        help='nearest-neighbor hopping amplitude (default: 1.0)',
    )
    p.add_argument(
        '--U', type=float, default=4.0, metavar='U',
        help='on-site Coulomb repulsion; 0 is the metallic case (default: 4.0)',
    )
    p.add_argument(
        '--symmetry', choices=list(_VALID_SYMMETRIES), default='Z2,SU2',
        help='band symmetry exploited (default: Z2,SU2, the cheapest)',
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
        '--init', choices=['product', 'random'], default='product',
        help=(
            'initial unit cell: product (default), which starts at width one '
            'and needs --scheme 2s to grow, or random'
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

    summary, _ = vumps_hubbard(
        N=args.cell,
        bond_dim=args.bond_dim,
        n_iters=args.n_iters,
        t=args.t,
        U=args.U,
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
        print(f"Exact  energy per site: {lieb_wu_energy(args.U, args.t) - args.U / 2.0:.10f}")
