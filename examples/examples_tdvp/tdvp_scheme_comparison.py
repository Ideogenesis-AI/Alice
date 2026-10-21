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


"""Example: TDVP's three update schemes compared on one quench.

The subject here is not the physics of the quench but the choice between
`'1s'`, `'1sp'` and `'2s'`. What separates them is where their bond dimension
comes from, and since an MPS represents only what its bonds allow, that one
difference decides which schemes can run this problem at all.

The test case is a free-fermion chain released from a half-filled product
state,

    H = -t Σ_j (c†_j c_{j+1} + h.c.),   |ψ(0)⟩ = |1 … 1 0 … 0⟩

chosen because it puts both hard parts in one run: every bond starts at
dimension 1, which a fixed-rank scheme can never leave, and the melting front
then keeps demanding more room as it advances. See `tdvp_domainwall.py` for
this state studied for its own sake — the entropy growth, the truncation error,
and the exact Gaussian solution in full. Here it is the instrument, not the
object.

The three answers:

- `'1s'` never decides, because it cannot: the update is confined to a
  manifold of *fixed* bond dimension. Nothing crosses a bond of dimension 1,
  so the state does not move, while the norm, the particle number and the
  energy stay conserved to machine precision and the run looks healthy.
- `'1sp'` decides before each update, by widening the bond ahead of it with
  directions drawn from the complement of the space that bond currently spans,
  up to `expand_k` of them. The local exponential stays 1-site sized.
- `'2s'` decides after each update, as a by-product: it evolves two sites at
  once and splits by SVD, and whatever survives truncation is the new bond.
  This needs no expansion parameters, which is what makes it the comparison.

Correctness is checked against the exact solution rather than against another
tensor network. A product state of free fermions stays Gaussian, so the
equal-time correlation matrix evolves as `C(t) = U C(0) U†` with
`U = exp(-i h t)` over the `L x L` hopping matrix, giving both the exact
occupations and the exact entanglement entropy.

What to watch in the output. The `bonds` column is the whole story — `'1s'`
stays at 1 forever, while `'1sp'` and `'2s'` climb, and `'1sp'` ends up
carrying less bond dimension than `'2s'` for the same accuracy. The energy,
the norm and the particle number are useless as diagnostics here, being
conserved exactly by all three including the frozen one, which is the point
worth taking away: on this failure mode the conservation laws say nothing.

Run from the repository root:

    uv run python examples/examples_tdvp/tdvp_scheme_comparison.py
    uv run python examples/examples_tdvp/tdvp_scheme_comparison.py --L 16 --time 4
    uv run python examples/examples_tdvp/tdvp_scheme_comparison.py --expand-k 1
    uv run python examples/examples_tdvp/tdvp_scheme_comparison.py --expand-k-scan
    uv run python examples/examples_tdvp/tdvp_scheme_comparison.py --dt-scan
    uv run python examples/examples_tdvp/tdvp_scheme_comparison.py --help
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
    exact_occupations,
    mps_entropy,
    mps_occupations,
    occupation_mpos,
)

_SCHEMES = ('1s', '1sp', '2s')


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _chain(L: int, t: float):
    """Build the free-fermion MPO, the occupation MPOs, and a state factory.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.

    Returns
    -------
    tuple
        `(mpo, n_obs, make_state)`, where `make_state(occ)` returns a fresh
        complex product state with the occupation pattern `occ`.
    """
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': L, 'bcx': 'OBC', 'n2x': True},
        'model': {'category': 'fermionic', 'label': 'FreeFermion',
                  'symmetry': 'U1', 't': t},
    }
    interactions, spc, geo = build_interaction(cfg)
    mpo = build_hamiltonian(interactions, geo.L, spc)

    _, ops = build_fermionic('U1')
    n_obs = occupation_mpos(geo.L, spc, ops)

    def make_state(occ):
        """Fresh product state with the given occupation pattern."""
        return init_mps(geo.L, spc, ops, bond_dim=1, config=list(occ),
                        dtype=torch.complex128)

    return mpo, n_obs, make_state


def _domain_wall(L: int) -> list:
    """Return the half-filled domain-wall occupation pattern."""
    return [1] * (L // 2) + [0] * (L - L // 2)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def tdvp_scheme_comparison(
    L: int = 12,
    t: float = 1.0,
    time: float = 2.0,
    dt: float = 0.05,
    order: int = 2,
    bond_dim: int = 64,
    expand_k: int = 4,
    expand_alpha: Optional[int] = None,
    krylov_dim: int = 16,
    snapshots: int = 5,
    verbose: bool = True,
) -> dict:
    """Evolve the same product state under all three schemes and compare.

    Parameters
    ----------
    L:
        Chain length. The left `L // 2` sites start occupied.
    t:
        Nearest-neighbor hopping amplitude.
    time:
        Total evolution time.
    dt:
        Time step.
    order:
        Integrator order, `1` or `2`.
    bond_dim:
        Maximum bond dimension.
    expand_k:
        Complement directions added per bond per local update, for `'1sp'`.
    expand_alpha:
        Internal bond of the cheap factorization inside the expansion.
        `None` keeps the full bond.
    krylov_dim:
        Maximum Krylov subspace dimension for the local propagators.
    snapshots:
        Number of intermediate times at which to report each trajectory.
    verbose:
        Print the per-scheme tables and the comparison.

    Returns
    -------
    dict
        Maps each scheme name to `(max occupation error, final bond profile,
        final occupations)`.
    """
    mpo, n_obs, make_state = _chain(L, t)
    occ0 = _domain_wall(L)
    half = L // 2
    n_total = max(1, round(time / dt))
    stride = max(1, n_total // max(1, snapshots))

    if verbose:
        print(f"Chain length  : {L}")
        print(f"Hopping t     : {t}")
        print(f"Initial state : |{''.join(str(n) for n in occ0)}⟩  "
              f"(product state, every bond dimension 1)")
        print(f"Time step     : {dt}   total time = {time}   order = {order}")
        print(f"expand_k      : {expand_k}   expand_alpha: {expand_alpha}")
        print()

    outcome = {}
    for scheme in _SCHEMES:
        opts = tdvp.Options(
            scheme=scheme, dt=dt, n_steps=n_total, order=order,
            max_bond=bond_dim, expand_k=expand_k, expand_alpha=expand_alpha,
            krylov_dim=krylov_dim, save_artifacts=False,
        )

        if verbose:
            print(f"scheme = {scheme!r}")
            print(f"  {'t':>6}  {'n_j err':>9}  {'S':>7}  {'S exact':>7}  "
                  f"{'E':>10}  {'norm':>8}  {'N':>7}  {'bonds':>5}  density")
            print('  ' + '-' * (71 + L))

        artifact = tdvp.Artifact(state=make_state(occ0), time=0.0, step=0)
        step = 0
        while step < n_total:
            step = min(n_total, step + stride)
            # Each segment resumes from the last, so the bond profile the
            # expansion has reached carries over as it would in one long run.
            _, artifact = tdvp.run(artifact, mpo,
                                   dataclasses.replace(opts, n_steps=step))
            occ = mps_occupations(artifact.state, n_obs)
            exact = exact_occupations(occ0, t, artifact.time)
            if verbose:
                M = correlation_matrix(occ0, t, artifact.time)
                print(f"  {artifact.time:>6.3f}  "
                      f"{np.max(np.abs(occ - exact)):>9.2e}  "
                      # Clamped because a product state's entropy comes back
                      # as a negative zero, which prints as -0.0000.
                      f"{max(0.0, mps_entropy(artifact.state, half)):>7.4f}  "
                      f"{exact_entropy(M, half):>7.4f}  "
                      f"{observe(artifact.state, mpo):>+10.2e}  "
                      f"{artifact.state.norm():>8.6f}  "
                      f"{occ.sum():>7.4f}  "
                      f"{max(artifact.state.bond_dims):>5d}  "
                      f"|{density_row(occ)}|")

        occ = mps_occupations(artifact.state, n_obs)
        exact = exact_occupations(occ0, t, artifact.time)
        outcome[scheme] = (float(np.max(np.abs(occ - exact))),
                           list(artifact.state.bond_dims), occ)
        if verbose:
            print()

    if verbose:
        print("Summary")
        print(f"  {'scheme':>7}  {'max |n_j - exact|':>19}  bond profile")
        print('  ' + '-' * 70)
        for scheme in _SCHEMES:
            err, bonds, _ = outcome[scheme]
            print(f"  {scheme:>7}  {err:>19.3e}  {bonds}")
        print()
        print("The '1s' error is the size of the dynamics it missed entirely:")
        print("its state is still the initial one, so refining dt will not")
        print("move it. Note that '1sp' reaches the accuracy of '2s' while")
        print("carrying a smaller bond profile.")
        print()

        exact = exact_occupations(occ0, t, time)
        window = list(range(max(0, half - 3), min(L, half + 4)))
        print(f"Occupations at t = {time:g}  (across the melting front)")
        print(error_table("n_j ('1sp')", outcome['1sp'][2], exact, window)[0])
        print()
        print("And the same sites under '2s':")
        print(error_table("n_j ('2s')", outcome['2s'][2], exact, window)[0])
        print()

    return outcome


# ---------------------------------------------------------------------------
# How much expansion is enough
# ---------------------------------------------------------------------------

def expand_k_scan(
    L: int = 12,
    t: float = 1.0,
    time: float = 2.0,
    dt: float = 0.05,
    ks: Tuple[int, ...] = (0, 1, 2, 4, 8),
) -> None:
    """Report accuracy and bond growth as `expand_k` is raised.

    `expand_k = 0` is included deliberately: it disables the expansion and so
    must reproduce `'1s'` exactly, which makes it the control that separates an
    error in the expansion from an error in the integrator around it.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Hopping amplitude.
    time:
        Total evolution time.
    dt:
        Time step.
    ks:
        Values of `expand_k` to try.
    """
    mpo, n_obs, make_state = _chain(L, t)
    occ0 = _domain_wall(L)
    exact = exact_occupations(occ0, t, time)
    n_total = max(1, round(time / dt))

    print(f"expand_k scan  (L = {L}, t_final = {time}, dt = {dt})")
    print("Reference: exact correlation matrix C(t) = U C(0) U†")
    print()
    print(f"{'expand_k':>9}  {'max |n_j - exact|':>19}  {'max bond':>9}  "
          f"bond profile")
    print('-' * 78)

    for k in ks:
        _, artifact = tdvp.run(
            tdvp.Artifact(state=make_state(occ0), time=0.0, step=0), mpo,
            tdvp.Options(scheme='1sp', dt=dt, n_steps=n_total, expand_k=k,
                         max_bond=64, save_artifacts=False),
        )
        occ = mps_occupations(artifact.state, n_obs)
        err = float(np.max(np.abs(occ - exact)))
        bonds = list(artifact.state.bond_dims)
        print(f"{k:>9}  {err:>19.3e}  {max(bonds):>9d}  {bonds}")

    print()
    print("`expand_k` is a cap, not a request. What a bond actually receives")
    print("is however many directions its complement holds above the noise")
    print("floor, and for a short-range H that number is small — on this")
    print("problem never more than two, which is why the table stops moving.")


# ---------------------------------------------------------------------------
# Time-step convergence
# ---------------------------------------------------------------------------

def dt_scan(
    L: int = 12,
    t: float = 1.0,
    time: float = 1.0,
    expand_k: int = 4,
    steps: Tuple[int, ...] = (10, 20, 40, 80),
) -> None:
    """Report the `'1sp'` error against the exact solution as `dt` is refined.

    The expansion is a per-step decision, so it is worth confirming that it
    adds no `dt`-independent error floor: once the bonds are open the error
    must fall by four per halving of the step, exactly as for `'1s'` and
    `'2s'`.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Hopping amplitude.
    time:
        Total evolution time, held fixed as the step count varies.
    expand_k:
        Complement directions added per bond per local update.
    steps:
        Step counts to use. Each entry doubles the resolution of the last.
    """
    mpo, n_obs, make_state = _chain(L, t)
    occ0 = _domain_wall(L)
    exact = exact_occupations(occ0, t, time)

    print(f"Time-step convergence of '1sp'  (L = {L}, t_final = {time}, "
          f"expand_k = {expand_k})")
    print("Reference: exact correlation matrix C(t) = U C(0) U†")
    print()
    print(f"{'steps':>7}  {'dt':>9}  {'max |n_j - exact|':>19}  {'ratio':>7}")
    print('-' * 50)

    previous = None
    for n_steps in steps:
        _, artifact = tdvp.run(
            tdvp.Artifact(state=make_state(occ0), time=0.0, step=0), mpo,
            tdvp.Options(scheme='1sp', dt=time / n_steps, n_steps=n_steps,
                         expand_k=expand_k, max_bond=64, save_artifacts=False),
        )
        err = float(np.max(np.abs(mps_occupations(artifact.state, n_obs) - exact)))
        ratio = '' if previous is None else f"{previous / err:>7.2f}"
        print(f"{n_steps:>7}  {time / n_steps:>9.5f}  {err:>19.3e}  {ratio}")
        previous = err

    print()
    print("Ratios near 4 mean the expansion contributes no error floor of")
    print("its own: what is left is the second-order integrator error.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "TDVP update schemes compared: one product state evolved under\n"
            "'1s', '1sp' and '2s', against the exact Gaussian solution.\n\n"
            "Examples:\n"
            "  uv run python examples/examples_tdvp/tdvp_scheme_comparison.py\n"
            "  uv run python examples/examples_tdvp/tdvp_scheme_comparison.py --L 16 --time 4\n"
            "  uv run python examples/examples_tdvp/tdvp_scheme_comparison.py --expand-k 1\n"
            "  uv run python examples/examples_tdvp/tdvp_scheme_comparison.py --expand-k-scan\n"
            "  uv run python examples/examples_tdvp/tdvp_scheme_comparison.py --dt-scan"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--L', type=int, default=12, metavar='N',
                   help='chain length (default: 12)')
    p.add_argument('--t', type=float, default=1.0, metavar='T',
                   help='nearest-neighbor hopping amplitude (default: 1.0)')
    p.add_argument('--time', type=float, default=2.0, metavar='T',
                   help='total evolution time (default: 2.0)')
    p.add_argument('--dt', type=float, default=0.05, metavar='DT',
                   help='time step (default: 0.05)')
    p.add_argument('--order', type=int, choices=[1, 2], default=2,
                   help='integrator order (default: 2)')
    p.add_argument('--bond-dim', type=int, default=64, metavar='D',
                   help='maximum bond dimension (default: 64)')
    p.add_argument('--expand-k', type=int, default=4, metavar='K',
                   help="complement directions per bond per update, for '1sp' "
                        '(default: 4)')
    p.add_argument('--expand-alpha', type=int, default=None, metavar='A',
                   help='internal bond of the cheap factorization inside the '
                        'expansion (default: full bond)')
    p.add_argument('--krylov-dim', type=int, default=16, metavar='K',
                   help='maximum Krylov subspace dimension (default: 16)')
    p.add_argument('--snapshots', type=int, default=5, metavar='K',
                   help='number of intermediate reports (default: 5)')
    p.add_argument('--expand-k-scan', action='store_true',
                   help='scan expand_k instead of comparing the schemes')
    p.add_argument('--dt-scan', action='store_true',
                   help="scan the time step for '1sp' instead of comparing "
                        'the schemes')
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

    if args.expand_k_scan:
        expand_k_scan(L=args.L, t=args.t, time=args.time, dt=args.dt)
    elif args.dt_scan:
        dt_scan(L=args.L, t=args.t, time=min(args.time, 1.0),
                expand_k=args.expand_k)
    else:
        tdvp_scheme_comparison(
            L=args.L,
            t=args.t,
            time=args.time,
            dt=args.dt,
            order=args.order,
            bond_dim=args.bond_dim,
            expand_k=args.expand_k,
            expand_alpha=args.expand_alpha,
            krylov_dim=args.krylov_dim,
            snapshots=args.snapshots,
            verbose=not args.quiet,
        )
