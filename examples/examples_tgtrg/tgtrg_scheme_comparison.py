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


"""Example: TgTRG's three update schemes compared on one cooling run.

The subject here is not the thermodynamics but the choice between `'1s'`,
`'1sp'` and `'2s'`. What separates them is where the bond dimension of ρ
comes from, and a cooling run makes that question unavoidable: ρ(τ₀) arrives
at the Hamiltonian's own rank, far below what a cold density matrix needs, so
a scheme that cannot grow a bond is stuck with a ρ that cannot represent the
answer.

The test case is the spinless free-fermion chain,

    H = -t Σ_j (c†_j c_{j+1} + h.c.),

cooled from the Taylor state ρ(τ₀) down the stacked schedule — exponential
out of the high-temperature regime, then linear. Its exact grand-canonical
log Z is a one-line sum over single-particle levels, so every scheme is
measured against the answer and not against another tensor network. See
`tgtrg_spinless.py` for this chain studied for its own sake, with the full
thermodynamics table.

The three answers:

- `'1s'` never decides, because it cannot: the update is confined to a
  manifold of *fixed* rank. Its bond dimension is whatever the Taylor state
  was handed to it with, and the error in log Z grows with β as ρ outgrows
  the manifold it is confined to.
- `'1sp'` decides before each update, expanding the bond by up to
  `expand_k` directions drawn from the complement of the space that bond
  currently spans, then cooling over the expanded space. The local
  propagator stays 1-site sized.
- `'2s'` decides after each update, as a by-product: it cools a pair at once
  and splits by SVD, and whatever survives truncation is the new bond. This
  needs no expansion parameters, making it the comparison.

What to watch in the output. The `D` column is the story — `'1s'` is flat
while the other two climb, and the `log Z err` column separates into a
fixed-rank branch and two adaptive ones that track the exact solution down to
the truncation floor. Unlike TDVP's real-time counterpart, nothing here is
silently frozen: the error is visible from the first cold step, because log Z
is measured against an exact answer at every β.

Run from the repository root:

    uv run python examples/examples_tgtrg/tgtrg_scheme_comparison.py
    uv run python examples/examples_tgtrg/tgtrg_scheme_comparison.py --L 12 --beta-max 8
    uv run python examples/examples_tgtrg/tgtrg_scheme_comparison.py --max-bond 16
    uv run python examples/examples_tgtrg/tgtrg_scheme_comparison.py --expand-k 2
    uv run python examples/examples_tgtrg/tgtrg_scheme_comparison.py --expand-k-scan
    uv run python examples/examples_tgtrg/tgtrg_scheme_comparison.py --help
"""

from __future__ import annotations

import logging
import math
import tempfile
from typing import List, Optional, Tuple

import alice
from alice.algorithm import tgtrg
from alice.network import build_hamiltonian, build_interaction
from alice.network.thermal import thermal_mpo

_SCHEMES = ('1s', '1sp', '2s')


# ---------------------------------------------------------------------------
# Exact reference
# ---------------------------------------------------------------------------

def exact_log_z(L: int, t: float, beta: float) -> float:
    """Exact grand-canonical log Z of the spinless OBC free-fermion chain.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.
    beta:
        Inverse temperature.

    Returns
    -------
    float
        `log Z(β) = Σ_k log(1 + exp(-β ε_k))` over
        `ε_k = -2t cos(kπ / (L+1))`.
    """
    return sum(
        math.log1p(math.exp(2.0 * beta * t * math.cos(k * math.pi / (L + 1))))
        for k in range(1, L + 1)
    )


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _chain(L: int, t: float):
    """Build the free-fermion MPO and its physical space.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.

    Returns
    -------
    tuple
        `(mpo, spc)`.
    """
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': L, 'bcx': 'OBC', 'n2x': True},
        'model': {'category': 'fermionic', 'label': 'FreeFermion',
                  'symmetry': 'U1', 't': t},
    }
    interactions, spc, geo = build_interaction(cfg)
    return build_hamiltonian(interactions, geo.L, spc), spc


def _cool(H, spc, scheme: str, tau0: float, exp_steps: int, tau: float,
          lin_steps: int, **shared) -> tgtrg.Summary:
    """Cool ρ(τ₀) down the stacked schedule under one scheme.

    The exponential phase runs first, then the linear one resumes from the
    `Artifact` it returned — all a stacked schedule takes. Checkpoints go to
    a scratch directory the caller never sees; this example is about the
    schemes, not the bookkeeping.

    Parameters
    ----------
    H:
        Hamiltonian MPO.
    spc:
        Physical space index, for the Taylor state.
    scheme:
        Update scheme to run.
    tau0:
        Inverse temperature of the Taylor state.
    exp_steps:
        Number of doubling steps.
    tau:
        Step size of the linear phase.
    lin_steps:
        Number of linear steps after the handover.
    **shared:
        Further `tgtrg.Options` fields common to every scheme.

    Returns
    -------
    Summary
        Thermodynamic history spanning both phases.
    """
    with tempfile.TemporaryDirectory() as scratch:
        common = dict(scheme=scheme, tau_0=tau0, checkpoint_dir=scratch,
                      save_artifacts=False, **shared)
        rho0 = thermal_mpo(H, tau0, common.get('taylor_order', 10), spc)
        state = tgtrg.Artifact(rho=rho0, beta=tau0, step=0)

        summary, artifact = tgtrg.run(
            state, H, tgtrg.Options(grid='exp', n_steps=exp_steps, **common))
        if lin_steps > 0:
            summary, _ = tgtrg.run(
                artifact, H,
                tgtrg.Options(grid='lin', tau=tau,
                              n_steps=exp_steps + lin_steps, **common))

    return summary


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def tgtrg_scheme_comparison(
    L: int = 10,
    t: float = 1.0,
    tau0: float = 2 ** -8,
    beta_switch: float = 2.0,
    tau: float = 0.5,
    beta_max: float = 6.0,
    order: int = 2,
    max_bond: Optional[int] = 32,
    expand_k: int = 4,
    expand_alpha: Optional[int] = 4,
    krylov_dim: int = 16,
    verbose: bool = True,
) -> dict:
    """Cool the same Taylor state under all three schemes and compare.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.
    tau0:
        Inverse temperature of the Taylor state.
    beta_switch:
        Inverse temperature at which the exponential phase hands over.
    tau:
        Step size of the linear phase.
    beta_max:
        Inverse temperature the linear phase stops at.
    order:
        Integrator order per step, `1` or `2`.
    max_bond:
        Maximum bond dimension of ρ, shared by all three schemes. For
        `'1s'` it fixes the rank outright, by compacting the Taylor state
        before the first step.
    expand_k:
        Complement directions added per bond per local update, for `'1sp'`.
    expand_alpha:
        Internal bond of the cheap factorization inside the expansion.
        `None` keeps the full bond.
    krylov_dim:
        Maximum Krylov dimension per local propagator.
    verbose:
        Print the per-scheme tables and the comparison.

    Returns
    -------
    dict
        Maps each scheme name to `(log Z error at β_max, largest bond
        dimension reached, largest discarded weight)`.
    """
    H, spc = _chain(L, t)

    exp_steps = max(0, math.ceil(math.log2(beta_switch / tau0)))
    beta_exp = tau0 * 2 ** exp_steps
    lin_steps = max(0, round((beta_max - beta_exp) / tau))
    beta_end = beta_exp + tau * lin_steps

    if verbose:
        print(f"Chain length  : {L}")
        print(f"Hopping t     : {t}")
        print(f"Schedule      : {exp_steps} doubling steps to β = {beta_exp:g}, "
              f"then {lin_steps} steps of τ = {tau:g} to β = {beta_end:g}")
        print(f"Integrator    : order {order}")
        print(f"Max bond dim  : {max_bond}")
        print(f"expand_k      : {expand_k}   expand_alpha: {expand_alpha}")
        print()

    shared = dict(order=order, max_bond=max_bond, expand_k=expand_k,
                  expand_alpha=expand_alpha, krylov_dim=krylov_dim)

    outcome = {}
    for scheme in _SCHEMES:
        summary = _cool(H, spc, scheme, tau0, exp_steps, tau, lin_steps,
                        **shared)

        if verbose:
            print(f"scheme = {scheme!r}")
            print(f"  {'beta':>8}  {'T':>8}  {'log Z':>14}  {'log Z err':>10}  "
                  f"{'f':>10}  {'D':>4}  {'dw':>9}")
            print('  ' + '-' * 72)
            for n, beta in enumerate(summary.betas):
                # One diagnostic per step, so the Taylor state has none.
                dw = summary.discarded_weights[n - 1] if n > 0 else 0.0
                print(f"  {beta:>8.4f}  {1.0 / beta:>8.4f}  "
                      f"{summary.log_z[n]:>14.8f}  "
                      f"{abs(summary.log_z[n] - exact_log_z(L, t, beta)):>10.2e}  "
                      f"{summary.free_energies[n]:>10.6f}  "
                      f"{summary.max_bonds[n]:>4d}  {dw:>9.2e}")
            print()

        outcome[scheme] = (
            abs(summary.log_z[-1] - exact_log_z(L, t, summary.betas[-1])),
            max(summary.max_bonds),
            max(summary.discarded_weights, default=0.0),
        )

    if verbose:
        print("Summary")
        print(f"  {'scheme':>7}  {'log Z err at β_max':>19}  {'largest D':>10}  "
              f"{'largest dw':>11}")
        print('  ' + '-' * 54)
        for scheme in _SCHEMES:
            err, bond, dw = outcome[scheme]
            print(f"  {scheme:>7}  {err:>19.3e}  {bond:>10d}  {dw:>11.2e}")
        print()
        print("The '1s' error is the rank it was never allowed to add: ρ(β)")
        print("outgrew the manifold, and no step size can buy that back —")
        print("only a larger max_bond, applied to the state before the run.")
        print("'1sp' and '2s' both climb to the cap and track the exact")
        print("log Z, '1sp' doing it with a 1-site-sized propagator.")

    return outcome


# ---------------------------------------------------------------------------
# How much expansion is enough
# ---------------------------------------------------------------------------

def expand_k_scan(
    L: int = 10,
    t: float = 1.0,
    tau0: float = 2 ** -8,
    beta_switch: float = 2.0,
    tau: float = 0.5,
    beta_max: float = 6.0,
    max_bond: Optional[int] = 32,
    ks: Tuple[int, ...] = (0, 1, 2, 4, 8),
    verbose: bool = True,
) -> List[Tuple[int, float, int]]:
    """Report accuracy and bond growth as `expand_k` is raised.

    `expand_k = 0` is included deliberately: it empties the complement and so
    must reproduce `'1s'` exactly, which makes it the control separating an
    error in the expansion from an error in the integrator around it.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.
    tau0:
        Inverse temperature of the Taylor state.
    beta_switch:
        Inverse temperature at which the exponential phase hands over.
    tau:
        Step size of the linear phase.
    beta_max:
        Inverse temperature the linear phase stops at.
    max_bond:
        Maximum bond dimension of ρ.
    ks:
        Values of `expand_k` to try.
    verbose:
        Print the scan table.

    Returns
    -------
    list of tuple
        `(expand_k, log Z error at β_max, largest bond dimension)` per value.
    """
    H, spc = _chain(L, t)

    exp_steps = max(0, math.ceil(math.log2(beta_switch / tau0)))
    beta_exp = tau0 * 2 ** exp_steps
    lin_steps = max(0, round((beta_max - beta_exp) / tau))

    if verbose:
        print(f"expand_k scan  (L = {L}, max_bond = {max_bond})")
        print(f"  {'expand_k':>8}  {'log Z err at β_max':>19}  {'largest D':>10}")
        print('  ' + '-' * 42)

    rows = []
    for k in ks:
        summary = _cool(H, spc, '1sp', tau0, exp_steps, tau, lin_steps,
                        max_bond=max_bond, expand_k=k, expand_alpha=k or None)
        err = abs(summary.log_z[-1] - exact_log_z(L, t, summary.betas[-1]))
        rows.append((k, err, max(summary.max_bonds)))
        if verbose:
            print(f"  {k:>8d}  {err:>19.3e}  {max(summary.max_bonds):>10d}")

    if verbose:
        print()
        print("k = 0 is the '1s' run under another name. Each further")
        print("direction buys accuracy until the bond reaches the cap, past")
        print("which only max_bond moves the floor.")

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "TgTRG's three update schemes compared on one cooling run.\n\n"
            "Examples:\n"
            "  uv run python examples/examples_tgtrg/tgtrg_scheme_comparison.py\n"
            "  uv run python examples/examples_tgtrg/tgtrg_scheme_comparison.py --L 12\n"
            "  uv run python examples/examples_tgtrg/tgtrg_scheme_comparison.py --max-bond 16\n"
            "  uv run python examples/examples_tgtrg/tgtrg_scheme_comparison.py --expand-k 2\n"
            "  uv run python examples/examples_tgtrg/tgtrg_scheme_comparison.py --expand-k-scan"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--L', type=int, default=10, metavar='N',
                   help='chain length (default: 10)')
    p.add_argument('--t', type=float, default=1.0, metavar='T',
                   help='nearest-neighbor hopping amplitude (default: 1.0)')
    p.add_argument('--tau0', type=float, default=2 ** -8, metavar='TAU',
                   help='inverse temperature of the Taylor state (default: 2**-8)')
    p.add_argument('--beta-switch', type=float, default=2.0, metavar='BETA',
                   help='where the exponential phase hands over (default: 2.0)')
    p.add_argument('--tau', type=float, default=0.5, metavar='TAU',
                   help='step size of the linear phase (default: 0.5)')
    p.add_argument('--beta-max', type=float, default=6.0, metavar='BETA',
                   help='inverse temperature to stop at (default: 6.0)')
    p.add_argument('--order', type=int, choices=[1, 2], default=2,
                   help='integrator order per step (default: 2)')
    p.add_argument('--max-bond', type=int, default=32, metavar='D',
                   help='maximum bond dimension, shared by all schemes (default: 32)')
    p.add_argument('--expand-k', type=int, default=4, metavar='K',
                   help="complement directions per bond for '1sp' (default: 4)")
    p.add_argument('--expand-alpha', type=int, default=4, metavar='A',
                   help='internal bond of the cheap factorization (default: 4)')
    p.add_argument('--krylov-dim', type=int, default=16, metavar='M',
                   help='maximum Krylov dimension per local propagator (default: 16)')
    p.add_argument('--expand-k-scan', action='store_true',
                   help='scan expand_k instead of comparing the three schemes')
    p.add_argument('--quiet', action='store_true',
                   help='suppress configuration and results output')
    p.add_argument('--verbose-log', action='store_true',
                   help='keep the per-step INFO log from the TgTRG driver, '
                        'which this script otherwise silences')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    alice.configure_logging()
    if not args.verbose_log:
        logging.getLogger('alice').setLevel(logging.WARNING)

    if args.expand_k_scan:
        expand_k_scan(
            L=args.L,
            t=args.t,
            tau0=args.tau0,
            beta_switch=args.beta_switch,
            tau=args.tau,
            beta_max=args.beta_max,
            max_bond=args.max_bond,
            verbose=not args.quiet,
        )
    else:
        tgtrg_scheme_comparison(
            L=args.L,
            t=args.t,
            tau0=args.tau0,
            beta_switch=args.beta_switch,
            tau=args.tau,
            beta_max=args.beta_max,
            order=args.order,
            max_bond=args.max_bond,
            expand_k=args.expand_k,
            expand_alpha=args.expand_alpha,
            krylov_dim=args.krylov_dim,
            verbose=not args.quiet,
        )
