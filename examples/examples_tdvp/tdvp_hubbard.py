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


"""Example: TDVP on the spinful Hubbard chain, Abelian and non-Abelian.

Melts a doubly-occupied/empty domain wall

    |ψ(0)⟩ = |↑↓ ↑↓ … ↑↓   0 0 … 0⟩

under the Hubbard Hamiltonian

    H = -t Σ_{j,σ} (c†_{j,σ} c_{j+1,σ} + h.c.) + U Σ_j n_{j,↑} n_{j,↓}

on an open chain of length L, with the symmetry group selectable so that the
*same physics* can be run through the Abelian and the non-Abelian code paths
and compared.

Why this initial state. Every site is a spin singlet — empty or doubly
occupied — so the state carries total spin zero and is representable in the
SU(2) multiplet basis. That is not a free choice: a spin-polarized state like
|↑ ↑ ↑ 0 0 0⟩ is *not* an eigenstate of total spin, so it has no SU(2)
representation at all, and a Néel state is no better.

Why `U = 0` by default. At `U = 0` the two spin channels decouple and each one
is an independent spinless free-fermion chain seeing the very same domain
wall. The exact many-body result is therefore twice the spinless one,

    n_j(t) = 2 n_j^{spinless}(t),      S(ℓ, t) = 2 S^{spinless}(ℓ, t),

which makes `freefermion_exact.py` a closed-form reference for the spinful
code, SU(2) included. Pass `--U` to turn the interaction on; there is then no
analytic reference, and the checks fall back to conservation laws and
agreement between the two symmetry groups.

What the non-Abelian comparison shows. Under SU(2) a bond index counts
*multiplets*, not states: a multiplet of spin `S` stands for `2S+1` states. A
faithful comparison therefore has to report both numbers, and when it does,
the same quench gives

    U1,U1    multiplets [4, 16, 60, 16, 4]    states [4, 16, 60, 16, 4]
    U1,SU2   multiplets [3, 10, 32, 10, 3]    states [4, 16, 60, 16, 4]

Identical state counts, because it is the same state; roughly half the
multiplets, because SU(2) stores each multiplet once instead of `2S+1` times.
That is the whole point of a non-Abelian implementation, and it only shows up
if the bookkeeping is right.

The same subtlety bites the entanglement entropy. Each singular value under
SU(2) describes a multiplet, so the Schmidt spectrum has to be weighted by
the degeneracies before taking `-Σ p log p`; `mps_entropy` does this, and
without it the SU(2) entropy comes out low by a factor approaching two while
still looking like a perfectly plausible number.

Only the `U1,*` groups are offered here. Under `Z2,*` the even-parity sector
is two-dimensional, merging |0⟩ with |↑↓⟩ into one sector, so a site
configuration cannot name them apart and this initial state cannot be built.
See `tdvp_hubbard_quench.py`, which starts from a DMRG ground state and
therefore covers all four groups.

Run from the repository root:

    uv run python examples/examples_tdvp/tdvp_hubbard.py
    uv run python examples/examples_tdvp/tdvp_hubbard.py --symmetry U1,SU2
    uv run python examples/examples_tdvp/tdvp_hubbard.py --compare-symmetries
    uv run python examples/examples_tdvp/tdvp_hubbard.py --U 4.0
    uv run python examples/examples_tdvp/tdvp_hubbard.py --L 8 --time 1.5
    uv run python examples/examples_tdvp/tdvp_hubbard.py --scheme 1s
    uv run python examples/examples_tdvp/tdvp_hubbard.py --help
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
from alice.physics.system import build_conductor
from freefermion_exact import (  # pyright: ignore[reportMissingImports]
    correlation_matrix,
    density_row,
    error_table,
    exact_entropy,
    mps_entropy,
    mps_occupations,
    occupation_mpos,
    onsite_mpos,
)

# Groups whose site sectors name a single physical state, which is what the
# domain-wall configuration needs. See the module docstring on Z2.
_VALID_SYMMETRIES = ('U1,U1', 'U1,SU2')


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _build_system(L: int, t: float, U: float, symmetry: str):
    """Build the Hubbard MPO and the on-site observables.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Nearest-neighbor hopping amplitude.
    U:
        On-site Coulomb repulsion.
    symmetry:
        Band symmetry string, one of `_VALID_SYMMETRIES`.

    Returns
    -------
    tuple
        `(mpo, n_obs, d_obs, phys, ops, spc)` where `n_obs[j]` measures the
        site occupation and `d_obs[j]` the double occupancy.
    """
    cfg = {
        'geometry': {'lattice': 'chain', 'lx': L, 'bcx': 'OBC', 'n2x': True},
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
    mpo = build_hamiltonian(interactions, geo.L, spc)
    phys, ops = build_conductor(symmetry)
    return (mpo,
            occupation_mpos(geo.L, spc, ops),
            onsite_mpos(geo.L, spc, ops, key='NN4', name='NN'),
            phys, ops, spc)


def _domain_wall_state(L: int, spc, ops, phys):
    """Build the doubly-occupied/empty domain wall, left half filled.

    The doubly-occupied sector is the last one of the physical space under
    both `U1,U1` and `U1,SU2`, and the empty sector is the first; both carry
    total spin zero, which is what makes the state SU(2)-representable.

    Parameters
    ----------
    L:
        Chain length.
    spc:
        Physical space from `build_interaction`.
    ops:
        Operator dictionary from `build_conductor`.
    phys:
        Physical Index from `build_conductor`, used to locate the sectors.

    Returns
    -------
    tuple
        `(mps, channel_occ)` where `channel_occ` is the initial occupation of
        a *single* spin channel, which is what the exact reference needs.
    """
    double, empty = len(phys.sectors) - 1, 0
    config = [double] * (L // 2) + [empty] * (L - L // 2)
    mps = init_mps(L, spc, ops, bond_dim=1, config=config,
                   dtype=torch.complex128)
    mps.canonical(0)
    mps.normalize()
    # Per spin channel the wall is the spinless one, so the reference is the
    # spinless result doubled.
    channel_occ = [1] * (L // 2) + [0] * (L - L // 2)
    return mps, channel_occ


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def tdvp_hubbard(
    L: int = 6,
    t: float = 1.0,
    U: float = 0.0,
    symmetry: str = 'U1,U1',
    time: float = 1.0,
    dt: float = 0.025,
    scheme: str = '2s',
    order: int = 2,
    bond_dim: int = 128,
    trunc_thresh: float = 1e-12,
    krylov_dim: int = 16,
    snapshots: int = 4,
    checkpoint_dir: Optional[str] = None,
    env_cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[tdvp.Summary, tdvp.Artifact]:
    """Melt a spinful domain wall and compare against the exact `U = 0` result.

    Parameters
    ----------
    L:
        Chain length. Kept small by default: the physical dimension is 4, so
        the bond dimension grows about twice as fast as in the spinless case.
    t:
        Nearest-neighbor hopping amplitude.
    U:
        On-site Coulomb repulsion. At `U = 0` an exact reference is available
        and is reported; otherwise only the conservation laws are checked.
    symmetry:
        Band symmetry, `'U1,U1'` (Abelian) or `'U1,SU2'` (non-Abelian). Both
        describe the same physics and must agree.
    time:
        Total evolution time.
    dt:
        Time step.
    scheme:
        `'2s'` (default) or `'1s'`. As always `'1s'` cannot grow a bond and so
        is frozen from this product state.
    order:
        Integrator order, `1` or `2`.
    bond_dim:
        Maximum bond dimension, counted in multiplets.
    trunc_thresh:
        SVD truncation threshold.
    krylov_dim:
        Maximum Krylov subspace dimension.
    snapshots:
        Number of intermediate times at which to report.
    checkpoint_dir:
        Directory for `tdvp.ckpt` and `evolution.ckpt`.
    env_cache_dir:
        Directory for spilling environment blocks to disk.
    verbose:
        Print the profile table and the error summaries.

    Returns
    -------
    Summary
        Trajectory history from the final evolution segment.
    Artifact
        Final state with its time and step index.
    """
    if symmetry not in _VALID_SYMMETRIES:
        raise ValueError(
            f"symmetry must be one of {_VALID_SYMMETRIES}, got {symmetry!r}; "
            "the Z2 groups merge the empty and doubly-occupied states into one "
            "two-dimensional sector, so this initial state cannot be named "
            "there -- see tdvp_hubbard_quench.py for those"
        )

    mpo, n_obs, d_obs, phys, ops, spc = _build_system(L, t, U, symmetry)
    mps, channel_occ = _domain_wall_state(L, spc, ops, phys)
    exact_available = (U == 0.0)

    if verbose:
        print(f"Chain length   : {L}")
        print(f"Hopping t      : {t}        U = {U}")
        print(f"Symmetry       : {symmetry}"
              f"   ({'non-Abelian' if 'SU2' in symmetry else 'Abelian'})")
        print(f"Physical space : {phys.dim} multiplets / "
              f"{phys.num_states} states per site")
        print("Initial state  : |" + ' '.join(
            '2' if j < L // 2 else '0' for j in range(L)) + ">  (total spin 0)")
        print(f"Scheme         : {scheme}   order = {order}")
        print(f"Time step      : {dt}   total time = {time}")
        print(f"Bond dim       : {bond_dim} multiplets")
        if exact_available:
            print("Reference      : exact, 2 x the spinless free-fermion chain "
                  "(the spin channels decouple at U = 0)")
        else:
            print(f"Reference      : none in closed form at U = {U}; "
                  "conservation laws only")
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
        krylov_dim=krylov_dim,
        checkpoint_dir=checkpoint_dir,
        env_cache_dir=env_cache_dir,
        save_artifacts=False,
    )

    if verbose:
        head = (f"{'t':>6}  {'mult':>5}  {'states':>6}  {'N-Nex':>9}  "
                f"{'E drift':>10}")
        if exact_available:
            head += f"  {'n_j err':>9}  {'S err':>9}"
        print(head + "  density profile")
        print('-' * (len(head) + L + 4))

    artifact = tdvp.Artifact(state=mps, time=0.0, step=0)
    summary = None
    step = 0
    e0 = observe(mps, mpo)
    while step < n_total:
        step = min(n_total, step + stride)
        summary, artifact = tdvp.run(artifact, mpo,
                                     dataclasses.replace(opts, n_steps=step))

        num = mps_occupations(artifact.state, n_obs)
        row = (f"{artifact.time:>6.3f}  "
               f"{max(artifact.state.bond_dims):>5d}  "
               f"{max(artifact.state.bond_states):>6d}  "
               f"{num.sum() - 2 * sum(channel_occ):>+9.1e}  "
               f"{observe(artifact.state, mpo) - e0:>+10.2e}")
        if exact_available:
            M = correlation_matrix(channel_occ, t, artifact.time)
            n_err = np.max(np.abs(num - 2 * np.real(np.diag(M))))
            s_err = max(
                abs(mps_entropy(artifact.state, ell) - 2 * exact_entropy(M, ell))
                for ell in range(1, L)
            )
            row += f"  {n_err:>9.2e}  {s_err:>9.2e}"
        if verbose:
            print(row + f"  |{density_row(num, vmax=2.0)}|")

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    if verbose:
        num = mps_occupations(artifact.state, n_obs)
        dbl = mps_occupations(artifact.state, d_obs)

        print()
        if exact_available:
            M = correlation_matrix(channel_occ, t, artifact.time)
            exact = 2 * np.real(np.diag(M))
            print(f"Occupations at t = {artifact.time:g}  "
                  f"(exact = 2 x spinless)")
            table, max_err = error_table('n_j (TDVP)', num, exact, range(L))
            print(table)
            print()
            print(f"Max |n_j - exact| : {max_err:.3e}")
            print()

            print(f"Entanglement entropy at t = {artifact.time:g}  "
                  f"(degeneracy-weighted; exact = 2 x spinless)")
            print(f"{'cut':>5}  {'S (TDVP)':>12}  {'S (exact)':>12}  "
                  f"{'error':>11}  {'mult':>5}  {'states':>7}")
            print('-' * 64)
            worst = 0.0
            for ell in range(1, L):
                s_num = mps_entropy(artifact.state, ell)
                s_ex = 2 * exact_entropy(M, ell)
                worst = max(worst, abs(s_num - s_ex))
                print(f"{ell:>5}  {s_num:>12.8f}  {s_ex:>12.8f}  "
                      f"{s_num - s_ex:>+11.2e}  "
                      f"{artifact.state.bond_dims[ell - 1]:>5d}  "
                      f"{artifact.state.bond_states[ell - 1]:>7d}")
            print()
            print(f"Max |S - exact| : {worst:.3e}")
            print()
        else:
            print(f"Occupations at t = {artifact.time:g}")
            print('  ' + '  '.join(f"{v:.6f}" for v in num))
            print()

        print(f"Double occupancy <n_up n_dn> : "
              f"{dbl.sum():.9f} total, "
              f"{', '.join(f'{v:.4f}' for v in dbl)}")
        print(f"Particle number : {num.sum():.10f}  "
              f"(exact {2 * sum(channel_occ)})")
        print(f"Energy          : {observe(artifact.state, mpo):+.10f}  "
              f"(initial {e0:+.10f}, conserved)")
        print(f"Norm            : {artifact.state.norm():.12f}")
        print(f"Bond multiplets : {artifact.state.bond_dims}")
        print(f"Bond states     : {artifact.state.bond_states}")
        if 'SU2' in symmetry:
            ratio = (sum(artifact.state.bond_states)
                     / max(1, sum(artifact.state.bond_dims)))
            print(f"  states per multiplet : {ratio:.3f}  "
                  f"(the SU(2) saving; 1.0 would mean no saving)")
        print(f"Max discarded weight : {max(summary.discarded_weights):.3e}")

    return summary, artifact


# ---------------------------------------------------------------------------
# Abelian versus non-Abelian comparison
# ---------------------------------------------------------------------------

def compare_symmetries(
    L: int = 6,
    t: float = 1.0,
    U: float = 0.0,
    time: float = 1.0,
    dt: float = 0.025,
    order: int = 2,
    bond_dim: int = 128,
) -> None:
    """Run the same quench under both symmetry groups and compare.

    The two runs share no tensor structure: one carries intertwiners and
    stores each multiplet once, the other enumerates every state. Agreement
    between them is therefore a strong statement that the non-Abelian
    bookkeeping is right, and it holds at `U != 0` where no closed-form
    reference exists.

    Parameters
    ----------
    L:
        Chain length.
    t:
        Hopping amplitude.
    U:
        On-site Coulomb repulsion.
    time:
        Total evolution time.
    dt:
        Time step.
    order:
        Integrator order.
    bond_dim:
        Maximum bond dimension in multiplets.
    """
    print(f"Abelian vs non-Abelian  (L = {L}, t = {t}, U = {U}, "
          f"t_final = {time}, dt = {dt})")
    print()

    out = {}
    for symmetry in _VALID_SYMMETRIES:
        summary, artifact = tdvp_hubbard(
            L=L, t=t, U=U, symmetry=symmetry, time=time, dt=dt,
            order=order, bond_dim=bond_dim, verbose=False,
        )
        mpo, n_obs, d_obs, _, _, _ = _build_system(L, t, U, symmetry)
        out[symmetry] = {
            'n': mps_occupations(artifact.state, n_obs),
            'd': mps_occupations(artifact.state, d_obs),
            'S': np.array([mps_entropy(artifact.state, ell)
                           for ell in range(1, L)]),
            'E': observe(artifact.state, mpo),
            'mult': artifact.state.bond_dims,
            'states': artifact.state.bond_states,
            'dw': max(summary.discarded_weights),
        }

    a, b = (out[s] for s in _VALID_SYMMETRIES)
    print(f"{'quantity':>22}  {_VALID_SYMMETRIES[0]:>16}  "
          f"{_VALID_SYMMETRIES[1]:>16}  {'difference':>12}")
    print('-' * 74)
    print(f"{'energy':>22}  {a['E']:>16.10f}  {b['E']:>16.10f}  "
          f"{a['E'] - b['E']:>+12.2e}")
    print(f"{'particle number':>22}  {a['n'].sum():>16.10f}  "
          f"{b['n'].sum():>16.10f}  {a['n'].sum() - b['n'].sum():>+12.2e}")
    print(f"{'double occupancy':>22}  {a['d'].sum():>16.10f}  "
          f"{b['d'].sum():>16.10f}  {a['d'].sum() - b['d'].sum():>+12.2e}")
    print(f"{'max |delta n_j|':>22}  {'':>16}  {'':>16}  "
          f"{np.max(np.abs(a['n'] - b['n'])):>12.2e}")
    print(f"{'max |delta S|':>22}  {'':>16}  {'':>16}  "
          f"{np.max(np.abs(a['S'] - b['S'])):>12.2e}")
    print()
    print(f"{'bond multiplets':>22}  {a['mult']!s:>16}")
    print(f"{'':>22}  {b['mult']!s:>16}")
    print(f"{'bond states':>22}  {a['states']!s:>16}")
    print(f"{'':>22}  {b['states']!s:>16}")
    print()
    print("The state counts agree because it is the same state. The multiplet")
    print("counts are what each group actually stores, and SU(2) needs fewer")
    print(f"of them: {sum(b['mult'])} against {sum(a['mult'])} summed over all "
          "bonds.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "TDVP domain-wall melting on the spinful Hubbard chain, run through\n"
            "the Abelian and non-Abelian code paths.\n\n"
            "Examples:\n"
            "  uv run python examples/examples_tdvp/tdvp_hubbard.py\n"
            "  uv run python examples/examples_tdvp/tdvp_hubbard.py --symmetry U1,SU2\n"
            "  uv run python examples/examples_tdvp/tdvp_hubbard.py --compare-symmetries\n"
            "  uv run python examples/examples_tdvp/tdvp_hubbard.py --U 4.0\n"
            "  uv run python examples/examples_tdvp/tdvp_hubbard.py --L 8 --time 1.5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--L', type=int, default=6, metavar='N',
                   help='chain length (default: 6; the physical dimension is 4, '
                        'so bonds grow fast)')
    p.add_argument('--t', type=float, default=1.0, metavar='T',
                   help='nearest-neighbor hopping amplitude (default: 1.0)')
    p.add_argument('--U', type=float, default=0.0, metavar='U',
                   help='on-site Coulomb repulsion; U=0 enables the exact '
                        'reference (default: 0.0)')
    p.add_argument('--symmetry', choices=list(_VALID_SYMMETRIES),
                   default='U1,U1',
                   help='band symmetry exploited (default: U1,U1)')
    p.add_argument('--time', type=float, default=1.0, metavar='T',
                   help='total evolution time (default: 1.0)')
    p.add_argument('--dt', type=float, default=0.025, metavar='DT',
                   help='time step (default: 0.025)')
    p.add_argument('--scheme', choices=['1s', '2s'], default='2s',
                   help='TDVP update scheme (default: 2s)')
    p.add_argument('--order', type=int, choices=[1, 2], default=2,
                   help='integrator order (default: 2)')
    p.add_argument('--bond-dim', type=int, default=128, metavar='D',
                   help='maximum bond dimension in multiplets (default: 128)')
    p.add_argument('--trunc-thresh', type=float, default=1e-12, metavar='THR',
                   help='SVD truncation threshold (default: 1e-12)')
    p.add_argument('--krylov-dim', type=int, default=16, metavar='K',
                   help='maximum Krylov subspace dimension (default: 16)')
    p.add_argument('--snapshots', type=int, default=4, metavar='K',
                   help='number of intermediate reports (default: 4)')
    p.add_argument('--checkpoint-dir', default=None, metavar='PATH',
                   help='directory for tdvp.ckpt and evolution.ckpt')
    p.add_argument('--env-cache-dir', default=None, metavar='PATH',
                   help='directory for spilling environment blocks to disk')
    p.add_argument('--compare-symmetries', action='store_true',
                   help='run the same quench under both symmetry groups and '
                        'diff the results instead of a single evolution')
    p.add_argument('--quiet', action='store_true',
                   help='suppress per-run output')
    p.add_argument('--quiet-log', action='store_true',
                   help='silence the per-step INFO log from the TDVP driver')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    alice.configure_logging()
    if args.quiet_log:
        logging.getLogger('alice').setLevel(logging.WARNING)

    if args.compare_symmetries:
        compare_symmetries(
            L=args.L,
            t=args.t,
            U=args.U,
            time=args.time,
            dt=args.dt,
            order=args.order,
            bond_dim=args.bond_dim,
        )
    else:
        tdvp_hubbard(
            L=args.L,
            t=args.t,
            U=args.U,
            symmetry=args.symmetry,
            time=args.time,
            dt=args.dt,
            scheme=args.scheme,
            order=args.order,
            bond_dim=args.bond_dim,
            trunc_thresh=args.trunc_thresh,
            krylov_dim=args.krylov_dim,
            snapshots=args.snapshots,
            checkpoint_dir=args.checkpoint_dir,
            env_cache_dir=args.env_cache_dir,
            verbose=not args.quiet,
        )
