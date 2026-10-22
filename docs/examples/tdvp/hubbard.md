# Hubbard Model

Spinful domain-wall melting with tunable symmetry, including the non-Abelian
SU(2) case — the example that validates TDVP on intertwiner-carrying tensors
against a closed-form reference.

\[
H = -t \sum_{j,\sigma} (c^\dagger_{j\sigma} c_{j+1\sigma} + \text{h.c.})
    + U \sum_j n_{j\uparrow} n_{j\downarrow}
\]

## Why U = 0 gives an exact reference for a spinful chain

At \(U = 0\) the two spin channels are completely decoupled, and each one is an
independent copy of the spinless free-fermion chain. A doubly-occupied domain
wall \(|2\,2\,2\,0\,0\,0\rangle\) is therefore one spinless wall per channel,
and every extensive observable is exactly twice the spinless value:

\[
n_j(t) = 2\,n_j^{\text{spinless}}(t), \qquad
S(\ell, t) = 2\,S^{\text{spinless}}(\ell, t)
\]

So the [domain-wall reference](domain-wall.md) carries over verbatim to a
four-dimensional local space with non-Abelian symmetry. That combination —
exact reference *and* intertwiners — is what makes this the decisive test of
the SU(2) code path.

The factor of two is not a fitted constant: it follows from the channels being
independent, and it would fail immediately if the SU(2) bookkeeping mixed them.

## Why the initial state must be all-singlet

The domain wall is built from doubly-occupied and empty sites only, never from
singly-occupied ones. A single electron carries spin 1/2, and a product state
containing a partially-filled SU(2) multiplet is not a total-spin eigenstate at
all, so it is not representable as a product state in an SU(2)-symmetric MPS.
Both \(|0\rangle\) and \(|\!\uparrow\downarrow\rangle\) are spin singlets, so
the wall as written is a legitimate \(S = 0\) state.

!!! warning "`init_mps` does not refuse an unrepresentable configuration"

    Asking for a configuration with a partially-filled SU(2) multiplet yields a
    silently mis-normalized state — a site that should read \(n = 1\) measures
    \(n = 0.707\) — accompanied only by an `auto-config could not return to
    Q_vac` warning. Keep SU(2) product states all-singlet.

## Symmetry groups

This example supports `'U1,U1'` and `'U1,SU2'`. The `Z2` variants are excluded
by the initial state rather than by TDVP: under `Z2` charge, the even-parity
sector merges \(|0\rangle\) with \(|\!\uparrow\downarrow\rangle\) into one
two-dimensional sector, so a site configuration cannot name them apart. The
[quench example](hubbard-quench.md) starts from a DMRG ground state instead and
covers all four.

## 1. Setup

```python
import alice
alice.configure_logging()

from alice.network import build_hamiltonian, build_interaction, init_mps
from alice.physics.system import build_conductor
from alice.algorithm import tdvp
import torch

config = {
    "geometry": {"lattice": "chain", "lx": 6, "bcx": "OBC", "n2x": True},
    "model": {"category": "conductor", "label": "Hubbard",
              "symmetry": "U1,SU2", "t": 1.0, "U": 0.0, "mu": 0.0},
}
interactions, spc, geo = build_interaction(config)
hamiltonian = build_hamiltonian(interactions, geo.L, spc)
phys, ops = build_conductor("U1,SU2")

# |2 2 2 0 0 0>: doubly occupied then empty, every site a singlet.
occ = [2] * (geo.L // 2) + [0] * (geo.L - geo.L // 2)
state = init_mps(geo.L, spc, ops, bond_dim=1, config=occ,
                 dtype=torch.complex128)
```

## 2. Evolve and compare

The reference is built from the *single-channel* occupation — one particle per
doubly-occupied site — and then doubled:

```python
from freefermion_exact import (correlation_matrix, exact_entropy,
                               exact_occupations, mps_entropy,
                               mps_occupations, occupation_mpos)

opts = tdvp.Options(scheme='2s', dt=0.05, n_steps=10, order=2, max_bond=128)
summary, artifact = tdvp.run(
    tdvp.Artifact(state=state, time=0.0, step=0), hamiltonian, opts)

channel_occ = [1 if o == 2 else 0 for o in occ]
n_exact = 2 * exact_occupations(channel_occ, 1.0, artifact.time)

M = correlation_matrix(channel_occ, 1.0, artifact.time)
s_exact = [2 * exact_entropy(M, ell) for ell in range(1, geo.L)]
```

## Multiplets versus states

Under a non-Abelian group an index counts *multiplets*, not states. The two
counts are both reported, since `max_bond` is expressed in multiplets while
memory and accuracy follow the state count:

| Property | Meaning |
|----------|---------|
| `MPS.bond_dims` | multiplets per bond — what `max_bond` limits |
| `MPS.bond_states` | degeneracy-resolved states per bond — the true dimension |

Their ratio is the SU(2) saving.

!!! note "Entanglement entropy needs degeneracy weighting"

    Each singular value under SU(2) describes an entire multiplet, standing for
    \(g = \mathrm{irrep\_dim}(q)\) degenerate Schmidt states of equal weight.
    With \(w_a\) the total weight of multiplet \(a\), the entropy is

    \[
    S = -\sum_a w_a \log\!\bigl(w_a / g_a\bigr)
    \]

    which reduces to \(-\sum p \log p\) when every \(g_a = 1\). Dropping the
    degeneracies does not shift the answer by a constant — it discards the
    \(\sum_a w_a \log g_a\) term entirely, which for this chain is a factor
    approaching two (0.72 reported against a true 1.06). `mps_entropy` in
    `freefermion_exact.py` handles this.

## Running the script

```bash
uv run python examples/examples_tdvp/tdvp_hubbard.py --symmetry U1,SU2 --L 6 --time 0.5
```

```text
Physical space : 3 multiplets / 4 states per site
Initial state  : |2 2 2 0 0 0>  (total spin 0)
Reference      : exact, 2 x the spinless free-fermion chain

     t   mult  states      N-Nex     E drift    n_j err      S err  density profile
----------------------------------------------------------------------------
 0.125     16      28   -4.4e-15   -7.17e-16   4.49e-06   2.33e-05  |@@@   |
 0.250     21      40   -4.7e-14   +2.50e-16   1.77e-05   6.39e-05  |@@%.  |
 0.375     28      52   +5.6e-14   +7.85e-15   3.73e-05   1.03e-04  |@@%.  |
 0.500     32      60   -6.2e-15   +9.41e-15   5.96e-05   1.29e-04  |@@#:  |

Max |n_j - exact| : 5.959e-05
Max |S - exact|   : 1.295e-04
Particle number : 6.0000000000  (exact 6)
Energy          : +0.0000000000  (initial +0.0000000000, conserved)
Bond multiplets : [3, 10, 32, 10, 3]
Bond states     : [4, 16, 60, 16, 4]
  states per multiplet : 1.724  (the SU(2) saving; 1.0 would mean no saving)
```

Occupations match twice the analytic spinless result to 6.0e-05 and the
degeneracy-weighted entropy to 1.3e-04, with particle number exact and energy
conserved to 9.4e-15. The SU(2) representation stores 32 multiplets where the
state count is 60.

## Bond expansion under a non-Abelian group

`--scheme 1sp` is worth more here than on the spinless chain. The local space
is four-dimensional, so a 2-site tensor is correspondingly larger and keeping
the local exponential 1-site sized saves more. The complement is assembled in
whatever basis the state is carried in, which means that under SU(2) the
directions it adds are *multiplets* and `--expand-k` is a count of multiplets
just as `--bond-dim` is:

```bash
uv run python examples/examples_tdvp/tdvp_hubbard.py \
    --symmetry U1,SU2 --L 6 --time 0.5 --scheme 1sp
```

| `scheme` | max \|n_j - exact\| | max \|S - exact\| | multiplets | states |
|----------|-------------------|-----------------|------------|--------|
| `'1sp'`  | 8.464e-05         | 1.197e-04       | 28         | 52     |
| `'2s'`   | 5.959e-05         | 1.295e-04       | 32         | 60     |

Comparable accuracy on both quantities from a state carrying 28 multiplets
rather than 32. Note that the two savings compose: SU(2) stores 52 states as 28
multiplets, and the expansion is what kept the count at 28 instead of 32 in the
first place.

The default `expand_k=4` therefore admits more states per bond here than it
would under `U1,U1`, since each multiplet it adds stands for \(2S+1\) states.
Raise it when the bond profile looks capped rather than converged.

## Cross-symmetry agreement at finite U

Switching on \(U\) removes the analytic reference but enables a different
check — the Abelian and non-Abelian paths must still agree:

```bash
uv run python examples/examples_tdvp/tdvp_hubbard.py --compare-symmetries --U 4.0
```

```text
Abelian vs non-Abelian  (L = 6, t = 1.0, U = 4.0, t_final = 1.0, dt = 0.025)

              quantity             U1,U1            U1,SU2    difference
--------------------------------------------------------------------------
                energy      0.0000000000      0.0000000000     -1.14e-13
       particle number      6.0000000000      6.0000000000     +1.00e-13
      double occupancy      2.7778232525      2.7778232525     +2.53e-14
       max |delta n_j|                                          3.15e-14
         max |delta S|                                          4.90e-14

       bond multiplets  [4, 16, 64, 16, 4]
                        [3, 10, 35, 10, 3]
           bond states  [4, 16, 64, 16, 4]
                        [4, 16, 64, 16, 4]
```

Every observable agrees to machine precision between two tensor structures that
share nothing — one carries intertwiners, the other does not. The state counts
match because it is the same state; the multiplet counts are what each group
actually stores, and SU(2) needs 61 against 104 summed over all bonds.

`--compare-symmetries` honors `--scheme`, and the comparison is noticeably
looser under `'1sp'` — the same run agrees to 1.6e-08 rather than 3.2e-14:

```text
Abelian vs non-Abelian  (L = 6, t = 1.0, U = 4.0, t_final = 1.0, dt = 0.025, scheme = 1sp)

              quantity             U1,U1            U1,SU2    difference
--------------------------------------------------------------------------
                energy     -0.0000000000      0.0000000000     -1.04e-13
       particle number      6.0000000000      6.0000000000     -3.11e-14
      double occupancy      2.7778095954      2.7778095820     +1.34e-08
       max |delta n_j|                                          1.63e-08
         max |delta S|                                          1.04e-08
```

This is expected rather than a bug, and it is specific to the expansion. Which
complement directions a bond receives depends on the basis the state is carried
in, and the two groups do not make identical choices, so the truncation that
follows differs slightly and the trajectories separate at that level. The
quantities protected by symmetry rather than by the expansion — energy and
particle number — still agree to 1e-13. Use `'2s'` when cross-group agreement
is itself the measurement.

## See Also

- [Hubbard quench](hubbard-quench.md) — interaction quench covering all four symmetry groups.
- [Domain wall](domain-wall.md) — the spinless reference this example doubles.
- [DMRG Hubbard](../dmrg/hubbard.md) — ground state of the same Hamiltonian.
- [build_conductor](../../api/local-space/build-conductor.md) — the spinful local space.
