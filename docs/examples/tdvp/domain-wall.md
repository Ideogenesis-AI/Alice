# Domain Wall Melting

A half-filled domain wall released on a free-fermion chain — the many-body
counterpart of the [single-particle benchmark](free-fermion.md), with
entanglement that actually grows and bonds that actually need truncating.

\[
H = -t \sum_{\langle i,j \rangle} (c^\dagger_i c_j + \text{h.c.}), \qquad
|\psi(0)\rangle = |\underbrace{1\cdots1}_{L/2}\underbrace{0\cdots0}_{L/2}\rangle
\]

## Why this state has an exact reference

The initial state is a product state, and the Hamiltonian is quadratic, so the
state stays Gaussian for all time and every observable follows from the
single-particle correlation matrix:

\[
M(t) = U(t)\,\mathrm{diag}(\mathrm{occ})\,U(t)^\dagger, \qquad U(t) = e^{-iht}
\]

Occupations are its diagonal, \(n_j(t) = M_{jj}(t)\). The entanglement entropy
across a cut at \(\ell\) comes from the eigenvalues \(\nu_a\) of the restricted
block \(M[{:}\ell, {:}\ell]\):

\[
S(\ell) = -\sum_a \bigl[\nu_a \log \nu_a + (1-\nu_a)\log(1-\nu_a)\bigr]
\]

This is the part that makes the example worth running: the entropy is what
truncation damages first, and it is checked here against a closed form rather
than against a larger-bond-dimension run.

Unlike the single-particle case, this is a genuine many-body state whose bond
dimension grows from 1 to several tens over the run, so both error
sources — truncation and time step — are live.

## 1. Setup and initial state

```python
import alice
alice.configure_logging()

from alice.network import build_hamiltonian, build_interaction, init_mps
from alice.algorithm import tdvp
import torch

config = {
    "geometry": {"lattice": "chain", "lx": 16, "bcx": "OBC", "n2x": True},
    "model": {"category": "fermionic", "label": "FreeFermion",
              "symmetry": "U1", "t": 1.0, "mu": 0.0},
}
interactions, spc, geo = build_interaction(config)
hamiltonian = build_hamiltonian(interactions, geo.L, spc)

config_occ = [1] * (geo.L // 2) + [0] * (geo.L - geo.L // 2)
state = init_mps(geo.L, spc, ops, bond_dim=1, config=config_occ,
                 dtype=torch.complex128)
```

## 2. Evolve

```python
opts = tdvp.Options(scheme='2s', dt=0.05, n_steps=40, order=2,
                    max_bond=128, trunc_thresh=1e-12)
summary, artifact = tdvp.run(
    tdvp.Artifact(state=state, time=0.0, step=0), hamiltonian, opts)
```

!!! warning "`scheme='1s'` does nothing here"

    Every bond of a product state has dimension 1, and the 1-site scheme
    cannot change a bond dimension. Started from this state it returns the
    initial state unchanged at every step, conserving norm and energy to 1e-15
    the whole way while the wall never melts. `run()` warns about the
    dimension-1 bonds at startup, but nothing in the trajectory itself looks
    wrong. The script accepts `--scheme 1s` precisely so the trap can be
    observed: the occupation error climbs to 0.42 while \(\chi\) stays at 1.

!!! tip "`scheme='1sp'` is the way out of it"

    The 1-site-plus scheme widens each bond before updating it, drawing up to
    `expand_k` directions from the complement of the space that bond currently
    spans. The rank adapts as it does under `'2s'`, but every local
    exponential stays 1-site sized. It melts this wall from the same product
    state:

    | `scheme` | max \|n_j - exact\| | max \|S - exact\| | \(\chi\) |
    |----------|-------------------|-----------------|----------|
    | `'1s'`   | 4.211e-01         | 7.115e-01       | 1        |
    | `'1sp'`  | 9.048e-05         | 4.719e-04       | 40       |
    | `'2s'`   | 9.138e-05         | 2.478e-04       | 53       |

    All at \(L = 16\), \(t_{\text{final}} = 2\), `dt=0.05`, and `expand_k=4`.
    `'1sp'` matches `'2s'` on the occupations while carrying \(\chi = 40\)
    against 53 — but its entropy error is about twice as large, which is the
    honest trade: the entropy is the quantity most sensitive to the Schmidt
    directions a capped expansion never added.

    `expand_k` is a third error source on top of truncation and `dt`, and the
    one to suspect when the error stops responding to either. `--bond-scan`
    accepts `--scheme 1sp` so this can be checked directly.

## 3. Compare against the Gaussian reference

```python
from freefermion_exact import (correlation_matrix, exact_entropy,
                               exact_occupations, mps_entropy,
                               mps_occupations, occupation_mpos)

obs = occupation_mpos(geo.L, spc, ops)
n_tdvp = mps_occupations(artifact.state, obs)
n_exact = exact_occupations(config_occ, 1.0, artifact.time)

# Both the occupations and the entropies come from the same correlation
# matrix: the diagonal for the former, the spectrum of a corner block for
# the latter.
M = correlation_matrix(config_occ, 1.0, artifact.time)
s_tdvp = [mps_entropy(artifact.state, ell) for ell in range(1, geo.L)]
s_exact = [exact_entropy(M, ell) for ell in range(1, geo.L)]
```

## Running the script

```bash
uv run python examples/examples_tdvp/tdvp_domainwall.py --L 16 --time 2.0 --dt 0.05
```

```text
Initial state  : |1111111100000000>
Light cone     : x0 +/- 2t  =>  front reaches +/-4 sites by t = 2

     t   chi    n_j err      S err         dw      N-Nex           E  density profile
----------------------------------------------------------------------------------------
  0.50    15   1.17e-04   2.60e-04   5.63e-25   +3.9e-14   +2.17e-14  |@@@@@@@#:       |
  1.00    27   2.06e-04   2.16e-04   1.79e-24   +9.2e-14   +2.45e-14  |@@@@@@%+=.      |
  1.50    39   8.03e-05   3.11e-04   9.55e-25   +4.6e-14   +2.22e-14  |@@@@@%*+=-.     |
  2.00    53   9.14e-05   2.48e-04   2.68e-24   +6.9e-14   +1.91e-14  |@@@@%*++==-.    |

Max |n_j - exact| over all sites : 9.138e-05
Max |S - exact| over all cuts    : 2.478e-04
Particle number : 8.000000000000  (exact 8)
Energy          : +1.907e-14  (exact 0, since the wall has no hopping energy)
Bond dims       : [2, 4, 8, 13, 21, 28, 40, 53, 40, 28, 21, 13, 8, 4, 2]
```

Occupations match to 1e-4 and the entropy profile to 2e-4 across all fifteen
cuts, while particle number and energy hold to 1e-14. The density profile shows
the wall spreading within its light cone: the front advances at the maximum
group velocity \(v_{\max} = 2t\), reaching four sites by \(t = 2\), and the
region beyond stays empty.

The energy is exactly zero for a reason worth noting — the initial state has no
bond with a particle on one side and a hole on the other in a coherent
superposition, so it carries no hopping energy, and the evolution must preserve
that.

## Separating the two error sources

```bash
uv run python examples/examples_tdvp/tdvp_domainwall.py --bond-scan --L 16 --time 2.0
```

The scan raises `max_bond` and then, at the largest value, also halves `dt`:

```text
 max_bond   chi    max |n_j - exact|    max |S - exact|      max dw
----------------------------------------------------------------------
        4     4            4.741e-03          1.483e-02    8.31e-06
        8     8            1.022e-04          2.340e-04    1.24e-08
       16    16            9.138e-05          2.478e-04    5.71e-12
       32    32            9.138e-05          2.478e-04    1.11e-18
       64    53            9.138e-05          2.478e-04    2.68e-24

Same run at the largest bond dimension with dt = 0.025:
       64    51            2.272e-05          6.172e-05    3.34e-24
```

Truncation dominates up to \(\chi = 8\); from \(\chi = 16\) on, both errors sit
at *identical* values while the discarded weight keeps falling by six orders of
magnitude per row — conclusive evidence that what remains is not truncation.
Halving the time step at the largest bond dimension drops both by a factor of
4.02, identifying the plateau as the `O(dt²)` integration error.

Note that \(\chi\) stops tracking `max_bond` at 64: the exact rank needed is
53, so the cap is no longer binding.

This is the practical lesson for any TDVP run: a plateau under increasing
`max_bond` means the bond dimension is no longer the limit, and further
accuracy has to come from `dt`.

Under `--scheme 1sp` the same scan carries one more candidate, since `expand_k`
caps how fast the bond profile can follow the entanglement. A plateau that
*also* survives halving `dt` is pointing at the expansion rather than the
integrator, and `expand_k` is what to raise. `--bond-scan` rejects `--scheme
1s` outright: with every bond pinned at 1, each row of the table would be
identical.

## See Also

- [Free fermion](free-fermion.md) — the single-particle benchmark with no truncation.
- [Hubbard model](hubbard.md) — the same melting problem with spin and SU(2) symmetry.
- [tdvp.Summary](../../algorithms/tdvp/summary.md) — reading `discarded_weights`.
