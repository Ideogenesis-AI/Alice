# Comparing the Update Schemes

`'1s'`, `'1sp'` and `'2s'` run side by side on one quench and scored against an
exact solution — the page to read when choosing between them.

What separates the three is where their bond dimension comes from, since an MPS
represents only what its bonds allow:

| Scheme | Local update | Bond dimension decided |
|--------|--------------|------------------------|
| `'1s'` | 1 site | Never. The update is confined to a manifold of fixed bond dimension |
| `'1sp'` | 1 site | Before the update, by widening the bond ahead of it from the complement of the space it currently spans |
| `'2s'` | 2 sites | After the update, as a by-product of the 2-site SVD and its truncation |

Everything below follows from that single column: which schemes can start from
a product state at all, which one needs parameters tuned, and which one ends up
carrying the least bond dimension for a given accuracy.

## The test case

\[
H = -t \sum_j \left(c^\dagger_j c_{j+1} + \text{h.c.}\right), \qquad
|\psi(0)\rangle = |\underbrace{1 \cdots 1}_{L/2}\underbrace{0 \cdots 0}_{L/2}\rangle
\]

A free-fermion chain released from a half-filled product state, chosen because
it puts both hard parts in one run: every bond starts at dimension 1, which a
fixed-rank scheme can never leave, and the front then keeps demanding more room
as it advances. The [domain wall](domain-wall.md) page studies this state for
its own sake — entropy growth, truncation error, the exact Gaussian solution in
full. Here it is the instrument rather than the object.

Why not the simpler starting point? The single-particle quench of the
[free fermion example](free-fermion.md) shows the frozen `'1s'` start just as
well, but it cannot test the expansion at all: a one-particle state has Schmidt
rank exactly 2 at every bond for all time, so the bond opens once and never
needs anything again. Any `expand_k ≥ 1` would look identical.

Correctness is checked against the exact solution. A product state of free
fermions stays Gaussian, so the equal-time correlation matrix evolves in the
single-particle space,

\[
C_{jk}(t) = \langle c^\dagger_j c_k \rangle, \qquad
C(t) = U(t)\, C(0)\, U(t)^\dagger, \qquad U = e^{-i h t}
\]

with \(h\) the \(L \times L\) hopping matrix and \(C(0)\) diagonal in the
initial occupations. The site occupations are the diagonal of \(C(t)\) and the
half-chain entanglement entropy follows from its eigenvalues on one half, so
both columns of the output have an exact counterpart.

## Setting up

The product state comes from `init_mps` with an explicit occupation `config`
and `bond_dim=1`:

```python
import torch
from alice import build_hamiltonian, build_interaction, init_mps, tdvp
from alice.physics.system import build_fermionic

L, t = 12, 1.0

cfg = {
    "geometry": {"lattice": "chain", "lx": L, "bcx": "OBC", "n2x": True},
    "model": {"category": "fermionic", "label": "FreeFermion",
              "symmetry": "U1", "t": t},
}
interactions, spc, geo = build_interaction(cfg)
mpo = build_hamiltonian(interactions, geo.L, spc)

_, ops = build_fermionic("U1")
config = [1] * (L // 2) + [0] * (L // 2)
state = init_mps(L, spc, ops, bond_dim=1, config=config,
                 dtype=torch.complex128)
```

Then the only thing that distinguishes the three runs is `scheme`:

```python
opts = tdvp.Options(scheme='1sp', dt=0.05, n_steps=40, order=2,
                    max_bond=64, expand_k=4)
summary, artifact = tdvp.run(
    tdvp.Artifact(state=state, time=0.0, step=0), mpo, opts)
```

`expand_k` is ignored by `'1s'` and `'2s'`, so the same `Options` can be reused
across schemes with only the name changed.

## Running the script

```bash
uv run python examples/examples_tdvp/tdvp_scheme_comparison.py --quiet-log
```

```text
Chain length  : 12
Hopping t     : 1.0
Initial state : |111111000000⟩  (product state, every bond dimension 1)
Time step     : 0.05   total time = 2.0   order = 2
expand_k      : 4   expand_alpha: None

scheme = '1s'
       t    n_j err        S  S exact           E      norm        N  bonds  density
  -----------------------------------------------------------------------------------
   0.400   1.42e-01   0.0000   0.4193   -1.66e-15  1.000000   6.0000      1  |@@@@@@      |
   0.800   3.96e-01   0.0000   0.7047   -1.66e-15  1.000000   6.0000      1  |@@@@@@      |
   1.200   5.00e-01   0.0000   0.6486   -1.66e-15  1.000000   6.0000      1  |@@@@@@      |
   1.600   4.49e-01   0.0000   0.6043   -1.66e-15  1.000000   6.0000      1  |@@@@@@      |
   2.000   4.21e-01   0.0000   0.7115   -1.66e-15  1.000000   6.0000      1  |@@@@@@      |

scheme = '1sp'
       t    n_j err        S  S exact           E      norm        N  bonds  density
  -----------------------------------------------------------------------------------
   0.400   1.19e-04   0.4194   0.4193   -1.79e-15  1.000000   6.0000     10  |@@@@@%.     |
   0.800   2.74e-04   0.7046   0.7047   -3.12e-15  1.000000   6.0000     16  |@@@@%+=.    |
   1.200   1.76e-04   0.6481   0.6486   -4.33e-15  1.000000   6.0000     24  |@@@@#=+:    |
   1.600   1.55e-04   0.6037   0.6043   -4.88e-15  1.000000   6.0000     29  |@@@%*+=-.   |
   2.000   9.05e-05   0.7111   0.7115   -1.15e-14  1.000000   6.0000     38  |@@%*++==-.  |

scheme = '2s'
       t    n_j err        S  S exact           E      norm        N  bonds  density
  -----------------------------------------------------------------------------------
   0.400   8.18e-05   0.4193   0.4193   -2.59e-15  1.000000   6.0000     14  |@@@@@%.     |
   0.800   2.00e-04   0.7047   0.7047   -9.66e-16  1.000000   6.0000     22  |@@@@%+=.    |
   1.200   1.69e-04   0.6483   0.6486   +2.61e-15  1.000000   6.0000     29  |@@@@#==:    |
   1.600   9.78e-05   0.6040   0.6043   +6.06e-15  1.000000   6.0000     37  |@@@%*+=-.   |
   2.000   9.14e-05   0.7113   0.7115   +1.93e-15  1.000000   6.0000     46  |@@%*++==-.  |

Summary
   scheme    max |n_j - exact|  bond profile
  ----------------------------------------------------------------------
       1s            4.211e-01  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
      1sp            9.049e-05  [2, 4, 8, 16, 25, 38, 25, 16, 8, 4, 2]
       2s            9.139e-05  [2, 4, 8, 16, 28, 46, 28, 16, 8, 4, 2]
```

Read the `bonds`, `S` and `density` columns together. Under `'1s'` the bond
profile never leaves 1, the entropy stays at zero where the exact value climbs
to 0.71, and the density pattern is still the initial wall — all while the
energy sits at 1.7e-15, the norm at 1.000000 and the particle number at exactly
6. That is a frozen run with every conservation law intact, which is why the
conserved quantities are the wrong place to look for trouble.

Under `'1sp'` and `'2s'` the wall melts, the entropy tracks the exact curve to
four digits, and the occupations land within 9e-05 of the Gaussian solution.
`'1sp'` gets there carrying a visibly smaller bond profile than `'2s'` — 38
against 46 at the center — because it only ever adds directions the complement
actually offers, where the 2-site SVD keeps whatever survives its truncation
threshold.

!!! note "`'1s'` warns about this"

    The driver detects the frozen situation and says so once per run:

    ```text
    scheme '1s' cannot grow bond dimensions, and bond(s) 0, 1, 2, 3, 4, 5, 6, 7, … (11 total)
    have dimension 1; no correlation can cross them, so the dynamics there will be frozen
    while norm and energy stay conserved. Use scheme '2s' or '1sp', or warm up with one of
    them until every bond has opened.
    ```

    The warning is what makes the `'1s'` row above an expected outcome rather
    than a silent failure.

## How much expansion is enough

```bash
uv run python examples/examples_tdvp/tdvp_scheme_comparison.py --quiet-log --expand-k-scan
```

```text
expand_k scan  (L = 12, t_final = 2.0, dt = 0.05)
Reference: exact correlation matrix C(t) = U C(0) U†

 expand_k    max |n_j - exact|   max bond  bond profile
------------------------------------------------------------------------------
        0            4.211e-01          1  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        1            9.048e-05         38  [2, 4, 8, 16, 25, 38, 25, 16, 8, 4, 2]
        2            9.049e-05         38  [2, 4, 8, 16, 25, 38, 25, 16, 8, 4, 2]
        4            9.049e-05         38  [2, 4, 8, 16, 25, 38, 25, 16, 8, 4, 2]
        8            9.049e-05         38  [2, 4, 8, 16, 25, 38, 25, 16, 8, 4, 2]
```

`expand_k = 0` is in the scan deliberately. It disables the expansion and so
must reproduce `'1s'` exactly — 4.211e-01, the same number as the summary table
above — which separates an error in the expansion from an error in the
integrator around it.

Above that the table stops moving, which is worth understanding rather than
dismissing. `expand_k` is a **cap, not a request**: a bond receives however
many directions its complement actually holds above the noise floor, and for a
short-range Hamiltonian that number is small — never more than two anywhere in
this run. Raising the cap past it costs an SVD of a slightly larger block and
buys nothing. This is also why the bonds still climb to 38 even though at most
two directions enter per update: the cap limits one bond in one half-sweep, not
the trajectory.

## The expansion adds no error floor

The expansion is a per-step decision, so it is worth confirming that what is
left once the bonds are open is the integrator's own error and nothing else:

```bash
uv run python examples/examples_tdvp/tdvp_scheme_comparison.py --quiet-log --dt-scan
```

```text
Time-step convergence of '1sp'  (L = 12, t_final = 1.0, expand_k = 4)
Reference: exact correlation matrix C(t) = U C(0) U†

  steps         dt    max |n_j - exact|    ratio
--------------------------------------------------
     10    0.10000            1.047e-03  
     20    0.05000            2.597e-04     4.03
     40    0.02500            6.462e-05     4.02
     80    0.01250            1.612e-05     4.01
```

Four per halving of the step, which is the second-order integrator. Had the
expansion been leaving behind an error of its own the ratios would have
flattened out as `dt` shrank, since choosing a subspace is not an operation
`dt` controls.

## See Also

- [Domain wall](domain-wall.md) — the same initial state studied for its own physics.
- [Free fermion](free-fermion.md) — the single-particle quench and the same analytic machinery.
- [Hubbard quench](hubbard-quench.md) — a quench from a DMRG ground state, where `'1s'` is usable.
- [Controlled bond expansion](../../algorithms/tdvp/index.md#controlled-bond-expansion) — how the widening works.
- [tdvp.Options](../../algorithms/tdvp/options.md) — `expand_k` and `expand_alpha`.
