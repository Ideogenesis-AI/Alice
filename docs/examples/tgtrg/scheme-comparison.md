# Comparing the Update Schemes

`'1s'`, `'1sp'` and `'2s'` cooled side by side down one schedule and scored
against an exact `log Z` — the page to read when choosing between them.

What separates the three is where the bond dimension of \(\rho\) comes from,
and a cooling run makes that question unavoidable: \(\rho(\tau_0)\) arrives at
the Hamiltonian's own rank, far below what a cold density matrix needs.

| Scheme | Local propagator | Bond dimension decided |
|--------|------------------|------------------------|
| `'1s'` | 1 site | Never. The update is confined to a manifold of fixed rank |
| `'1sp'` | 1 site | Before the update, by expanding the bond from the complement of the space it currently spans |
| `'2s'` | 2 sites | After the update, as a by-product of the 2-site SVD and its truncation |

Everything below follows from that single column.

## The test case

\[
H = -t \sum_j \left(c^\dagger_j c_{j+1} + \text{h.c.}\right)
\]

The spinless free-fermion chain, cooled from the Taylor state
\(\rho(\tau_0) = e^{-\tau_0 H}\) down the stacked schedule — exponential out of
the high-temperature regime, then linear. The [free fermion](free-fermion.md)
page studies this chain for its own sake, with the full thermodynamics table
against internal energy, specific heat and entropy. Here it is the instrument,
not the object.

Correctness is checked against the exact grand-canonical partition function,
which for an open chain is a one-line sum over single-particle levels:

\[
\varepsilon_k = -2t \cos\frac{k\pi}{L+1}, \qquad
\log Z(\beta) = \sum_{k=1}^{L} \log\left(1 + e^{-\beta \varepsilon_k}\right)
\]

TgTRG reads `log Z` off the cooled density matrix at every step, so every
scheme is measured against the answer at every \(\beta\) and not against
another tensor network.

## Setting up

The Taylor state and the Hamiltonian are built once and shared, since the
schemes differ only in `Options`:

```python
from alice.algorithm import tgtrg
from alice.network import build_hamiltonian, build_interaction
from alice.network.thermal import thermal_mpo

L, t, tau0 = 10, 1.0, 2 ** -8

cfg = {
    "geometry": {"lattice": "chain", "lx": L, "bcx": "OBC", "n2x": True},
    "model": {"category": "fermionic", "label": "FreeFermion",
              "symmetry": "U1", "t": t},
}
interactions, spc, geo = build_interaction(cfg)
H = build_hamiltonian(interactions, geo.L, spc)

rho0 = thermal_mpo(H, tau0, 10, spc)
state = tgtrg.Artifact(rho=rho0, beta=tau0, step=0)
```

Then the only thing that distinguishes the three runs is `scheme`:

```python
opts = tgtrg.Options(scheme='1sp', grid='exp', tau_0=tau0, n_steps=9,
                     order=2, max_bond=32, expand_k=4, expand_alpha=4)
summary, artifact = tgtrg.run(state, H, opts)
```

`expand_k` and `expand_alpha` are ignored by `'1s'` and `'2s'`, so the same
`Options` can be reused across schemes with only the name changed. The linear
phase then resumes from the `Artifact` the exponential phase returned, which
is all a stacked schedule takes.

## Running the script

```bash
uv run python examples/examples_tgtrg/tgtrg_scheme_comparison.py
```

```text
Chain length  : 10
Hopping t     : 1.0
Schedule      : 9 doubling steps to β = 2, then 8 steps of τ = 0.5 to β = 6
Integrator    : order 2
Max bond dim  : 32
expand_k      : 4   expand_alpha: 4

scheme = '1s'
      beta         T           log Z   log Z err           f     D         dw
  ------------------------------------------------------------------------
    0.0039  256.0000      6.93150614    1.78e-15  -177.446557    10   0.00e+00
    0.0625   16.0000      6.94025690    1.64e-11  -11.104411    10   0.00e+00
    0.2500    4.0000      7.07109262    4.56e-08   -2.828437    10   0.00e+00
    1.0000    1.0000      8.96442165    1.49e-04   -0.896442    10   0.00e+00
    2.0000    0.5000     13.54487867    4.26e-03   -0.677244    10   0.00e+00
    4.0000    0.2500     24.71257482    3.45e-02   -0.617814    10   0.00e+00
    6.0000    0.1667     36.42303029    8.47e-02   -0.607051    10   0.00e+00

scheme = '1sp'
      beta         T           log Z   log Z err           f     D         dw
  ------------------------------------------------------------------------
    0.0039  256.0000      6.93150614    1.78e-15  -177.446557    10   0.00e+00
    0.0625   16.0000      6.94025690    7.82e-14  -11.104411    14   0.00e+00
    0.2500    4.0000      7.07109257    1.40e-11   -2.828437    26   0.00e+00
    1.0000    1.0000      8.96457424    3.47e-06   -0.896457    32   1.15e-11
    2.0000    0.5000     13.54919630    6.19e-05   -0.677460    32   3.06e-09
    4.0000    0.2500     24.74663019    4.71e-04   -0.618666    32   8.61e-08
    6.0000    0.1667     36.50639470    1.34e-03   -0.608440    32   1.78e-07

scheme = '2s'
      beta         T           log Z   log Z err           f     D         dw
  ------------------------------------------------------------------------
    0.0039  256.0000      6.93150614    1.78e-15  -177.446557    10   0.00e+00
    0.0625   16.0000      6.94025690    1.51e-14  -11.104411    21   6.15e-25
    0.2500    4.0000      7.07109257    8.89e-12   -2.828437    32   4.41e-19
    1.0000    1.0000      8.96457082    3.97e-08   -0.896457    32   3.12e-11
    2.0000    0.5000     13.54912811    6.25e-06   -0.677456    32   5.06e-08
    4.0000    0.2500     24.74687305    2.29e-04   -0.618672    32   1.62e-06
    6.0000    0.1667     36.50704834    6.83e-04   -0.608451    32   4.64e-06

Summary
   scheme   log Z err at β_max   largest D   largest dw
  ------------------------------------------------------
       1s            8.470e-02          10     0.00e+00
      1sp            1.336e-03          32     1.78e-07
       2s            6.825e-04          32     4.64e-06
```

(Rows thinned for the page; the script prints every step.)

Read the `D` and `log Z err` columns together. All three agree to roundoff
while \(\beta\) is small, where \(\rho\) really does have rank 10 and the
manifold is not binding. Past \(\beta \approx 1\) they separate: `'1s'` holds
at 10 and its error climbs two orders of magnitude on the way to
\(\beta = 6\), while `'1sp'` and `'2s'` climb to the cap and stay within
1e-03.

Nothing is frozen here the way a product-state quench freezes under
[TDVP's `'1s'`](../tdvp/scheme-comparison.md) — the Taylor state has rank 10,
not rank 1, so the run moves and simply moves on the wrong manifold. The
damage is visible from the first cold step because `log Z` has an exact
counterpart at every \(\beta\); in a problem without one it would not be, and
the only signal would be that `'1s'` reports a discarded weight of exactly
zero at every step. That zero is honest and useless: a fixed-rank scheme
discards nothing because it never looks at anything to discard.

!!! warning "Discarded weight is not comparable across schemes"

    `'1sp'` reports a *smaller* largest discarded weight than `'2s'` —
    1.78e-07 against 4.64e-06 — and a *larger* error in `log Z`. The two
    numbers measure different things. `'2s'` truncates the full 2-site
    space, so its discarded weight covers everything the update could have
    kept; `'1sp'` truncates a space that was only expanded by `expand_k`
    directions, so it reports what it dropped from the subspace it chose to
    look at, not what lay outside it. Compare discarded weights within a
    scheme, across \(\beta\) or across `max_bond`, never between `'1sp'` and
    `'2s'`.

## How much expansion is enough

```bash
uv run python examples/examples_tgtrg/tgtrg_scheme_comparison.py --expand-k-scan
```

```text
expand_k scan  (L = 10, max_bond = 32)
  expand_k   log Z err at β_max   largest D
  ------------------------------------------
         0            8.470e-02          10
         1            2.241e-02          32
         2            1.092e-03          32
         4            1.336e-03          32
         8            1.280e-03          32
```

`expand_k = 0` is in the scan deliberately. It empties the complement and so
must reproduce `'1s'` exactly — 8.470e-02, the same number as the summary
table above — which separates an error in the expansion from an error in the
integrator around it.

The `k = 1` row is the one worth reading closely. Its bond *does* reach the
cap, yet the error is twenty times what `k = 2` achieves, because a single
direction per bond per half-sweep gets there far too late: the rank arrives
after the cooling steps that needed it. Bond dimension at the end of a run is
not evidence that the run had it when it mattered. Past `k = 2` the table
flattens, the remaining 1e-03 being the truncation floor that `max_bond = 32`
sets, and raising `expand_k` there costs an SVD of a larger block and buys
nothing.

A reasonable starting point is \(k \approx \sqrt{D}\), checked with one scan
like this one on a short run before committing to a long schedule.

## See Also

- [Free fermion](free-fermion.md) — the same chain studied for its own thermodynamics.
- [Hubbard model](hubbard.md) — a cooling run where `max_bond` is the binding constraint.
- [TgTRG overview](../../algorithms/tgtrg/index.md) — the schemes and the stacked schedule.
- [tgtrg.Options](../../algorithms/tgtrg/options.md) — `expand_k` and `expand_alpha`.
- [TDVP scheme comparison](../tdvp/scheme-comparison.md) — the same three schemes in real time.
