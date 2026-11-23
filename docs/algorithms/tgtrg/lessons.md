# Lessons

Hard-won notes on running TgTRG — behavior that is correct by implementation
but easy to misread, and the practical consequences for how a run should be
set up and judged. To be updated over time.

## How Large a Step May Be

The step size is the one parameter of TgTRG whose effect is nothing like its
effect in TDVP, and the reason is that imaginary-time propagators are not
unitary. A real-time step cannot change the scale of anything; an
imaginary-time one spreads the local spectrum over \(e^{\tau \Delta E}\) in
magnitude, where \(\Delta E\) is the range of the effective Hamiltonian being
exponentiated. Everything below follows from that single exponent.

### Not the only accuracy knob

Start with the case where the step size does nothing at all. When the bond
dimension is large enough that the manifold is the whole space, the tangent
projector is the identity and the projector splitting integrates the flow
**exactly** — at either order, and for any \(\tau\). Verified to `1e-10` on an
`L = 4` chain in a single `tau = 8` step, and the test
`test_full_rank_step_is_exact` keeps it honest.

So the exponential grid is not making an approximation when it takes
\(\tau = \beta\). What limits the step is arithmetic, and it comes in two
regimes with very different manners of failing.

### Regime 1: the column stops being representable

Inside `lanczos_expv` the propagated vector is the first column of
\(e^{-\tau T}\), assembled as a sum over the eigenvectors of the projected
matrix with weights \(e^{-\tau \lambda_j} \cdot \text{overlap}_j\). Those
weights span \(e^{\tau \Delta E}\), and float64 holds exponents to about
\(\pm 709\), so the sum stops being representable once \(\tau \Delta E\)
passes roughly \(1400\).

Probing one real step on the `L = 4` chain, whose local spectrum runs over
\([-2.24, +2.24]\):

| τ | τ·ΔE | smallest / largest naive column norm | columns lost |
|---|------|--------------------------------------|--------------|
| 0.5 | 2 | 6.3e-1 / 1.5e0 | none |
| 8 | 36 | 1.3e-4 / 7.7e3 | none |
| 32 | 143 | 2.9e-16 / 3.5e15 | none |
| 128 | 572 | 7.1e-63 / 1.4e62 | none |
| 320 | 1431 | 4.2e-156 / `inf` | 13 of 18 non-finite |
| 512 | 2290 | 0 / `inf` | 4 zero, 13 non-finite |

This regime is closed off by construction: `_propagator_coeffs` assembles the
column in log space and divides out its largest term, so the result is always
a unit vector and the magnitude lives in `log_scale`. See
[Why the scale is measured every call](#why-the-scale-is-measured-every-call)
for why that maximum is taken per call and not precomputed.

### Regime 2: the exponent exceeds what float64 can resolve

The regime that remains open is forty times earlier in \(\tau\) and far less
polite, because the run completes and reports plausible numbers. The backward
bond update propagates by \(e^{+\tau H^{(0)}}\), amplifying the high-energy end
of the local spectrum; once that amplification exceeds the inverse of float64
resolution, what it amplifies is roundoff.

Measured on `L = 4` free fermions, varying the hopping so that \(\Delta E\)
changes while nothing else does. Note that `order=2` propagates in half-steps
of \(\tau/2\), so the exponent of a single propagation is
\((\tau/2)\,\Delta E\):

| t | ΔE | τ | (τ/2)·ΔE | relative error on log Z |
|---|-----|---|----------|-------------------------|
| 1 | 4.5 | 0.5 | 1.1 | 4.0e-14 |
| 1 | 4.5 | 8 | 17.9 | 4.1e-14 |
| 2 | 8.9 | 8 | 35.8 | 2.0e-14 |
| 1 | 4.5 | 32 | 71.6 | 9.0e-3 |
| 4 | 17.9 | 8 | 71.6 | unphysical: the run aborts |

The two failures sit at the same exponent, reached by different routes, and
the safe cases are at or under \(36 \approx \log(1/\varepsilon)\) for float64.
That is the criterion:

\[
\frac{\tau}{2}\, \Delta E \lesssim \log(1/\varepsilon) \approx 36
\]

for `order=2`, and \(\tau \Delta E \lesssim 36\) for `order=1`. The limit is on
the **exponent**, not on \(\tau\) and not on \(\beta\): a chain with a wider
spectrum reaches it at a proportionally smaller step.

Nothing downstream notices. The Krylov expansions converge to `1e-13` in the
failing step exactly as in the good ones, because they solve the local problem
they were handed correctly — what drifts is the state they are applied to. The
`1e-13` is not a reassurance.

### What the warning does and does not cover

`run()` warns when a schedule contains a step past `_TAU_SOFT_LIMIT = 8.0`.
That constant corresponds to \(\Delta E\) of order a few, which covers the
free-fermion and Hubbard chains in the tests and examples, and it is
deliberately conservative there. It is **not** universal: \(\Delta E\) grows
with the chain length and with the coupling strengths, so on a long chain or a
strongly interacting model the criterion above is reached at a step the
warning will not mention. Read the criterion, not the constant, and treat the
warning as a floor on caution, never a bound on it.

The warning is a warning and not a refusal for the same reason: the driver
does not know the spectral range, and a model normalized differently may be
perfectly fine past the constant.

### Hence the stacked schedule

The exponential grid walks into regime 2 unaided, its step size *being*
\(\beta\). So the schedule to use is the stacked one — exponential out of the
high-temperature regime where the Taylor state lives, linear from there on, as
in the [overview](index.md#stacking-the-two-grids) and both
[examples](../../examples/tgtrg/free-fermion.md). The default `n_steps` of 13
ends the exponential phase at \(\beta = 2\), where that handover belongs for a
spectrum of order a few.

## Step Size Versus Bond Dimension

Under a rank limit the step size does buy accuracy, and not much of it. It is
no substitute for the bond dimension.

Two error sources are at work, and they scale differently in the number of
steps. The splitting error is per step and shrinks with \(\tau\) in the usual
way. The projection error is a **rate**: at every instant the exact flow
\(-H\rho\) points partly out of the tangent space, and the component that
cannot be followed accrues per unit of \(\beta\), not per step. Taking more
steps over the same interval samples the same inability more finely without
removing any of it.

Cooling \(\beta = 0.5 \to 1.0\) at `max_bond = 4`, varying only how many steps
cover the interval:

| steps | τ | order 1 | order 2 |
|-------|---|---------|---------|
| 1 | 0.5 | 3.02e-3 | 2.23e-3 |
| 2 | 0.25 | 2.23e-3 | 1.78e-3 |
| 4 | 0.125 | 1.78e-3 | 1.55e-3 |
| 8 | 0.0625 | 1.55e-3 | 1.44e-3 |
| 16 | 0.031 | 1.44e-3 | 1.39e-3 |
| 64 | 0.0078 | 1.36e-3 | 1.35e-3 |

Sixty-four times the work buys a factor of 2.2, nearly all of it in the first
three refinements, and the floor of `1.35e-3` is the manifold's. For scale,
the state handed in at \(\beta = 0.5\) already carried `1.2e-4` from its own
compaction to rank 4, and the best a rank-4 MPO can do at \(\beta = 1.0\) is
`3.5e-3` — the tangent trajectory beats the pointwise truncation of the exact
\(\rho\), which is worth knowing but not something to rely on.

The order columns make the point from another angle. `order=2` is a forward
half-sweep of \(\tau/2\) and a backward one, so order 2 at `n` steps and order
1 at `2n` steps perform the same number of half-sweeps at the same size — and
they agree to every digit shown, diagonal by diagonal. In this regime the
integrator order earns nothing, because what is being measured is not a
discretization error at all.

**So tune in one order.** Set `max_bond` from the accuracy you need, judged on
`Summary.discarded_weights`; then take \(\tau\) as large as the exponent
criterion allows, the sweeps it saves costing a fraction of what the rank
already concedes. A refinement of \(\tau\) is worth one or two attempts after
that, and almost nothing past them.

## Why the Scale Is Measured Every Call

`_propagator_coeffs` recomputes its log-space shift on every call, which
invites the question of whether one estimate of \(\lVert H \rVert\), taken
once before the run, could serve instead. It cannot, and it would save
nothing.

The first thing to see is that the shift does not affect the answer. The
routine forms `exp(logs - c)`, assembles, normalizes, and folds `c` back into
`log_scale`; for *any* `c` the returned coefficients and `log_scale` are
identical. The shift's only job is to keep the intermediate exponentials in
range, so the question is not whether a global constant is accurate but
whether it can keep them there.

It cannot, for two reasons. The exponents are not a property of \(H\) alone:
each term carries \(-\tau \lambda_j + \log|\text{overlap}_j|\), and the
overlap is between the local starting vector and the eigenvectors of the
projected matrix — that is the state, and it changes at every site, every
half-sweep, and every step. A shift drawn from the spectrum alone misses it,
and when the propagator's favored eigenvector has little weight in the
starting vector, every surviving coefficient is small and the column
underflows to zero together. And on the spectrum itself a global
\(\lVert H \rVert\) is only an upper bound — the local effective Hamiltonians
are compressions of \(H\) by the isometries of the canonical form, so their
eigenvalues lie inside \(H\)'s range but need not fill it. A bound loose by
more than about 700 in the exponent underflows the largest term, which is
regime 1 again.

Nor is there anything to save. The `eigh` is not optional: applying
\(e^{-\tau T}\) requires the spectral decomposition, and \(T\) is tridiagonal
of dimension at most `krylov_dim`. Timed over an `L = 8` run of 14 steps to
\(\beta = 4\) with `D` reaching 128 — 2,925 propagator solves — all of
`_propagator_coeffs` accounted for 0.49 % of a 37-second wall time, and the
log-space assembly within it for 0.08 %. The average Krylov dimension was 5.3.
Everything expensive is elsewhere: the matvecs at \(O(D^3)\) and the SVD
splits.

## Why `'1sp'` Re-Gauges After Expanding

The expansion stacks the complement onto the kept basis with `oplus`, then
immediately re-decomposes the stack to recover an exact isometry. TDVP's
expansion does without that step, so it looks redundant. It is a conditioning
fix, and the condition it guards against is where cooling starts.

### The construction is exact

Write the halves and the neighbor with the indices the code uses,

\[
\mathcal{L}_{a\alpha prs}, \qquad \mathcal{R}_{\alpha pcuv}, \qquad
M_{bcuv}, \qquad \sum_{cuv} M^{*}_{bcuv} M_{dcuv} = \delta_{bd},
\]

so the kept space of the bond is the span of \(M\) over the composite index
\((cuv)\). The projection subtracts exactly that span, leaving

\[
\sum_{cuv} \mathcal{R}^{\mathrm{d}}_{\alpha pcuv} M^{*}_{bcuv} = 0,
\]

and the doubly-discarded tensor
\(\Theta'_{acrsuv} = \sum_{\alpha p} \mathcal{L}^{\mathrm{d}}_{a\alpha prs} \mathcal{R}^{\mathrm{d}}_{\alpha pcuv}\)
inherits it through the \(\alpha p\) sum. Its singular vectors \(N_{ncuv}\)
are then orthogonal to the kept space, the stack is isometric, and \(\rho\)
comes through the expansion unchanged. None of that is approximate, and the
measurement agrees: projecting \(\mathcal{R}^{\mathrm{d}}\) a second time
finds 1e-15 left in the kept space, at every β. The projection is not the
weak step.

### The condition

What fails is the conditioning of the contraction, and the quantity that
decides it is

\[
\frac{\lVert \Theta' \rVert}{\lVert \mathcal{L}^{\mathrm{d}} \rVert \, \lVert \mathcal{R}^{\mathrm{d}} \rVert},
\]

which can sit far below one: the corner discarded at *both* ends is much
smaller than either single-sided discard suggests. That happens whenever the
bond's kept space already nearly exhausts what \(H\) can reach from the
current \(\rho\). Leaving the kept space on one side of the bond is first
order in \(H\); leaving it on both sides at once needs \(H\) to act twice, so
the doubly-discarded corner is second order where the halves that build it
are first, and the ratio is quadratically small.

A cooling run begins exactly there, for a sharper reason than
\(\rho(\tau_0) \approx \mathbb{1}\). The Taylor state is a Taylor-accurate
\(e^{-\tau_0 H}\), so its bonds are built from \(H\) and already nearly span
what \(H\) reaches. This is the approach to saturation, whose endpoint the
saturation guard and `_noise_floor` already decline. The re-gauge covers the
band just above that floor, where a genuine complement exists but arrives
with five fewer correct digits than usual.

### What the ratio costs

In floating point \(\mathrm{fl}(\Theta') = \Theta' + E\) with
\(\lVert E \rVert \sim \epsilon \lVert \mathcal{L}^{\mathrm{d}} \rVert \lVert \mathcal{R}^{\mathrm{d}} \rVert\).
The decisive property is that \(E\) does not factorize: the \(\alpha p\) sum
is rounded independently for each output entry, so no \(X\) exists with
\(E_{acrsuv} = \sum_{\alpha p} X_{a\alpha prs} \mathcal{R}^{\mathrm{d}}_{\alpha pcuv}\),
and the exactness above needed that form to route the \((cuv)\) contraction
into the zero. The leak into the kept space is therefore absolute, fixed by
the two inputs, while the signal is the small \(\Theta'\):

\[
\mu_{nb} \equiv \sum_{cuv} N_{ncuv} M^{*}_{bcuv}
\;\sim\; \frac{\epsilon \, \lVert \mathcal{L}^{\mathrm{d}} \rVert \lVert \mathcal{R}^{\mathrm{d}} \rVert}{\lVert \Theta' \rVert}.
\]

\(\mu\) is the off-diagonal block of the stacked neighbor's Gram matrix, and
the fill-in contracts \(\Theta\) against that Gram, so it returns
\(\Theta(1 + \mu)\) and \(\rho\) moves by \(\lVert \mu \rVert\). Expanding one
interior bond of an `L = 6` chain at rank 6:

| β | \(\lVert \Theta' \rVert / \lVert \mathcal{L}^{\mathrm{d}} \rVert \lVert \mathcal{R}^{\mathrm{d}} \rVert\) | \(\lVert \mu \rVert\) | Δρ, stacked | Δρ, re-gauged |
|---|---|---|---|---|
| 0.05 | 5.5e-06 | 2.5e-11 | 2.5e-11 | 2.8e-16 |
| 0.2 | 3.4e-04 | 4.1e-13 | 4.1e-13 | 8.8e-16 |
| 1.0 | 2.6e-02 | 1.9e-15 | 6.3e-16 | 4.8e-16 |
| 4.0 | 3.8e-02 | 4.1e-16 | 3.1e-16 | 3.6e-16 |

The second column sets the third and the third is what moves \(\rho\). The
last two rows are the same code on a generic \(\rho\), with nothing to fix.

### Why the fix is a re-decomposition

A more careful projection cannot help, the projection's own residual being
amplified identically: contracted against \(\mathcal{L}^{\mathrm{d}}\), the
1e-15 that survives the second projection is itself 1e-10 of \(\Theta'\). The
only quantity that matters is how far \(\lVert \Theta' \rVert\) sits below the
product of the halves, so the cure has to avoid that ratio instead of
improving the inputs. `decomp` orthogonalizes against its own input, with an
error of \(\epsilon\) relative to \(\lVert M^{\mathrm{exp}} \rVert\) and the
ratio in no denominator. It is inert in exact arithmetic, returning a
different basis for the same span against which the fill-in reproduces
\(\rho\) identically, so it removes a failure mode and changes no result.

The 2.5e-11 earns this page because nothing downstream would flag it. It is
not a truncation, so it never reaches `Summary.discarded_weights`; it enters
`log Z` directly, at every step, and accumulates. The asymmetry with TDVP is
one of regime and not of real against imaginary time: a product-state quench
leaves almost everything \(H\) can reach outside a bond of dimension 1, the
ratio is of order one, and the complement comes back orthogonal to
\(\epsilon\). Cooling has no such option, starting in the bad regime on its
first step.

## `run()` Cools In Place

`run()` mutates the `rho` inside the `Artifact` it is handed. The stacked
schedule wants exactly that — the artifact one phase returns is passed
straight into the next — and it matches the rest of Alice, but it means an
`Artifact` loaded from a checkpoint is consumed by the first `run()` it is
given to. Branching two continuations off one checkpoint therefore needs the
state reloaded, or copied, between them; a second `run()` on the same object
starts from wherever the first one left it, and reports the \(\beta\) it was
asked for instead of the \(\beta\) it reached.

## See Also

- [Overview](index.md) — the schemes, the grid modes, and the stacked schedule.
- [Options](options.md) — `max_bond`, `trunc_thresh`, `order`, and the grid parameters.
- [Summary](summary.md) — which columns carry the grid's error and which do not.
- [Scheme comparison](../../examples/tgtrg/scheme-comparison.md) — the three schemes side by side, and sizing `expand_k`.
- [XTRG Lessons](../xtrg/lessons.md) — the other route to the same `NormalMPO`, by squaring instead of cooling.
