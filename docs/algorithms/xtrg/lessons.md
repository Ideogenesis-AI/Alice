# Lessons

Hard-won notes on running XTRG — behavior that is correct by implementation but
easy to misread, and the practical consequences for how a run should be set up
and judged. To be updated over time.

## Truncation Scale

`trunc_thresh` is a **relative** singular value cutoff. The object being
truncated inside a squaring step has a Frobenius norm that is exponentially
small in the chain length, so the conversion into the absolute cutoff that
Nicole's `thresh` mode expects is performed by the algorithm, at each
truncation point, against the local scale.

### What `trunc_thresh` does

Writing `trunc_thresh` as \(\epsilon\), a singular value is discarded when

\[
s < \epsilon\, \lVert C \rVert_F,
\]

where \(C\) is the first tensor the half-sweep decomposes. In canonical form
that tensor carries the full Frobenius norm of the compressed density matrix,
so the comparison is against the weight actually present in the state. A
sector whose entire spectrum lies below the cutoff is dropped completely.

The conversion lives in `_Truncation` in `alice.algorithm.xtrg.sweep`, which
measures the scale at the first truncation of a half-sweep and holds it for
the splits, center moves, and trial SVDs that follow. One measurement is
enough because \(\lVert C \rVert\) grows by well under a factor of two over a
sweep, as the variational fit tightens; the first tensor carries the smallest
norm, which also makes it the conservative choice. Nicole's own `thresh` mode
is absolute — it compares bare singular values — and never sees the relative
number.

### Why the scale needs converting

The squaring step fits \(C \approx A \cdot B\) where \(A\) and \(B\) are the
unit-norm internal tensors of the two `NormalMPO` factors (the physical
magnitude lives in `log_scale` and is folded back in only after the sweeps
finish). The truncations inside the variational sweeps therefore act on an
object whose Frobenius norm is \(\lVert A \cdot B \rVert_F\), not \(1\).

Near infinite temperature the density matrix is proportional to the identity.
For \(N\) spin-½ sites, with Hilbert space dimension \(D = 2^N\),

\[
A = \frac{I}{\sqrt{D}}, \qquad \lVert A \rVert_F = 1, \qquad
\lVert A^2 \rVert_F = \frac{1}{\sqrt{D}} = 2^{-N/2}.
\]

So the product being fitted starts a factor \(2^{-N/2}\) below unit norm, and
climbs back toward \(1\) as the run cools and \(\rho\) approaches a projector.
An absolute cutoff applied to that spectrum would be exponentially too
aggressive on a long chain — discarding small correlation components that
neither the final normalization nor a larger `max_bond` can recover — and too
permissive late in the run. Measuring the scale as the sweeps go removes both
failure modes, and makes a `trunc_thresh` tuned on a short chain or a warm run
carry over unchanged.

!!! note "Relative is not an error bound"

    A relative cutoff fixes the *scale* the threshold is measured against. It
    is neither a convergence guarantee nor a bound on the error of any
    observable: truncation error still accumulates across bonds, sweeps, and
    cooling steps, and the fitting environments matter too.

### The same reasoning for `z_tol`

`z_tol` compares \(\lvert \lVert C \rVert - \lVert C_\mathrm{prev} \rVert
\rvert\) between full sweeps. It is relative to \(\lVert C_\mathrm{prev}
\rVert\) for exactly the reason above: measured absolutely, the small product
scale would satisfy the criterion immediately on a long chain and stop the
inner fit after a single sweep. `z_tol = 0.0` still means "never stop early"
and always runs the full `n_sweeps` budget.

### Where this does not apply

`compact()` on a [NormalMPO](../../api/network/normal-mpo.md) extracts the
Frobenius norm into `log_scale` and normalizes the site tensors *before* the
truncating sweep, so its cutoff always acts on a unit-scale spectrum and takes
`trunc_thresh` directly. Both
[thermal_mpo](../../api/network/thermal-mpo.md), which builds \(\rho(\tau_0)\)
from the Taylor series, and the AutoMPO Hamiltonian construction compress this
way with their own cutoffs. Their accuracy still has to be checked on its own
terms.

## See Also

- [Options](options.md) — `trunc_thresh`, `max_bond`, and `z_tol`.
- [NormalMPO](../../api/network/normal-mpo.md) — scale handling and `compact()`.
- [thermal_mpo](../../api/network/thermal-mpo.md) — the initial density matrix.
