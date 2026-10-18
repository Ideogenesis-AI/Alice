# Lessons

Hard-won notes on running XTRG — behavior that is correct by implementation but
easy to misread, and the practical consequences for how a run should be set up
and judged. To be updated over time.

## Truncation Scale

`trunc_thresh` is an **absolute** singular value cutoff, not a relative
discarded-weight tolerance. Because XTRG squares normalized MPOs, the scale of
the object being fitted shrinks exponentially with chain length, so the same
`trunc_thresh` becomes exponentially more aggressive as the chain grows.

### What `trunc_thresh` does

`trunc_thresh` is forwarded to Nicole's `thresh` truncation mode, which keeps
singular values satisfying \(s \geq\) `trunc_thresh` within each charge sector.
The comparison is against the bare singular value — there is no division by the
largest singular value of the sector or of the full spectrum. A sector whose
entire spectrum lies below the cutoff is dropped completely.

!!! warning "The cutoff is not scale-free"

    A relative criterion (discarded weight, or \(s / s_\mathrm{max}\)) is
    invariant under rescaling the tensor being decomposed. An absolute cutoff
    is not. Whenever the magnitude of the local tensor changes, the same
    `trunc_thresh` corresponds to a different *relative* accuracy.

### Why the scale shrinks with chain length

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

So the product being fitted is already a factor \(2^{-N/2}\) below unit norm
before any truncation happens, and a fixed `trunc_thresh` sits that much higher
in the spectrum. Small correlation components are discarded prematurely, and
neither the final normalization nor a larger `max_bond` can recover them.

As an example, going from 81 to 108 sites reduces the product scale by
\(2^{27/2} \approx 1.2 \times 10^4\). Preserving the relative strictness of
`trunc_thresh = 1e-20` at 81 sites would require roughly `8.6e-25` at 108 sites.

!!! note "This is a scale estimate, not an error bound"

    The \(2^{-N/2}\) factor is a high-temperature estimate of the magnitude of
    the fitted product. It is neither a convergence guarantee nor a bound on
    the error of any observable. Finite-temperature spectra and the local
    fitting environments both matter, and the estimate says nothing about them.

### Where this does not apply

`compact()` on a [NormalMPO](../../api/network/normal-mpo.md) extracts the
Frobenius norm into `log_scale` and normalizes the site tensors *before* the
truncating sweep, so its cutoff always acts on a unit-scale spectrum. Both
[thermal_mpo](../../api/network/thermal-mpo.md), which builds \(\rho(\tau_0)\)
from the Taylor series, and the AutoMPO Hamiltonian construction compress this
way with their own cutoffs, and so avoid this mechanism. Their accuracy still
has to be checked on its own terms.

The mechanism does reach `z_tol`, which compares \(\lvert \lVert C \rVert -
\lVert C_\mathrm{prev} \rVert \rvert\) against a fixed number. Here the small
product scale makes the criterion too *permissive*, declaring the inner fit
converged earlier than intended, and deserves separate scrutiny.

### Practical guidance

- Judge a run by the convergence of observables under varying `trunc_thresh`
  and `max_bond`, not by whether the run completes.
- Unusually small bond dimensions or unusually fast cooling are not evidence of
  accuracy; on a long chain they are the expected symptom of an over-aggressive
  effective cutoff.
- When moving to a longer chain, scale `trunc_thresh` down rather than carrying
  over the value that worked on a shorter one.

## See Also

- [Options](options.md) — `trunc_thresh`, `max_bond`, and `z_tol`.
- [NormalMPO](../../api/network/normal-mpo.md) — scale handling and `compact()`.
- [thermal_mpo](../../api/network/thermal-mpo.md) — the initial density matrix.
