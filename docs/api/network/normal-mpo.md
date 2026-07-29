# NormalMPO

MPO subclass that tracks the overall scale factor separately from the unit-normed internal tensors.

::: alice.network.NormalMPO
    options:
      show_source: false
      heading_level: 2
      members:
        - from_mpo
        - log_scale
        - scale
        - scale_by
        - compact
        - log_trace
        - trace

## Notes

**Representation.** A `NormalMPO` stores the physical operator as `exp(log_scale) × mpo`, where after `compact()` or `from_mpo()` the internal MPO satisfies `norm() == 1`. Tracking the magnitude in log form as `log_scale` keeps it representable even far outside float64 range — this matters for XTRG, where `Tr[ρ]` is repeatedly squared and can reach `~10^500` or beyond at low temperature. The *sign* of the physical operator lives directly in the internal tensor data: `mpo.norm() == 1` never pins down a sign (`-X` and `X` have the same Frobenius norm), so a sign flip (e.g. `__mul__` by a negative scalar, or the alternating `(-β)^n` terms in `thermal_mpo`'s Taylor accumulation) is folded into site 0's tensor by negating it — site 0 is used by convention since, in an OBC chain, its trivial boundary bond makes it the cheapest tensor to touch. Canonicalization (`compact()`, `canonical()`) is an exact gauge transform, so a sign baked into the tensors survives it unchanged.

**Arithmetic.** The operators `@`, `+`, `*`, and `*` (right) build new raw tensors and update `log_scale` analytically (by addition, never by exponentiating a possibly-astronomical magnitude) without performing any canonicalization. Since tensor contraction and addition are linear, sign carried by the operands' tensor data propagates through `@` and `+` automatically. Call `compact()` explicitly afterward to restore the unit-norm invariant and truncate bond dimension. `scale_by()` provides the same safe, log-domain update for mutating an existing `NormalMPO`'s scale magnitude in place.

**Overflow-safe trace.** `trace()` returns the raw magnitude `Tr[ρ]` and is convenient for moderate magnitudes, but may return `±inf` once the true value exceeds float64 range. `log_trace()` returns `(log|Tr[ρ]|, sign)` instead, and never overflows in this regime — prefer it whenever the trace magnitude may be extreme (as in XTRG's cooling loop). The `sign` in this pair is derived by contracting the tensor data.

**Dtype preservation.** All arithmetic keeps real-valued (`float64`) tensors real. In `__add__`, the per-site scale factor is `α^{1/L}` where `α = other.scale / self.scale` is a non-negative magnitude ratio, so no complex roots arise.

## See Also

- [MPO](mpo.md) — base class.
- [thermal_mpo](thermal-mpo.md) — constructs `exp(-βH)` as a `NormalMPO`.
- [observe](observe.md) — accepts a `NormalMPO` as `state`; computes `Tr[ρ O] / Tr[ρ]`.
