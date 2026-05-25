# NormalMPO

MPO subclass that tracks the overall scale factor separately from the unit-normed internal tensors.

::: alice.network.NormalMPO
    options:
      show_source: false
      heading_level: 2
      members:
        - from_mpo
        - scale
        - compact
        - trace

## Notes

**Representation.** A `NormalMPO` stores the physical operator as `_scale × mpo`, where after `compact()` or `from_mpo()` the internal MPO satisfies `norm() == 1`. Keeping the scale separate prevents numerical over/underflow when iterating a Taylor series whose terms span many orders of magnitude.

**Arithmetic.** The operators `@`, `+`, `*`, and `*` (right) build new raw tensors and update `_scale` analytically without performing any canonicalization. Call `compact()` explicitly afterward to restore the unit-norm invariant and truncate bond dimension.

**Dtype preservation.** All arithmetic keeps real-valued (`float64`) tensors real. In `__add__`, the per-site scale factor is `|α|^{1/L}`, with the sign of the relative weight `α` absorbed into the first site, avoiding complex roots when `α < 0`.

## See Also

- [MPO](mpo.md) — base class.
- [thermal_mpo](thermal-mpo.md) — constructs `exp(-βH)` as a `NormalMPO`.
- [observe](observe.md) — accepts a `NormalMPO` as `state`; computes `Tr[ρ O] / Tr[ρ]`.
