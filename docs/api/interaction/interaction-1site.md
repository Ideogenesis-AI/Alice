# Interaction1Site

On-site (1-site) interaction term.

::: alice.Interaction1Site
    options:
      heading_level: 2

## Notes

The `tnsr` field must have axes `(L_trivial_IN, R_trivial_OUT, bra_OUT, ket_IN)`. `build_hamiltonian` multiplies the tensor by `cpl` at accumulation time — do not bake the coupling into the tensor.

## See Also

- [Interaction](interaction.md) — base class.
- [Interaction2Site](interaction-2site.md) — two-site partner.
- [build_hamiltonian](../hamiltonian/build-hamiltonian.md) — consumes these objects.
