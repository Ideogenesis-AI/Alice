# Interaction2Site

Two-site interaction term.

::: alice.Interaction2Site
    options:
      heading_level: 2

## Notes

Tensor axis conventions:

- `leading_tnsr`: `(L_trivial_IN, op_OUT, bra_OUT, ket_IN)`
- `terminal_tnsr`: `(op_IN, R_trivial_OUT, bra_OUT, ket_IN)` — scaled by `cpl` at accumulation time.
- `intermid_tnsr`: `(op_IN, op_OUT, bra_OUT, ket_IN)` — required when `terminal_site > leading_site + 1`. For bosonic systems this is the physical identity dressed with operator-channel bonds; for fermionic systems it is the Jordan-Wigner string operator.

## See Also

- [Interaction](interaction.md) — base class.
- [Interaction1Site](interaction-1site.md) — on-site partner.
- [build_hamiltonian](../hamiltonian/build-hamiltonian.md) — consumes these objects.
