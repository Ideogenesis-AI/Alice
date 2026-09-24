# InteractionNSite

N-site interaction term spanning a contiguous window of MPO sites.

::: alice.InteractionNSite
    options:
      heading_level: 2

## Notes

Unlike `Interaction1Site` and `Interaction2Site`, which name their sites explicitly and let `build_hamiltonian` synthesize the operator string in between, an `InteractionNSite` carries the **entire window verbatim**. The window is the contiguous range `[min(sites), max(sites)]`, and `tnsrs` must supply one tensor per site in that range — including any identity or Jordan-Wigner string tensors on sites that carry no operator.

- `sites` records the operator site indices in *operator order* (e.g. `[m, n, k, l]` for `c†_m c†_n c_k c_l`). This order is metadata for the model builder; `build_hamiltonian` only reads `min(sites)` and `max(sites)` to locate the window.
- `tnsrs[offset]` is placed at site `min(sites) + offset`. Every tensor has axes `(L_bond_IN, R_bond_OUT, bra_OUT, ket_IN)`; the two outer bonds of the window must be trivial (dim-1, charge-neutral) so the term composes with the identity tensors on the remaining sites.
- `build_hamiltonian` scales only the **last** window tensor by `cpl` — do not bake the coupling into the tensors.

Since the window is taken verbatim, a term whose operators are not contiguous is expressed by padding the gaps with identity (or string) tensors rather than by splitting the term.

## See Also

- [Interaction](interaction.md) — base class.
- [Interaction2Site](interaction-2site.md) — two-site specialization with an implicit operator string.
- [build_hamiltonian](../hamiltonian/build-hamiltonian.md) — consumes these objects.
