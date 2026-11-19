# build_ihamiltonian

Build a Hamiltonian iMPO over a unit cell from a list of `Interaction` objects.

::: alice.build_ihamiltonian
    options:
      show_source: false
      heading_level: 2

## Notes

The interactions come from `build_interaction` on a geometry whose `bcx` is `IBC` and whose `lx` is the length of the unit cell, not of a chain. An interaction that reaches past the end of the cell wraps onto the next copy, the way a translation-invariant chain is described by a finite list.

The operator is left uncompressed on purpose. The environment solver keys on its automaton structure, start channel first and finish channel last; see [iMPO](../network/impo.md).

## See Also

- [build_interaction](../interaction/build-interaction.md) — produces the `interactions` list.
- [build_hamiltonian](build-hamiltonian.md) — the finite counterpart.
- [iMPO](../network/impo.md) — returned type.
- [vumps.run](../../algorithms/vumps/run.md) — passes this iMPO to the VUMPS solver.
