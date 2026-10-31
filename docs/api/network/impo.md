# iMPO

Infinite matrix product operator: a unit cell of `N` site tensors that tiles the chain.

::: alice.iMPO
    options:
      show_source: false
      heading_level: 2

## Notes

A Hamiltonian iMPO is never canonicalized or compressed, unlike an [MPO](mpo.md). It carries the upper-triangular automaton structure the environment solver reads: the first bond sector is the start channel, the last is the finish channel, and both act as the identity on the diagonal. Compressing the operator would mix those channels together and leave the environment fixed point with nothing to key on.

## See Also

- [iNetwork](inetwork.md) — base class providing bond bookkeeping and `serialize()`.
- [iMPS](imps.md) — infinite matrix product state.
- [MPO](mpo.md) — the finite counterpart.
- [build_ihamiltonian](../hamiltonian/build-ihamiltonian.md) — assemble a Hamiltonian iMPO.
- [vumps.run](../../algorithms/vumps/run.md) — the solver that consumes it.
