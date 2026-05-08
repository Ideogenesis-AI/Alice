# observe

Compute the expectation value of an observable for a given state.

::: alice.observe.observe
    options:
      heading_level: 2

## Notes

`observe` performs a left-to-right MPS–MPO–MPS transfer-matrix sweep. At each site it contracts the bra (conjugated MPS tensor), the MPO tensor, and the ket (MPS tensor) into an updated environment. The final environment is a 1×1×1 tensor; the scalar value — corrected for SU(2) Bridge normalization — is returned.

## See Also

- [MPS](mps.md), [MPO](mpo.md) — input types.
- [build_hamiltonian](../hamiltonian/build-hamiltonian.md) — create the MPO to pass here.
- [dmrg.run](../dmrg/run.md) — uses `observe` internally for energy evaluation.
