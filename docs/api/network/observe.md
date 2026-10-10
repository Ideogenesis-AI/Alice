# observe

Compute the expectation value of an observable for a given state.

::: alice.observe.observe
    options:
      heading_level: 2

## Notes

`observe` performs a left-to-right MPS–MPO–MPS transfer-matrix sweep. At each site it contracts the bra (conjugated MPS tensor), the MPO tensor, and the ket (MPS tensor) into an updated environment. The final environment is a 1×1×1 tensor; the scalar value — corrected for SU(2) Bridge normalization — is returned.

### Complex states

Nicole contracts only operands sharing a single dtype. When either the state or the observable carries complex blocks — as a state evolved in real time by [tdvp.run](../../algorithms/tdvp/run.md) does — `observe` promotes the whole sweep to `complex128`, including the real-valued MPO and the identity-derived boundary tensor. The bra side is conjugated, so the result is the physical ⟨ψ|O|ψ⟩.

The imaginary part of the final scalar is discarded once it has been checked to be at roundoff level. A larger imaginary part means the observable is not Hermitian on this state, and raises `ValueError` rather than returning a misleading real number.

## See Also

- [MPS](mps.md), [MPO](mpo.md) — input types.
- [build_hamiltonian](../hamiltonian/build-hamiltonian.md) — create the MPO to pass here.
- [dmrg.run](../../algorithms/dmrg/run.md) — uses `observe` internally for energy evaluation.
- [tdvp.run](../../algorithms/tdvp/run.md) — produces the complex states handled above.
