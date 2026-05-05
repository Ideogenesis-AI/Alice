# Network

The `alice.network` module provides the core tensor network data structures: `MPS`, `MPO`, `Network`, and the `observe` function. These objects are re-exported at the top-level `alice` namespace.

## Classes

| Class | Description |
|-------|-------------|
| [Network](network.md) | Base 1D tensor network chain |
| [MPS](mps.md) | Matrix product state |
| [MPO](mpo.md) | Matrix product operator |

## Functions

| Function | Description |
|----------|-------------|
| [observe](observe.md) | Compute expectation value ⟨ψ\|O\|ψ⟩ |

## See Also

- [Core Concepts](../../getting-started/core-concepts.md) — conceptual introduction to MPS and MPO.
- [Interaction](../interaction/index.md) — Interaction dataclasses used together with `MPO`.
- [Hamiltonian](../hamiltonian/build-hamiltonian.md) — assembles an MPO from a list of interactions.
