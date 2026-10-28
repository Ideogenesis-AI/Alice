# Network

The `alice.network` module provides the core tensor network data structures: `Network`, `MPS`, `MPO`, their infinite counterparts `iNetwork`, `iMPS`, and `iMPO`, `NormalMPO`, and associated functions. These objects are re-exported at the top-level `alice` namespace.

## Classes

| Class | Description |
|-------|-------------|
| [Network](network.md) | Base 1D tensor network chain |
| [MPS](mps.md) | Matrix product state |
| [MPO](mpo.md) | Matrix product operator |
| [iNetwork](inetwork.md) | Base 1D tensor network unit cell |
| [iMPS](imps.md) | Infinite matrix product state, described by a unit cell |
| [iMPO](impo.md) | Infinite matrix product operator, described by a unit cell |
| [NormalMPO](normal-mpo.md) | Unit-normed MPO with separately tracked scale factor |

## Functions

| Function | Description |
|----------|-------------|
| [init_mps](init-mps.md) | Construct an initial MPS for DMRG |
| [init_imps](init-imps.md) | Construct an initial unit cell for VUMPS |
| [observe](observe.md) | Compute expectation value ⟨ψ\|O\|ψ⟩ or thermal average Tr[ρO]/Tr[ρ] |
| [thermal_mpo](thermal-mpo.md) | Approximate \(e^{-\beta H}\) via Taylor expansion |

## See Also

- [Core Concepts](../../getting-started/core-concepts.md) — conceptual introduction to MPS and MPO.
- [Interaction](../interaction/index.md) — Interaction dataclasses used together with `MPO`.
- [Hamiltonian](../hamiltonian/build-hamiltonian.md) — assembles an MPO from a list of interactions.
