# Algorithms

Alice's tensor network algorithms, each built on the shared MPS/MPO core documented in the [API Reference](../api/index.md).

## DMRG

Ground-state search via alternating sweep optimization of MPS tensors.

| Symbol | Description |
|--------|-------------|
| [Overview](dmrg/index.md) | Update schemes and usage pattern |
| [Options](dmrg/options.md) | DMRG run options |
| [Summary](dmrg/summary.md) | DMRG output dataclass |
| [run](dmrg/run.md) | Top-level DMRG entry point |

## XTRG

Finite-temperature thermodynamics via exponential cooling of the thermal density matrix.

| Symbol | Description |
|--------|-------------|
| [Overview](xtrg/index.md) | Update schemes, temperature grid, usage pattern |
| [Options](xtrg/options.md) | XTRG run options |
| [Summary](xtrg/summary.md) | XTRG output dataclass |
| [run](xtrg/run.md) | Top-level XTRG entry point |

## See Also

- [API Reference](../api/index.md) — Network, Interaction, Geometry, Local Space, and Hamiltonian building blocks.
- [Examples](../examples/index.md) — complete, runnable walkthroughs for both algorithms.
