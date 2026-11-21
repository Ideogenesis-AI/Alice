# Algorithms

Alice's tensor network algorithms, each built on the shared MPS/MPO core documented in the [API Reference](../api/index.md).

## DMRG

Ground-state search via alternating sweep optimization of MPS tensors.

| Symbol | Description |
|--------|-------------|
| [Overview](dmrg/index.md) | Update schemes and usage pattern |
| [Options](dmrg/options.md) | DMRG run options |
| [Summary](dmrg/summary.md) | DMRG output dataclass |
| [Launch](dmrg/run.md) | Top-level DMRG entry point |

## VUMPS

Ground-state search directly in the thermodynamic limit, on a uniform MPS described by a unit cell.

| Symbol | Description |
|--------|-------------|
| [Overview](vumps/index.md) | Update schemes, the gradient, environments as fixed points, usage pattern |
| [Options](vumps/options.md) | VUMPS run options |
| [Summary](vumps/summary.md) | VUMPS output dataclass |
| [Lessons](vumps/lessons.md) | Practical notes, starting with the environment budget |
| [Launch](vumps/run.md) | Top-level VUMPS entry point |

## TDVP

Real-time evolution under a Hamiltonian MPO, by projecting the Schrödinger equation onto the tangent space of the MPS manifold.

| Symbol | Description |
|--------|-------------|
| [Overview](tdvp/index.md) | Update schemes, integrator order, the tangent-space projection, usage pattern |
| [Options](tdvp/options.md) | TDVP run options |
| [Summary](tdvp/summary.md) | TDVP output dataclass |
| [Artifact](tdvp/artifact.md) | State snapshot at one time step |
| [Launch](tdvp/run.md) | Top-level TDVP entry point |

## TgTRG (tangent-space TRG)

Finite-temperature thermodynamics by cooling the thermal density matrix along the tangent space of the MPO manifold — imaginary-time TDVP for ρ, with a β grid that need not double.

| Symbol | Description |
|--------|-------------|
| [Overview](tgtrg/index.md) | Update schemes, integrator order, grid modes, step size, usage pattern |
| [Options](tgtrg/options.md) | TgTRG run options |
| [Summary](tgtrg/summary.md) | TgTRG output dataclass |
| [Artifact](tgtrg/artifact.md) | Density-matrix snapshot at one cooling step |
| [Lessons](tgtrg/lessons.md) | Practical notes, starting with how large a step may be |
| [Launch](tgtrg/run.md) | Top-level TgTRG entry point |

## XTRG

Finite-temperature thermodynamics via exponential cooling of the thermal density matrix.

| Symbol | Description |
|--------|-------------|
| [Overview](xtrg/index.md) | Update schemes, temperature grid, usage pattern |
| [Options](xtrg/options.md) | XTRG run options |
| [Summary](xtrg/summary.md) | XTRG output dataclass |
| [Artifact](xtrg/artifact.md) | Density-matrix snapshot at one cooling step |
| [Lessons](xtrg/lessons.md) | Practical pitfalls, starting with the absolute truncation scale |
| [Launch](xtrg/run.md) | Top-level XTRG entry point |

## See Also

- [API Reference](../api/index.md) — Network, Interaction, Geometry, Local Space, and Hamiltonian building blocks.
- [Examples](../examples/index.md) — complete, runnable walkthroughs for every algorithm.
