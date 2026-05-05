# API Reference

Welcome to the Alice API Reference. All public classes and functions are documented here, organized by conceptual topic.

## Network

Matrix product states, operators, and measurement.

| Symbol | Description |
|--------|-------------|
| [MPS](network/mps.md) | Matrix product state |
| [MPO](network/mpo.md) | Matrix product operator |
| [Network](network/network.md) | Base tensor network chain |
| [observe](network/observe.md) | Compute expectation values |

## Interaction

Interaction term dataclasses and the TOML-driven builder.

| Symbol | Description |
|--------|-------------|
| [Interaction](interaction/interaction.md) | Base interaction term |
| [Interaction1Site](interaction/interaction-1site.md) | On-site interaction |
| [Interaction2Site](interaction/interaction-2site.md) | Two-site interaction |
| [build_interaction](interaction/build-interaction.md) | Build interactions from TOML config |

## Geometry

Lattice geometry builders and traversal-order generators.

| Symbol | Description |
|--------|-------------|
| [generate_snake_order](geometry/generate-snake-order.md) | Snake-like traversal for 2D lattices |
| [intrcmap_1dchain](geometry/intrcmap-1dchain.md) | Interaction map for 1D chains |
| [intrcmap_square](geometry/intrcmap-square.md) | Interaction map for 2D square lattices |
| [build_geometry](geometry/build-geometry.md) | TOML dispatcher for geometry builders |

## Local Space

Physical Hilbert space builders for bosonic and fermionic sites.

| Symbol | Description |
|--------|-------------|
| [build_bosonic](local-space/build-bosonic.md) | Spin-`s` site with MPO operator templates |
| [build_fermionic](local-space/build-fermionic.md) | Spinless-fermion site with JW templates |
| [build_conductor](local-space/build-conductor.md) | Spinful-fermion (Band) site |

## Hamiltonian

MPO Hamiltonian assembly and built-in physics models.

| Symbol | Description |
|--------|-------------|
| [build_hamiltonian](hamiltonian/build-hamiltonian.md) | Assemble Hamiltonian MPO from interactions |
| [build_heisenberg](hamiltonian/build-heisenberg.md) | Heisenberg spin model |
| [build_free_fermion](hamiltonian/build-free-fermion.md) | Spinless free-fermion (tight-binding) model |
| [build_hubbard](hamiltonian/build-hubbard.md) | Hubbard model |

## DMRG

Ground-state DMRG algorithm.

| Symbol | Description |
|--------|-------------|
| [Options](dmrg/options.md) | DMRG run options |
| [Summary](dmrg/summary.md) | DMRG output dataclass |
| [run](dmrg/run.md) | Top-level DMRG entry point |

## Logging

| Symbol | Description |
|--------|-------------|
| [configure_logging](logging.md) | Set up Alice's logging scheme |

---

## Quick Links by Task

### I want to...

**Run DMRG on a spin chain**
→ [build_interaction](interaction/build-interaction.md), [build_hamiltonian](hamiltonian/build-hamiltonian.md), [dmrg.run](dmrg/run.md)

**Define a model from a TOML file**
→ [build_interaction](interaction/build-interaction.md) — see [AutoMPO from TOML](../examples/autompo-toml.md)

**Use a 2D square lattice geometry**
→ [intrcmap_square](geometry/intrcmap-square.md), [generate_snake_order](geometry/generate-snake-order.md)

**Define a custom geometry**
→ [build_geometry](geometry/build-geometry.md) — see [Custom geometry](../examples/extensions/custom-geometry.md)

**Build a custom local space**
→ [build_bosonic](local-space/build-bosonic.md) / [build_fermionic](local-space/build-fermionic.md) / [build_conductor](local-space/build-conductor.md) — see [Custom local space](../examples/extensions/custom-space.md)

**Compute an expectation value**
→ [observe](network/observe.md)

**Save and reload a result**
→ [Network.serialize / deserialize](network/network.md), [dmrg.Summary](dmrg/summary.md)

**Configure logging and diagnostics**
→ [configure_logging](logging.md)

**See complete working examples**
→ [Examples](../examples/index.md)
