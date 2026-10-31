# API Reference

Welcome to the Alice API Reference. All public classes and functions are documented here, organized by conceptual topic. For specific 1D algorithms, see the [Algorithms](../algorithms/index.md) tab.

## Network

Matrix product states and operators, finite and infinite, with their initializers and measurement.

| Symbol | Description |
|--------|-------------|
| [Network](network/network.md) | Base tensor network chain |
| [MPS](network/mps.md) | Matrix product state |
| [MPO](network/mpo.md) | Matrix product operator |
| [iNetwork](network/inetwork.md) | Base tensor network unit cell |
| [iMPS](network/imps.md) | Infinite matrix product state |
| [iMPO](network/impo.md) | Infinite matrix product operator |
| [NormalMPO](network/normal-mpo.md) | Unit-normed MPO with a separately tracked scale |
| [init_mps](network/init-mps.md) | Construct an initial MPS |
| [init_imps](network/init-imps.md) | Construct an initial unit cell |
| [observe](network/observe.md) | Compute expectation values |
| [thermal_mpo](network/thermal-mpo.md) | Approximate \(e^{-\beta H}\) via Taylor expansion |

## Interaction

Interaction term dataclasses and the TOML-driven builder.

| Symbol | Description |
|--------|-------------|
| [Interaction](interaction/interaction.md) | Base interaction term |
| [Interaction1Site](interaction/interaction-1site.md) | On-site interaction |
| [Interaction2Site](interaction/interaction-2site.md) | Two-site interaction |
| [InteractionNSite](interaction/interaction-nsite.md) | N-site interaction over a contiguous window |
| [build_interaction](interaction/build-interaction.md) | Build interactions from TOML config |

## Geometry

Lattice geometry builders and traversal-order generators.

| Symbol | Description |
|--------|-------------|
| [Geometry](geometry/geometry.md) | Resolved lattice geometry struct |
| [build_geometry](geometry/build-geometry.md) | Construct a `Geometry` from a config dict |
| [build_intrcmap](geometry/build-intrcmap.md) | Generate the interaction map from a `Geometry` |
| [intrcmap_1dchain](geometry/intrcmap-1dchain.md) | Interaction map for 1D chains |
| [intrcmap_square](geometry/intrcmap-square.md) | Interaction map for 2D square lattices |
| [intrcmap_kagome](geometry/intrcmap-kagome.md) | Interaction map for kagome lattices |

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
| [build_ihamiltonian](hamiltonian/build-ihamiltonian.md) | Assemble Hamiltonian iMPO over a unit cell |
| [build_heisenberg](hamiltonian/build-heisenberg.md) | Heisenberg spin model |
| [build_free_fermion](hamiltonian/build-free-fermion.md) | Spinless free-fermion (tight-binding) model |
| [build_hubbard](hamiltonian/build-hubbard.md) | Hubbard model |

## Logging

| Symbol | Description |
|--------|-------------|
| [configure_logging](logging.md) | Set up Alice's logging scheme |

---

## Quick Links by Task

### I want to...

**Run DMRG on a spin chain**
→ [build_interaction](interaction/build-interaction.md), [build_hamiltonian](hamiltonian/build-hamiltonian.md), [dmrg.run](../algorithms/dmrg/run.md)

**Run VUMPS on an infinite chain**
→ [build_interaction](interaction/build-interaction.md), [build_ihamiltonian](hamiltonian/build-ihamiltonian.md), [init_imps](network/init-imps.md), [vumps.run](../algorithms/vumps/run.md)

**Evolve a state in real time**
→ [build_interaction](interaction/build-interaction.md), [build_hamiltonian](hamiltonian/build-hamiltonian.md), [tdvp.run](../algorithms/tdvp/run.md)

**Run XTRG at finite temperature**
→ [build_interaction](interaction/build-interaction.md), [build_hamiltonian](hamiltonian/build-hamiltonian.md), [xtrg.run](../algorithms/xtrg/run.md)

**Define a model from a TOML file**
→ [build_interaction](interaction/build-interaction.md) — see [AutoMPO from TOML](../examples/autompo-toml.md)

**Use a 2D square lattice geometry**
→ [intrcmap_square](geometry/intrcmap-square.md), [build_geometry](geometry/build-geometry.md)

**Define a custom geometry**
→ [build_geometry](geometry/build-geometry.md) — see [Custom geometry](../examples/extensions/custom-geometry.md)

**Build a custom local space**
→ [build_bosonic](local-space/build-bosonic.md) / [build_fermionic](local-space/build-fermionic.md) / [build_conductor](local-space/build-conductor.md) — see [Custom local space](../examples/extensions/custom-space.md)

**Compute an expectation value**
→ [observe](network/observe.md)

**Save and reload a tensor network**
→ [Network.serialize / deserialize](network/network.md)

**Configure logging and diagnostics**
→ [configure_logging](logging.md)

**See complete working examples**
→ [Examples](../examples/index.md)
