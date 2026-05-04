# Changelog

## 0.1.0 — May 4, 2026

Initial stable release of Alice.

### Tensor Network Infrastructure

- `MPS`, `MPO`, and `Network` classes: block-sparse matrix product states and operators inheriting Nicole's exact symmetry engine with full support for U(1), Z(2), SU(2), and product symmetry groups.
- `Network.canonical()`: left/right QR canonicalization with configurable bond truncation.
- `Network.norm()` and `normalize()`: Frobenius norm via the center tensor or a full transfer-matrix sweep.
- `Network.serialize()` / `Network.deserialize()`: `torch.save`-compatible serialization preserving all index metadata.
- `MPO.compact()` and `MPO.redistribute_norm()`: SVD compression and norm redistribution.
- `observe()`: expectation-value computation via MPS–MPO–MPS transfer-matrix sweep, with SU(2) Bridge weight support.

### AutoMPO Construction

- `build_interaction()`: TOML-configured interaction map builder with built-in Heisenberg, free-fermion, and Hubbard model presets and support for user-defined model functions via `[plugin]` sections.
- `build_hamiltonian()`: term-by-term MPO assembler using Nicole's `oplus`; compresses the result with two canonical sweeps; handles non-Abelian SU(2).
- `Interaction`, `Interaction1Site`, `Interaction2Site` dataclasses.
- Incremental `compact_every` parameter to limit bond dimension growth during large builds.

### Lattice Geometries

- `intrcmap_1dchain()`: nearest-neighbor 1D chain with optional PBC.
- `intrcmap_square()`: 2D square lattice with NN and NNN bonds, PBC support, and snake-order traversal.
- `generate_snake_order()`: boustrophedon traversal for rectangular lattices.
- `build_geometry()`: TOML dispatcher for lattice and traversal selection.

### Physical Spaces

- `build_bosonic()`: spin-`s` site; U(1) and SU(2) symmetry.
- `build_fermionic()`: spinless-fermion site; U(1) and Z(2) symmetry.
- `build_conductor()`: spinful-fermion (Band) site; U(1)×U(1), Z(2)×U(1), U(1)×SU(2), Z(2)×SU(2) symmetry.

### Physics Models

- `build_heisenberg()`: Heisenberg spin model with NN coupling `J` and optional NNN `J'`.
- `build_free_fermion()`: spinless tight-binding chain with hopping `t`, optional NNN `t'`, and chemical potential `µ`.
- `build_hubbard()`: Hubbard model with hopping `t`, on-site `U`, optional NNN `t'`, and `µ` at half-filling.

### DMRG Algorithm

- `dmrg.Options` dataclass: all run parameters with TOML loading via `Options.from_toml()`.
- `dmrg.Summary` dataclass: energy, optimized MPS, convergence info, and serialization.
- `dmrg.run()`: alternating sweep optimization with three update schemes (`1s`, `2s`, `1sp`).
- Davidson eigensolver with thick restart operating directly in the symmetry-block-sparse space.
- `Environment` class with optional disk-spilling (sliding window + async I/O).

### Logging

- `configure_logging()`: one-call setup of Alice's two-handler logging scheme (console + timestamped file). Emits startup banner, model specification summary, sweep-by-sweep diagnostics, and geometry ASCII diagrams.

### Statistics

- 538 tests across 19 test modules.
- 196 commits across 10+ feature branches.
- 64 files, ~16,000 lines of code.
- 16 source modules in three subpackages: `alice.network`, `alice.physics`, `alice.algorithm.dmrg`.
