# Changelog

## 0.1.1 — May 6, 2026

**Documentation Website**

Introduces the complete Alice documentation site: Material-themed MkDocs with a full API
reference, worked DMRG examples, and getting-started guides. Also
renames the PyPI distribution to `alice-net`. No changes to the Alice project; fully
backward compatible with v0.1.0.

### Documentation Infrastructure

- Custom `alice` color scheme, three-tab navigation, dark/light/system theme switching, search with suggestions and shareable links
- Content features: code copy, inline annotations, tabbed content, in-page edit and view actions
- Custom home page template (`docs/overrides/home.html`) with hero image and feature cards; custom stylesheet (`docs/stylesheets/extra.css`)
- Build hook (`docs/hooks.py`) for table-cell bullet-list post-processing
- Plugins: **mkdocstrings** (NumPy-style API docs), **markdown-exec** (configured for future live execution), **git-revision-date-localized**, **git-committers**
- Extensions: MathJax via `arithmatex`, Mermaid via `superfences`, tabbed content, FontAwesome/Material emoji

### Getting Started (7 pages)

- **What is Alice**: philosophy, relationship with Nicole, supported symmetry groups and algorithms
- **Installation**: `pip install alice-net` / `uv add alice-net`, development setup, optional dependency groups
- **Core Concepts**: MPS/MPO block-sparse structure, symmetry sectors, DMRG sweep logic
- **Quick Start**: end-to-end Heisenberg DMRG example from site definition through energy output
- **Contributing**: branch model, coding conventions, test requirements
- **Git Control**: tagging, branching, and release workflow for the Alice project
- **Changelog**: version history beginning with v0.1.0

### API Reference (30 pages)

- **Network** — `Network`, `MPS`, `MPO`: construction, canonicalization, norm, serialization, SVD compression, norm redistribution; `observe`: expectation-value sweep with SU(2) Bridge weight support
- **Interaction** — `Interaction`, `Interaction1Site`, `Interaction2Site` dataclass references; `build_interaction`: TOML-configured builder with plugin section documentation
- **Geometry** — `generate_snake_order`, `intrcmap_1dchain`, `intrcmap_square`, `build_geometry`: function references with parameter tables and usage notes
- **Local Space** — `build_bosonic`, `build_fermionic`, `build_conductor`: site Hilbert space constructors with symmetry-mode tables
- **Hamiltonian** — `build_hamiltonian`: MPO assembler; `build_heisenberg`, `build_free_fermion`, `build_hubbard`: model-specific builders
- **DMRG** — `dmrg.Options`: parameter reference with TOML key mapping; `dmrg.Summary`: output fields and serialization; `dmrg.run`: sweep logic, update schemes (`1s`, `2s`, `1sp`), convergence criteria
- **Logging** — `configure_logging`: handler configuration, log levels, output file naming

### Examples (8 pages)

- **Heisenberg chain**: ground-state energy of a spin-1/2 chain, U(1) and SU(2) symmetry comparison
- **Free fermion**: tight-binding chain benchmark against exact diagonalization
- **Hubbard model**: charge and spin sector targeting in a single-band system
- **AutoMPO from TOML**: `[[interaction]]` tables, `[plugin]` sections for user-defined models, built-in presets
- **Custom geometry**: implementing a user-defined lattice traversal and registering it with `build_geometry`
- **Custom local space**: defining a new site Hilbert space outside the built-in presets
- **Custom model**: wrapping a user-defined Hamiltonian function as an Alice-compatible builder

### Packaging

- PyPI distribution renamed from `alice` to `alice-net`; `pyproject.toml` updated with wheel target and project URLs; README and logging banner updated
- **Install:** `pip install alice-net` or `uv add alice-net`; the import namespace `alice` is unchanged

### Statistics

- **45 documentation pages**; all public symbols documented
- **39 commits** since v0.1.0
- **112 files changed**, 3,161 insertions, 63 deletions

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.0; import paths, function signatures, and TOML configuration formats are unchanged
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.6

---

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
