# Changelog

## [Unreleased]

**Two-Site BUG Time Integrator**

Adds `alice.algorithm.two_site_bug`, the faithful rank-adaptive two-site BUG
(Basis-Update & Galerkin) integrator of Ceruti, Kusch & Lubich
([arXiv:2304.05660](https://arxiv.org/abs/2304.05660)) for real- and
imaginary-time evolution of an MPS under a nearest-neighbour Hamiltonian. The
Alice-facing driver is built on the existing Alice/Nicole stack — `MPS`, the
AutoMPO interaction list, and the PyTorch backend; the symmetry-aware faithful-KLS
local kernel is vendored, Nicole-native, in a private `_kernel` subpackage.

### `alice.algorithm.two_site_bug`

- **`run(mps, interactions, opts)`** evolves the state with commuting even/odd
  Trotter sweeps of *local* K/L/S bond updates: each update augments the left and
  right frames from the evolved K and L factors, evolves the small core in the
  augmented bases (Galerkin), and truncates with an SVD so the bond dimension
  adapts (the basis augmentation). The local substeps exponentiate the projected
  effective Hamiltonian internally (Krylov `expv`) — exact at full rank. Supports
  first-order (`'lie'`) and symmetric second-order (`'strang'`) steps and
  imaginary-time cooling.
- **Bond Hamiltonians** are reused from the AutoMPO interaction list: the leading
  and terminal MPO tensors of each nearest-neighbour `Interaction2Site` are
  contracted over their operator channel to form the bare two-site term fed to the
  KLS kernel. The kernel is symmetry-aware (works with the U(1) charge sectors of
  the MPS).
- **`Options`** (TOML-loadable) and **`Summary`** mirror the DMRG interface. The
  summary records, per step, the kept bond dimension and the *proposed* augmented
  dimension, so the rank growth and the discarded augmentation are both visible.
- Validated against exact diagonalization (state fidelity, exact norm
  conservation, U(1) charge conservation, and second-order Trotter convergence).

## [0.1.6] - 2026-06-10

**MPS Initialization for Odd Chains**

Extends `init_mps` with a `target_qn` parameter for explicit quantum-number targeting,
fixes a silent zero-MPS bug in random initialization for odd-L chains, and replaces
`warnings.warn` with structured `logging` throughout `automps`. No breaking changes.

### `init_mps`: `target_qn` Parameter

- New `target_qn` keyword on `init_mps`: the desired right-boundary charge `Q[L]` (total
  quantum number of the chain). Defaults to `Q_vac` (half-filling), preserving behavior
  for even-L even-filling cases.
- When `target_qn` is given and the auto-config cannot reach it (e.g. `target_qn=0` for
  odd-L spin-½), `init_mps` raises a `ValueError` in both modes rather than silently
  producing an MPS in the wrong sector.

### Odd-L Random MPS Fix

- **Bug fixed:** `_random_mps` previously pinned both boundary indices to `Op['vac']`
  (charge `Q_vac`). For odd L, where no config can return to `Q_vac`, every charge block
  of the last tensor was forbidden and the MPS canonicalized to zero.
- The right boundary is now constructed with `Q[L]` (the actual charge reached by the
  auto-config path) when `target_qn` was not given explicitly. For even L this is still
  `Q_vac`; for odd L it is the correct non-vacuum charge.

### Warning System Upgrade

- `init_mps` no longer calls `warnings.warn(UserWarning)` for odd-L chains where
  `Q_vac` is unreachable. The message is now emitted via `logging.getLogger(__name__)`
  and recommends passing `target_qn` explicitly.

### `_auto_config` Generalization

- `_auto_config` now accepts `target_qn` and all three internal strategies (single-sector
  fill, period-2 alternation, greedy fallback) target `target_qn` instead of `Q_vac`.
- The function is now a pure helper with no side effects; the sole warning site is
  `init_mps`.

### Documentation

- `init-mps.md` gains a **Logic Overview** decision diagram, an **Odd-chain lengths**
  section with a spin-½ example, and a corrected **Bond Sectors in Random Mode** table
  note (BFS is seeded from `Q_c = Q[L//2]`, not from `Q_vac`).

### Statistics

- **814 tests** across 23 test modules (up from ~795 / 23 modules in v0.1.5).
- **5 commits** since v0.1.5.
- **4 files changed**, 423 insertions, 89 deletions.
- **22 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.5.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.1.5] - 2026-06-08

**Cache Isolation and Thermal MPO**

Fixes DMRG environment-block caching for concurrent runs and corrects the early-stopping
criterion in `thermal_mpo` to account for the operator norm of `H`. No breaking changes.

### DMRG Environment Cache Isolation

- `dmrg.run()` now creates a unique subdirectory (first 8 hex characters of a UUID4,
  e.g. `{env_cache_dir}/a1b2c3d4/`) inside `env_cache_dir` per invocation. Previously,
  concurrent runs sharing the same `env_cache_dir` wrote to the same `left/` and
  `right/` paths and could corrupt each other's cached blocks.
- The unique subdirectory is removed automatically in a `finally` block on return or
  exception, leaving no stale files behind.
- `Options.env_cache_dir` docstring updated to document the subdirectory scheme and
  automatic cleanup.

### `thermal_mpo` Early-Stopping Correction

- The early-stopping guard now tests `|β^n / n!| × ‖H^n‖_F < coeff_thresh` rather
  than the bare coefficient `|β^n / n!|` alone. For Hamiltonians with large operator
  norm, `‖H^n‖_F` can be much larger than 1, causing the old criterion to exit
  prematurely before the series had converged.
- The conditional that skipped the power-update step when the next coefficient was small
  has been removed; `H_pow` is now always advanced when `n < order`.
- `order` and `coeff_thresh` parameter docstrings updated to describe the corrected
  criterion.

### Statistics

- **~795 tests** across 23 test modules (up from ~740 / 23 modules in v0.1.4).
- **3 commits** since v0.1.4.
- **4 files changed**, 141 insertions, 19 deletions.
- **22 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.4.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.1.4] - 2026-05-31

**Thermal Density Matrix**

Introduces `NormalMPO`, an `MPO` subclass with a separately tracked scale factor, and
`thermal_mpo`, which approximates `ρ(β) = exp(−βH)` via a truncated Taylor series in MPO
arithmetic. The `observe` function is extended to accept `NormalMPO` as the state
argument, enabling finite-temperature expectation values within the existing workflow.
No breaking changes.

### `NormalMPO`

- New `NormalMPO` class in `alice.network.thermal`: represents an operator as
  `scale × mpo_unit`, keeping the internal MPO at unit Frobenius norm and carrying the
  physical magnitude in a separate `_scale` attribute.
- `*` (scalar multiply): only `_scale` is updated; site tensors are not modified.
  `@` (MPO product) and `+` (MPO sum) produce new site tensors with `_scale` set to
  the combined physical magnitude. `compact()` folds the extracted SVD norm into
  `_scale` after each compression sweep and restores standard bond-arrow directions
  via `capcup`.
- Class method `NormalMPO.from_mpo(mpo)` constructs a `NormalMPO` from any plain `MPO`.
- Exported from `alice.network` and the `alice` top-level namespace.

### `thermal_mpo`

- New `thermal_mpo(H, beta, order, spc)` function: computes
  `ρ(β) ≈ Σ (−β)^n/n! · H^n` via MPO arithmetic, calling `compact()` after every
  addition and power step to control bond growth. Early exit when the Taylor coefficient
  drops below `coeff_thresh` (default `1e-15`).
- The returned `NormalMPO` carries `_scale ≈ Tr[ρ]` (the partition function, up to the
  MPO norm).
- Exported from `alice.network` and the `alice` top-level namespace.

### `observe` Updated for Thermal States

- `observe` now accepts a `NormalMPO` as the state argument: forms the MPO product
  `ρ @ O`, compresses it with `compact()`, and returns `Tr[ρ O] / Tr[ρ]` — the
  normalized thermal expectation value.

### Documentation

- New API reference pages for `NormalMPO` and `thermal_mpo`; `alice.network` index and
  `observe` reference updated to document the new types.

### Statistics

- **~740 tests** across 23 test modules (up from ~700 / 22 modules in v0.1.3).
- **13 commits** since v0.1.3.
- **9 files changed**, 1,192 insertions, 10 deletions.
- **22 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.3.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.1.3] - 2026-05-13

**MPS Init and DMRG Checkpointing**

Introduces `init_mps`, a universal MPS initializer that replaces the per-example
`_random_mps` helpers with a single particle-type–agnostic function. Adds atomic
per-sweep checkpointing to DMRG via `dmrg.Options.checkpoint_dir`, and exposes
`Network.bond_states` for counting physical states at each bond in non-Abelian
simulations. No breaking changes.

### Universal MPS Initializer

- New `alice.network.automps` module with `init_mps`: constructs an initial MPS for
  DMRG from any `(Spc, Op)` pair returned by `load_space`. Works for bosonic,
  fermionic, and conductor sites without a `particle_type=` argument.
- `bond_dim=1`: deterministic product state with exact charge targeting, one sector per
  bond; best paired with CBE (`scheme='1sp'`) or 2-site (`scheme='2s'`) DMRG.
- `bond_dim>1`: random MPS with group-derived bond sectors; reachable charges are
  selected by BFS from the center-bond charge to depth 2.
- Auto-balanced `config=None` heuristic covers all standard even-*L* half-filled cases:
  alternating high/low for 2-sector spaces, single neutral sector for 3-sector spaces,
  alternating neutral-pair for 4-sector spaces, SU(2) dimer path for pure-SU(2) spaces.
- Exported from `alice` top-level namespace and from `alice.network`.
- DMRG example scripts (`dmrg_heisenberg`, `dmrg_freefermion`, `dmrg_conductor`)
  updated to accept an `init` parameter (`'iter_diag'` or `'random'`) using `init_mps`;
  per-example `_random_mps` helpers removed.

### DMRG Checkpointing

- New `checkpoint_dir` option in `dmrg.Options`: after every completed sweep the
  current state is serialized to `dmrg.ckpt` via an atomic write (write to
  `dmrg_lock.ckpt`, then rename). On POSIX systems the rename is atomic, so a crash
  during serialization cannot corrupt the previous checkpoint.
- Checkpoint file is in PyTorch format, loadable via `dmrg.Summary.load`.
- Defaults to the current working directory at `run()` call time (matching `.logging`).

### `Network.bond_states`

- New `bond_states` property on `Network`: returns the number of physical states per
  internal bond (length `L - 1`). Equal to `bond_dims` for Abelian groups; larger for
  SU(2) due to multiplet degeneracy (*2j+1* states per multiplet of spin *j*).

### Documentation

- New API reference page for `init_mps` with parameter table, charge-convention notes,
  and runnable examples; `Network` reference updated with `bond_states`.
- Quick-start guide updated to use `init_mps` and the `Summary` checkpoint interface.

### Statistics

- **~700 tests** across 22 test modules (up from ~650 / 21 modules in v0.1.2).
- **16 commits** since v0.1.2.
- **17 files changed**, 1,319 insertions, 187 deletions.
- **21 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.2.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.6.

---

## [0.1.2] - 2026-05-11

**Geometry Expansion and Refactor**

Introduces Kagome lattice support, a full refactor of the geometry subsystem around a
`Geometry` dataclass, a unified traversal-order naming scheme, and ASCII text diagrams
for MPS and MPO chains. Also adds `Sp4`/`Sm4` operator templates to `build_bosonic` and
`build_conductor`. Several breaking changes to the geometry and `build_interaction` APIs;
the DMRG, Hamiltonian, and algorithm APIs are unchanged.

### Kagome Lattice

- New `alice.physics.kagome` module with `intrcmap_kagome`: nearest-neighbor bond
  generation for Kagome lattices, covering N2U (upward-triangle: A–B, A–C, B–C within
  each unit cell) and N2D (downward-triangle: bonds between adjacent unit cells), with
  OBC/PBC boundary conditions along both axes.
- Traversal orders `'sequential'` (column-major, default) and `'serpentine'`
  (column-major with alternating row direction) for Kagome lattices.
- `intrcmap_kagome` is exported from `alice.physics` alongside `intrcmap_1dchain`
  and `intrcmap_square`.

### Geometry Module Refactor

- New `Geometry` dataclass returned by `build_geometry`; carries `cfg`, `ord_map`,
  and `latt` as a single typed object with derived properties `lattice`, `traverse`,
  `lx`, `ly`, `L`, and helpers `to_1d`/`to_2d`.
- All `intrcmap_*` functions now take a `Geometry` as input (previously a plain dict)
  and continue to return `List[Interaction2Site]`.
- `build_interaction` now returns `(interactions, spc, geo)` — the third element
  changed from an `int` (site count) to the `Geometry` instance.
- New `build_intrcmap(geo)` dispatcher: reconstructs the bond list from any `Geometry`
  instance, decoupling lattice construction from bond enumeration.
- New `build_traversal` helpers in all three lattice modules (`chain.py`, `square.py`,
  `kagome.py`); 1D chain and square lattice geometries separated into their own modules.
- `ord_map` keys changed from integer flat indices to coordinate tuples: `(row, col)`
  for chain and square, `(row, col, u)` for Kagome (where `u` is the sublattice index
  0=A, 1=B, 2=C), making coordinate lookups explicit for all lattice types.

### Traversal Order Naming

- `'snake'` is renamed to `'serpentine'` (same behavior: columns alternate
  direction, even columns top→bottom, odd columns bottom → top).
- `'sequential'` is a new order (all columns top → bottom, no reversal) and is
  now the default for square and Kagome lattices.
- All documentation, example configs, and tests updated to the new naming.

### MPS/MPO Text Display

- New `alice.network.display` module with `network_summary`: Unicode ASCII-art chain
  diagrams showing tensor nodes, bond dimensions, center site, and summary info.
- `MPS.__repr__` and `MPO.__repr__` delegate to `network_summary`, giving readable
  representations in REPLs and notebooks.

### New Operator Templates

- `build_bosonic` and `build_conductor` gain `Sp4`/`Sp4dag`/`Sm4`/`Sm4dag` templates
  (U(1)-only), enabling AutoMPO construction for models where raising and lowering
  channels must be handled separately.

### Documentation

- New API reference pages for `intrcmap_kagome` and `build_intrcmap`; geometry and
  traversal pages updated to reflect the refactored two-stage construction pipeline.
- All documentation page headings standardized to title case.

### Statistics

- **~650 tests** across 21 test modules (up from 538 / 19 modules in v0.1.0).
- **119 commits** since v0.1.1.
- **65 files changed**, 3,380 insertions, 824 deletions.
- **20 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:**
    - `build_geometry` return type changed from `List[Interaction2Site]` to `Geometry`.
    - All `intrcmap_*` functions now take a `Geometry` as input instead of a plain dict;
      return type `List[Interaction2Site]` is unchanged.
    - `build_interaction` third return value changed from `int` (site count) to `Geometry`.
    - `geometry_fn` callable override signature changed from
      `(geo: dict) -> List[Interaction2Site]` to `(geo_cfg: dict) -> Geometry`; a new
      `intrcmap_fn` override was added with signature
      `(geo: Geometry) -> List[Interaction2Site]`.
    - The public function `generate_snake_order` is removed; its traversal logic is now
      internal.
    - The TOML key `traverse: 'snake'` is deprecated; `'snake'` was renamed to
      `'serpentine'`, and `'sequential'` is the new default order.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.6.

---

## [0.1.1] - 2026-05-06

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

## [0.1.0] - 2026-05-04

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

[0.1.6]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.6
[0.1.5]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.5
[0.1.4]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.4
[0.1.3]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.3
[0.1.2]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.2
[0.1.1]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.1
[0.1.0]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.0
