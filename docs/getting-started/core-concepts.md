# Core Concepts

This page introduces the principal abstractions in Alice. Understanding these concepts will help you navigate the API and write correct, efficient code.

## Physical Index and Local Space

Every site on the lattice has a **local Hilbert space** — the set of quantum states available at that site. In Alice, this is represented as a Nicole `Index` object, which encodes the symmetry quantum numbers (charges) and their dimensions.

Alice provides three functions to construct a local space:

- `build_bosonic` — a spin-`s` site (Hilbert space of a spin-1/2, spin-1, etc.).
- `build_fermionic` — a spinless fermion site (vacuum and occupied).
- `build_conductor` — a spinful fermion site (four states: empty, spin-up, spin-down, doubly occupied).

Each function returns `(spc, Op)` where `spc` is the physical `Index` and `Op` is a dictionary of pre-built MPO operator templates for that space.

## Matrix Product State (MPS)

A **Matrix Product State** is a variational ansatz for a 1D quantum state. It represents the wavefunction \(|\psi\rangle\) as a chain of tensors, one per site:

\[
|\psi\rangle = \sum_{\sigma_0, \ldots, \sigma_{L-1}} A^{\sigma_0}_{a_0 a_1} A^{\sigma_1}_{a_1 a_2} \cdots A^{\sigma_{L-1}}_{a_{L-2} a_{L-1}} |\sigma_0 \cdots \sigma_{L-1}\rangle
\]

The matrices \(A^{\sigma_i}_{a_i a_{i+1}}\) are block-sparse due to symmetry. The **bond dimension** \(\chi\) (the maximum size of the virtual indices \(a_i\)) controls the expressibility of the ansatz and the computational cost.

In Alice, an MPS is created by wrapping a list of Nicole `Tensor` objects in the `MPS` class. Each tensor has exactly three axes:

- Axis 0: left bond (virtual index \(a_i\))
- Axis 1: right bond (virtual index \(a_{i+1}\))
- Axis 2: physical index (\(\sigma_i\))

**Canonical form.** The MPS can be brought into mixed-canonical form around a single site (the *orthogonality center*) by calling `mps.canonical(target)`. In this form the norm of the state equals the Frobenius norm of the center tensor.

## Matrix Product Operator (MPO)

A **Matrix Product Operator** represents a quantum operator (typically a Hamiltonian) as a chain of tensors. Each MPO tensor has four axes:

- Axis 0: left bond
- Axis 1: right bond
- Axis 2: physical ket index
- Axis 3: physical bra index

The MPO class in Alice extends the same `Network` base class as `MPS`. Hamiltonians are constructed as MPOs via the [AutoMPO pipeline](#autompo-pipeline) described below.

## Network

`Network` is the base class shared by `MPS` and `MPO`. It stores the list of site tensors, exposes `.canonical()`, `.norm()`, `.normalize()`, `.serialize()`, and `.deserialize()`, and enforces bond-consistency invariants on construction.

## Interaction

An **Interaction** describes a single term in the Hamiltonian. Alice uses three dataclasses:

- `Interaction` — base class carrying a `cpl` (coupling constant) and a `label` list (bond topology tags).
- `Interaction1Site` — on-site term: adds `site` and `tnsr` fields.
- `Interaction2Site` — two-site term: adds `leading_site`, `terminal_site`, `leading_tnsr`, `terminal_tnsr`, and `intermid_tnsr` fields.

A list of `Interaction` objects forms the complete description of a Hamiltonian. The AutoMPO pipeline populates these objects in two stages (geometry, then model) and passes the result to `build_hamiltonian`.

## Geometry

The **geometry stage** of the AutoMPO pipeline is split into two steps. First, `build_geometry` resolves the lattice configuration into a `Geometry` struct. Second, `build_intrcmap` uses that struct to create the list of bare `Interaction2Site` objects.

The `Geometry` dataclass stores:

- `cfg` — the raw config dict from `[geometry]`.
- `ord_map` — a `dict[tuple[int, ...], int]` mapping a lattice coordinate tuple → MPS site index.
- `latt` — a `list[tuple[int, ...]]` mapping MPS site index → lattice coordinate tuple.

It also exposes convenience properties (`lx`, `ly`, `L`, `lattice`, `traverse`) and coordinate-conversion methods (`to_1d(coord)` and `to_2d(site)`).

Alice provides built-in builders for:

- `intrcmap_1dchain` — a simple nearest-neighbor 1D chain.
- `intrcmap_square` — a 2D square lattice with configurable traversal order.
- `intrcmap_kagome` — a 2D Kagome lattice with configurable traversal order.
- `build_geometry` — constructs a `Geometry` from a config dict.
- `build_intrcmap` — generates the interaction list from a `Geometry`.

Custom geometries can be plugged in by passing `geometry_fn` and/or `intrcmap_fn` to `build_interaction`, or by pointing to external Python files via the `[plugin]` section of a TOML config.

## AutoMPO Pipeline

The full pipeline from a TOML config to a Hamiltonian MPO has four stages:

```
TOML config
    │
    ▼
 build_interaction(config)
    │
    ├── Stage 1: Geometry    →  Geometry (cfg, ord_map, latt, lx, ly, L, …)
    │
    ├── Stage 2: Intrcmap    →  list[Interaction2Site]  (sites + labels)
    │
    ├── Stage 3: Model       →  fills cpl + tensors on each Interaction
    │
    └── returns (interactions, spc, geo)
                │
                ▼
         build_hamiltonian(interactions, geo.L, spc)
                │
                └── returns MPO
```

`build_interaction` accepts either a path to a TOML file or a pre-loaded config dict. You can replace any stage by passing `geometry_fn`, `intrcmap_fn`, `model_fn`, or `space_fn` kwargs, or via `[plugin]` entries in the TOML config. `build_hamiltonian` assembles the MPO term by term using Nicole's `oplus`, then compresses it with SVD sweeps.

## Measurement

Once you have an MPS state and an MPO observable, the expectation value \(\langle\psi|O|\psi\rangle\) is computed with:

```python
energy = alice.observe(mps, hamiltonian_mpo)
```

`observe` dispatches to an efficient left-to-right transfer-matrix sweep.
