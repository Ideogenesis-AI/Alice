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

The **geometry stage** of the AutoMPO pipeline creates the list of `Interaction2Site` objects with `leading_site`, `terminal_site`, and `label` fields filled in, but all tensor fields set to `None` and all coupling constants left at `0.0`.

Alice provides built-in builders for:

- `intrcmap_1dchain` — a simple nearest-neighbor 1D chain.
- `intrcmap_square` — a 2D square lattice traversed in snake order.
- `build_geometry` — the TOML dispatcher that calls the right builder based on the `lattice` key.

Custom geometries can be registered by passing a `geometry_fn` to `build_interaction`, or by pointing to an external Python file via the `[plugin]` section of a TOML config.

## AutoMPO Pipeline

The full pipeline from a TOML config to a Hamiltonian MPO has three stages:

```
TOML config
    │
    ▼
 build_interaction(config)
    │
    ├── Stage 1: Geometry  →  list[Interaction2Site]  (sites + labels)
    │
    ├── Stage 2: Model     →  fills cpl + tensors on each Interaction
    │
    └── returns (interactions, spc, L)
                │
                ▼
         build_hamiltonian(interactions, L, spc)
                │
                └── returns MPO
```

`build_interaction` accepts either a path to a TOML file or a pre-loaded config dict. `build_hamiltonian` assembles the MPO term by term using Nicole's `oplus`, then compresses it with SVD sweeps.

## Measurement

Once you have an MPS state and an MPO observable, the expectation value \(\langle\psi|O|\psi\rangle\) is computed with:

```python
energy = alice.observe(mps, hamiltonian_mpo)
```

`observe` dispatches to an efficient left-to-right transfer-matrix sweep.
