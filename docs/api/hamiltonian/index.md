# Hamiltonian

This section covers the final stage of the AutoMPO pipeline: assembling an MPO from a populated interaction list, and the three built-in model builders.

## Functions

| Function | Description |
|----------|-------------|
| [build_hamiltonian](build-hamiltonian.md) | Assemble Hamiltonian MPO from interactions |
| [build_heisenberg](build-heisenberg.md) | Heisenberg spin model |
| [build_free_fermion](build-free-fermion.md) | Spinless free-fermion (tight-binding) model |
| [build_hubbard](build-hubbard.md) | Hubbard model |

## Pipeline position

```
(interactions, spc, L)    ← from build_interaction()
            │
            ▼ build_hamiltonian(interactions, L, spc)
            │
            └── MPO
```

`build_hamiltonian` accumulates each interaction as a rank-1 MPO update via Nicole's `oplus`, then compresses the result. The built-in model builders (`build_heisenberg`, etc.) are called internally by `build_interaction`, but can also be invoked directly for custom pipelines.

## See Also

- [Interaction](../interaction/index.md) — input to `build_hamiltonian`.
- [MPO](../network/mpo.md) — output type.
- [AutoMPO from TOML example](../../examples/autompo-toml.md)
