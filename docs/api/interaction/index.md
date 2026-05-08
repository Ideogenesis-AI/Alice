# Interaction

The `alice.network.interaction` module defines the dataclasses that represent Hamiltonian interaction terms, and `build_interaction`, the TOML-driven dispatcher that orchestrates the full AutoMPO pipeline.

## Dataclasses

| Class | Description |
|-------|-------------|
| [Interaction](interaction.md) | Base interaction (coupling + label) |
| [Interaction1Site](interaction-1site.md) | On-site term |
| [Interaction2Site](interaction-2site.md) | Two-site term |

## Builder

| Function | Description |
|----------|-------------|
| [build_interaction](build-interaction.md) | Build interaction list from TOML config |

## Pipeline overview

```
config (TOML or dict)
    │
    ▼ build_interaction()
    │
    ├─ Stage 1: geometry_fn(geo_cfg)  →  Geometry
    │
    ├─ Stage 2: intrcmap_fn(geo)      →  list[Interaction2Site]  (sites + labels, no tensors, cpl=0.0)
    │
    ├─ Stage 3: model_fn(interactions, L, **model_cfg)
    │       → fills cpl + tensor fields in place
    │
    └─ returns (interactions, spc, geo)
```

## See Also

- [Geometry](../geometry/index.md) — geometry builders that produce the bare interaction list.
- [Hamiltonian](../hamiltonian/build-hamiltonian.md) — assembles the MPO from the populated list.
- [AutoMPO from TOML example](../../examples/autompo-toml.md)
