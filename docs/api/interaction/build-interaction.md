# build_interaction

Build a fully populated interaction list from a TOML config or dict.

::: alice.build_interaction
    options:
      heading_level: 2

## TOML Config Structure

A minimal config dict (or TOML section) must contain two sub-tables:

```toml
[model.geometry]
lattice = "chain"
lx      = 10
bcx     = "OBC"
n2x     = true

[model.model]
category = "bosonic"
label    = "Heisenberg"
symmetry = "U1"
spin     = 0.5
J        = 1.0
```

An optional `[model.plugin]` table can override geometry, intrcmap, space, or model callables:

```toml
[model.plugin]
geometry = "my_dir/my_geometry.py:build_my_lattice"
intrcmap = "my_dir/my_intrcmap.py:build_my_intrcmap"
model    = "my_dir/my_model.py:build_my_model"
```

Plugin specs use the format `"path/to/file.py:function_name"`.

## Extension Hooks

| Kwarg | Plugin key | Signature | Description |
|-------|------------|-----------|-------------|
| `geometry_fn` | `[plugin] geometry` | `(dict) -> Geometry` | Replace the geometry-building step |
| `intrcmap_fn` | `[plugin] intrcmap` | `(Geometry) -> list[Interaction2Site]` | Replace the interaction-map step |
| `model_fn` | `[plugin] model` | `(interactions, L, **cfg) -> None` | Replace the model-filling step |
| `space_fn` | `[plugin] space` | `(category, **cfg) -> (Index, dict)` | Replace the local-space step |

Keyword arguments take priority over `[plugin]` entries.

## Return Value

`build_interaction` returns a three-tuple `(interactions, spc, geo)`:

| Element | Type | Description |
|---------|------|-------------|
| `interactions` | `list[Interaction]` | Fully populated interaction list |
| `spc` | `Index` | Physical index (local space) |
| `geo` | `Geometry` | Resolved lattice geometry; use `geo.L` as the chain length for `build_hamiltonian` |

The chain length is accessed as `geo.L`:

```python
interactions, spc, geo = build_interaction(config)
hamiltonian = build_hamiltonian(interactions, geo.L, spc)
```

## See Also

- [Interaction](interaction.md), [Interaction2Site](interaction-2site.md) — objects returned in the list.
- [Geometry](../geometry/geometry.md) — the geometry struct returned as the third element.
- [build_hamiltonian](../hamiltonian/build-hamiltonian.md) — next step after `build_interaction`.
- [AutoMPO from TOML example](../../examples/autompo-toml.md)
- [Custom geometry example](../../examples/extensions/custom-geometry.md)
- [Custom intrcmap example](../../examples/extensions/custom-intrcmap.md)
