# Geometry

The geometry module provides the `Geometry` dataclass and functions to generate interaction maps for 1D MPS traversing 1D or 2D lattices.

## Classes

| Class | Description |
|-------|-------------|
| [Geometry](geometry.md) | Resolved lattice geometry struct |

## Functions

| Function | Description |
|----------|-------------|
| [build_geometry](build-geometry.md) | Construct a `Geometry` from a config dict |
| [build_intrcmap](build-intrcmap.md) | Generate the interaction map from a `Geometry` |
| [intrcmap_1dchain](intrcmap-1dchain.md) | Interaction map for a 1D chain |
| [intrcmap_square](intrcmap-square.md) | Interaction map for a 2D square lattice |
| [intrcmap_kagome](intrcmap-kagome.md) | Interaction map for a 2D Kagome lattice |

## How It Fits in the Pipeline

The geometry stage creates `Interaction2Site` objects with `leading_site`, `terminal_site`, and `label` filled in. Coupling constants (`cpl`) are left at `0.0` and tensor fields at `None`; the model builder fills these in the next stage.

```
geo_cfg dict
    │
    ▼ build_geometry(geo_cfg)
    │
    └── Geometry  ──▶  build_intrcmap(geo)
                             │
                             └── list[Interaction2Site]  (sites + labels only)
```

## Bond Labels

Each `Interaction2Site` produced by a geometry builder carries a `label` list that encodes the bond topology. The model builder uses these labels to assign coupling constants and operator tensors to the correct interaction terms. Labels always begin with the range tag (`'NN'` for nearest-neighbor, `'NNN'` for next-nearest-neighbor) and include `'PBC'` when the bond closes a periodic boundary.

### 1D Chain

| Label | Meaning |
|-------|---------|
| `['NN', 'N2X']` | Nearest-neighbor bond |
| `['NN', 'PBC', 'N2X']` | PBC wrap-around bond |

### Square Lattice

| Label | Meaning |
|-------|---------|
| `['NN', 'N2X']` | Nearest-neighbor along x |
| `['NN', 'N2Y']` | Nearest-neighbor along y |
| `['NN', 'PBC', 'N2X']` | PBC bond along x |
| `['NN', 'PBC', 'N2Y']` | PBC bond along y |
| `['NNN', 'N3D']` | Next-nearest-neighbor diagonal |
| `['NNN', 'N3O']` | Next-nearest-neighbor off-diagonal |

### Kagome Lattice

| Label | Meaning |
|-------|---------|
| `['NN', 'N2U']` | Upward-triangle NN bond (A–B, A–C, B–C within a unit cell) |
| `['NN', 'N2D']` | Downward-triangle NN bond (between adjacent unit cells) |
| `['NN', 'N2U', 'PBC']` | N2U bond closing a periodic boundary |
| `['NN', 'N2D', 'PBC']` | N2D bond closing a periodic boundary |

## See Also

- [Interaction2Site](../interaction/interaction-2site.md) — objects produced by geometry builders.
- [Custom geometry example](../../examples/extensions/custom-geometry.md)
- [Custom intrcmap example](../../examples/extensions/custom-intrcmap.md)
