# Geometry

The geometry module provides functions to generate interaction maps for 1D MPS traversing 1D or 2D lattices.

## Functions

| Function | Description |
|----------|-------------|
| [intrcmap_1dchain](intrcmap-1dchain.md) | Interaction map for a 1D chain |
| [intrcmap_square](intrcmap-square.md) | Interaction map for a 2D square lattice |
| [build_geometry](build-geometry.md) | TOML dispatcher for geometry builders |

## How it fits in the pipeline

The geometry stage creates `Interaction2Site` objects with `leading_site`, `terminal_site`, and `label` filled in. Coupling constants (`cpl`) are left at `0.0` and tensor fields at `None`; the model builder fills these in the second stage.

```
geo_cfg dict
    │
    ▼ build_geometry(geo_cfg)  — dispatches by lattice + traverse
    │
    └── list[Interaction2Site]  (sites + labels only)
```

## Bond labels

| Label | Meaning |
|-------|---------|
| `['NN', 'N2X']` | Nearest-neighbor along x |
| `['NN', 'N2Y']` | Nearest-neighbor along y |
| `['NN', 'PBC', 'N2X']` | PBC bond along x |
| `['NN', 'PBC', 'N2Y']` | PBC bond along y |
| `['NNN', 'N3D']` | Next-nearest-neighbor diagonal |
| `['NNN', 'N3O']` | Next-nearest-neighbor off-diagonal |

## See Also

- [Interaction2Site](../interaction/interaction-2site.md) — objects produced by geometry builders.
- [Custom geometry example](../../examples/extensions/custom-geometry.md)
