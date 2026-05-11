# build_geometry

Construct a `Geometry` from a `[geometry]` config dict.

::: alice.physics.build_geometry
    options:
      heading_level: 2

## Supported Values

| `lattice` | Builder |
|-----------|---------|
| `"chain"` | `intrcmap_1dchain` |
| `"square"` | `intrcmap_square` |
| `"kagome"` | `intrcmap_kagome` |

## See Also

- [Geometry](geometry.md) — the struct returned by this function.
- [build_intrcmap](build-intrcmap.md) — the next step; generates `list[Interaction2Site]` from the `Geometry`.
- Built-in interaction-map builders:
    - [intrcmap_1dchain](intrcmap-1dchain.md) — 1D chain.
    - [intrcmap_square](intrcmap-square.md) — 2D square lattice.
    - [intrcmap_kagome](intrcmap-kagome.md) — 2D Kagome lattice.
- [Custom geometry example](../../examples/extensions/custom-geometry.md) — how to replace this function entirely.
