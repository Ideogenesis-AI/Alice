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

## See Also

- [Geometry](geometry.md) — the struct returned by this function.
- [build_intrcmap](build-intrcmap.md) — the next step; generates `list[Interaction2Site]` from the `Geometry`.
- [Custom geometry example](../../examples/extensions/custom-geometry.md) — how to replace this function entirely.
