# build_geometry

Dispatch geometry construction from a `[geometry]` config dict.

::: alice.physics.build_geometry
    options:
      heading_level: 2

## Supported values

| `lattice` | Builder |
|-----------|---------|
| `"chain"` | `intrcmap_1dchain` |
| `"square"` | `intrcmap_square` |

| `traverse` | Generator |
|-----------|-----------|
| `"snake"` | `generate_snake_order` |

## See Also

- [intrcmap_1dchain](intrcmap-1dchain.md), [intrcmap_square](intrcmap-square.md) — concrete builders.
- [Custom geometry example](../../examples/extensions/custom-geometry.md) — how to register a new lattice type.
