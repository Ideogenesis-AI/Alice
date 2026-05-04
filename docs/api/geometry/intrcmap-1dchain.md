# intrcmap_1dchain

Generate an interaction map for a 1D chain.

::: alice.physics.intrcmap_1dchain
    options:
      heading_level: 2

## TOML keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lx` | int | required | Number of sites |
| `bcx` | str | `"OBC"` | Boundary condition: `"OBC"` or `"PBC"` |
| `n2x` | bool | `true` | Include nearest-neighbor bonds |

## See Also

- [intrcmap_square](intrcmap-square.md) — 2D square lattice.
- [build_geometry](build-geometry.md) — dispatches to this function when `lattice = "chain"`.
