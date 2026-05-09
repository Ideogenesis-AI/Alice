# intrcmap_1dchain

Generate an interaction map for a 1D chain.

Call `build_geometry(geo_cfg)` to construct a `Geometry`, then pass it here. The TOML keys below are the fields that belong in the `[geometry]` section.

::: alice.physics.intrcmap_1dchain
    options:
      heading_level: 2

## TOML Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lx` | int | required | Number of sites |
| `bcx` | str | `"OBC"` | Boundary condition: `"OBC"` or `"PBC"` |
| `n2x` | bool | `true` | Include nearest-neighbor bonds |

## See Also

- [intrcmap_square](intrcmap-square.md) — 2D square lattice.
- [build_geometry](build-geometry.md) — constructs the `Geometry` passed to this function when `lattice = "chain"`.
- [build_intrcmap](build-intrcmap.md) — calls this function based on `geo.lattice`.
