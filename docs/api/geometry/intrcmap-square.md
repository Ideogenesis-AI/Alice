# intrcmap_square

Generate an interaction map for a 2D square lattice.

Call `build_geometry(geo_cfg)` to construct a `Geometry`, then pass it here. The TOML keys below are the fields that belong in the `[geometry]` section.

::: alice.physics.intrcmap_square
    options:
      heading_level: 2

## TOML keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lx` | int | required | Number of columns |
| `ly` | int | required | Number of rows |
| `bcx` | str | `"OBC"` | Boundary condition along x |
| `bcy` | str | `"OBC"` | Boundary condition along y |
| `n2x` | bool | `true` | Include NN bonds along x |
| `n2y` | bool | `true` | Include NN bonds along y |
| `n3d` | bool | `false` | Include NNN diagonal bonds |
| `n3o` | bool | `false` | Include NNN off-diagonal bonds |

### Traversal order (`traverse`)

| Value | Ordering |
|-------|----------|
| `"snake"` | column-major, alternating direction (default) |
| `"zigzag"` | column-major, top→bottom every column |

## See Also

- [intrcmap_1dchain](intrcmap-1dchain.md) — 1D chain version.
- [build_geometry](build-geometry.md) — constructs the `Geometry` passed to this function when `lattice = "square"`.
- [build_intrcmap](build-intrcmap.md) — calls this function based on `geo.lattice`.
