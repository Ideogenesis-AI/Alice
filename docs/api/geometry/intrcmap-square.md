# intrcmap_square

Generate an interaction map for a 2D square lattice.

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

## See Also

- [generate_snake_order](generate-snake-order.md) — default traversal used by this function.
- [intrcmap_1dchain](intrcmap-1dchain.md) — 1D chain version.
- [build_geometry](build-geometry.md) — dispatches to this function when `lattice = "square"`.
