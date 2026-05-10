# intrcmap_kagome

Generate an interaction map for a 2D Kagome lattice.

Call `build_geometry(geo_cfg)` to construct a `Geometry`, then pass it here. The TOML keys below are the fields that belong in the `[geometry]` section.

::: alice.physics.intrcmap_kagome
    options:
      heading_level: 2

## TOML Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lx` | int | required | Number of unit-cell columns |
| `ly` | int | required | Number of unit-cell rows |
| `bcx` | str | `"OBC"` | Boundary condition along x |
| `bcy` | str | `"OBC"` | Boundary condition along y |
| `n2u` | bool | `true` | Include N2U (upward-triangle) bonds |
| `n2d` | bool | `true` | Include N2D (downward-triangle) bonds |

The total number of MPS sites is `lx * ly * 3` (three sublattice sites A, B, C per unit cell).

### Traversal Order (`traverse`)

| Value | Ordering |
|-------|----------|
| `"serpentine"` | column-major, alternating direction (default) |
| `"sequential"` | column-major, same direction every column |

### Bond Labels

| Label | Bond family |
|-------|-------------|
| `['NN', 'N2U']` | Upward-triangle NN bond (A–B, A–C, B–C within a unit cell) |
| `['NN', 'N2D']` | Downward-triangle NN bond (between adjacent unit cells) |
| `['NN', 'N2U', 'PBC']` | N2U bond closing a periodic boundary |
| `['NN', 'N2D', 'PBC']` | N2D bond closing a periodic boundary |

## See Also

- [intrcmap_square](intrcmap-square.md) — 2D square lattice version.
- [intrcmap_1dchain](intrcmap-1dchain.md) — 1D chain version.
- [build_geometry](build-geometry.md) — constructs the `Geometry` passed to this function when `lattice = "kagome"`.
- [build_intrcmap](build-intrcmap.md) — calls this function based on `geo.lattice`.
