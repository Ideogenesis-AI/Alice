# build_intrcmap

Generate a bare interaction map from a `Geometry` instance.

::: alice.physics.build_intrcmap
    options:
      heading_level: 2

## See Also

- [Geometry](geometry.md) — the input struct; construct it with `build_geometry`.
- [build_geometry](build-geometry.md) — the preceding step that produces the `Geometry`.
- Built-in interaction-map builders called internally:
    - [intrcmap_1dchain](intrcmap-1dchain.md) — 1D chain.
    - [intrcmap_square](intrcmap-square.md) — 2D square lattice.
    - [intrcmap_kagome](intrcmap-kagome.md) — 2D Kagome lattice.
- [Custom intrcmap example](../../examples/extensions/custom-intrcmap.md) — how to replace this function with a user-defined builder.
