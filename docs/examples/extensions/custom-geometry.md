# Custom Geometry

Alice separates geometry construction into two stages: `build_geometry` constructs a `Geometry` struct from the `[geometry]` config dict, and `build_intrcmap` generates the list of bare `Interaction2Site` objects from that struct. You can replace either stage — or both — with your own callables.

## What a Geometry Function Does

A `geometry_fn` receives the `[geometry]` sub-dict and returns a `Geometry` instance. The `Geometry` struct carries the resolved lattice layout so that the interaction-map step can use it. Construct a `Geometry` with:

- `cfg` — the raw config dict.
- `ord_map` — a 2D list mapping `(row, col)` → MPS site index.
- `latt` — a list mapping MPS site index → `(row, col)`.

The `intrcmap_fn` downstream can then query `geo.lx`, `geo.ly`, `geo.L`, `geo.to_1d(row, col)`, and `geo.to_2d(site)` to build the interaction list.

## Example: Honeycomb Lattice

Suppose you want to define a honeycomb lattice traversed with a custom MPS ordering. Split the work into two functions:

```python
# my_geometry.py
from dataclasses import dataclass
from alice.physics import Geometry
from alice import Interaction2Site


def build_honeycomb_geometry(geo_cfg: dict) -> Geometry:
    """Build the MPS site ordering for a honeycomb lattice.

    Expected geo_cfg keys:
        lx — number of unit cells along x
        ly — number of unit cells along y
    """
    lx = geo_cfg["lx"]
    ly = geo_cfg["ly"]

    # --- define your serpentine / brickwork ordering here ---
    # ord_map[row][col] = MPS site index (0-based)
    ord_map = [[row * lx + col for col in range(lx)] for row in range(ly)]
    latt    = [(row, col) for row in range(ly) for col in range(lx)]

    return Geometry(cfg=geo_cfg, ord_map=ord_map, latt=latt)


def build_honeycomb_intrcmap(geo: Geometry) -> list[Interaction2Site]:
    """Build NN interaction map for a honeycomb lattice.

    Returns a list of Interaction2Site with labels ['NN', 'A'], ['NN', 'B'],
    ['NN', 'C'] for the three bond orientations.
    """
    interactions = []

    # --- define your nearest-neighbor bonds here ---
    # For each bond (i, j) with i < j:
    #
    # interactions.append(Interaction2Site(
    #     leading_site  = i,
    #     terminal_site = j,
    #     label         = ['NN', 'A'],   # bond type A
    # ))

    return interactions
```

Key rules:

- **`leading_site < terminal_site`** — always ordered so the leading site index is smaller.
- **Do not set `cpl` or any tensor fields** — those are the model builder's responsibility.
- **`label` contents are arbitrary strings** — the model builder uses them to assign couplings.

## Method 1: Pass Callables Directly

```python
from alice import build_interaction, build_hamiltonian
from my_geometry import build_honeycomb_geometry, build_honeycomb_intrcmap

config = {
    "geometry": {"lx": 4, "ly": 3, "bcx": "OBC"},
    "model": {"category": "bosonic", "label": "Heisenberg",
              "symmetry": "U1", "spin": 0.5, "J": 1.0},
}

interactions, spc, geo = build_interaction(
    config,
    geometry_fn=build_honeycomb_geometry,
    intrcmap_fn=build_honeycomb_intrcmap,
)
hamiltonian = build_hamiltonian(interactions, geo.L, spc)
```

Keyword arguments take priority over any `[plugin]` section in the config.

## Method 2: TOML Plugin Spec

Specify the functions in the TOML file using `"path/to/file.py:function_name"` syntax:

```toml
[honeycomb.geometry]
lx  = 4
ly  = 3
bcx = "OBC"

[honeycomb.model]
category = "bosonic"
label    = "Heisenberg"
symmetry = "U1"
spin     = 0.5
J        = 1.0

[honeycomb.plugin]
geometry = "my_geometry.py:build_honeycomb_geometry"
intrcmap = "my_geometry.py:build_honeycomb_intrcmap"
```

Then load normally:

```python
import tomllib
from alice import build_interaction

with open("honeycomb.toml", "rb") as f:
    cfg = tomllib.load(f)

interactions, spc, geo = build_interaction(cfg["honeycomb"])
```

Relative paths in the plugin spec are resolved relative to the TOML file's directory.

## Replacing Only One Stage

You can replace just the geometry stage and keep the built-in `build_intrcmap`, or vice versa. For example, if your honeycomb ordering is compatible with the standard square-lattice bond rules, you only need `geometry_fn`:

```python
interactions, spc, geo = build_interaction(
    config,
    geometry_fn=build_honeycomb_geometry,
)
```

See [Custom intrcmap](custom-intrcmap.md) for examples of replacing only the interaction-map stage.

## Handling Intermediate Sites

For long-range bonds where `terminal_site > leading_site + 1`, the model builder must also set `intermid_tnsr`. When writing a custom model for your geometry, make sure to populate `intermid_tnsr` for all such bonds. See [Custom model](custom-model.md) for how to do this.

## See Also

- [Geometry API](../../api/geometry/geometry.md) — the struct returned by `geometry_fn`.
- [build_geometry API](../../api/geometry/build-geometry.md), [build_intrcmap API](../../api/geometry/build-intrcmap.md) — built-in implementations.
- [intrcmap_1dchain](../../api/geometry/intrcmap-1dchain.md), [intrcmap_square](../../api/geometry/intrcmap-square.md) — built-in examples to follow.
- [Custom intrcmap](custom-intrcmap.md) — replace only the interaction-map stage.
- [Custom model](custom-model.md) — pair a custom geometry with a custom model.
