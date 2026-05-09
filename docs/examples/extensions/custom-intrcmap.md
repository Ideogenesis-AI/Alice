# Custom Interaction Map

Alice separates geometry construction into two stages: `build_geometry` creates a `Geometry` struct from a config dict, and `build_intrcmap` generates the list of bare `Interaction2Site` objects from that struct. You can replace the second stage — or both stages — with your own callables.

## What an Intrcmap Function Does

An `intrcmap_fn` receives a fully-constructed `Geometry` and returns a list of `Interaction2Site` objects with:

- `leading_site` and `terminal_site` filled in (0-based MPS site indices).
- `label` set to a list of strings encoding bond topology (e.g. `['NN', 'N2X']`).
- `cpl = 0.0` (coupling is assigned by the model builder, not the geometry builder).
- Tensor fields left as `None`.

Use `geo.lx`, `geo.ly`, `geo.L`, `geo.to_1d(row, col)`, and `geo.to_2d(site)` to query the lattice layout.

## Example: Custom Bond Filter on the Square Lattice

Suppose you want a square lattice interaction map that only includes bonds along x (suppressing y bonds entirely):

```python
# my_intrcmap.py
from alice import Interaction2Site
from alice.physics import Geometry

def build_xonly(geo: Geometry) -> list[Interaction2Site]:
    """NN interactions along x only (no y bonds).

    Uses the MPS ordering already stored in `geo`.
    """
    interactions = []
    for row in range(geo.ly):
        for col in range(geo.lx - 1):
            i = geo.to_1d(row, col)
            j = geo.to_1d(row, col + 1)
            s, t = min(i, j), max(i, j)
            interactions.append(Interaction2Site(
                leading_site  = s,
                terminal_site = t,
                label         = ['NN', 'N2X'],
            ))
    return interactions
```

Key rules:

- **`leading_site < terminal_site`** — always ordered so the leading site index is smaller.
- **Do not set `cpl` or any tensor fields** — those are the model builder's responsibility.
- **`label` contents are arbitrary strings** — the model builder uses them to assign couplings.

## Method 1: Pass `intrcmap_fn` Directly

```python
from alice import build_interaction, build_hamiltonian
from my_intrcmap import build_xonly

config = {
    "geometry": {"lattice": "square", "lx": 4, "ly": 3},
    "model": {"category": "bosonic", "label": "Heisenberg",
              "symmetry": "U1", "spin": 0.5, "J": 1.0},
}

interactions, spc, geo = build_interaction(config, intrcmap_fn=build_xonly)
hamiltonian = build_hamiltonian(interactions, geo.L, spc)
```

The `intrcmap_fn` keyword takes priority over any `[plugin]` section in the config.

## Method 2: TOML Plugin Spec

Specify the function in the TOML file using `"path/to/file.py:function_name"` syntax:

```toml
[model.geometry]
lattice = "square"
lx      = 4
ly      = 3

[model.model]
category = "bosonic"
label    = "Heisenberg"
symmetry = "U1"
spin     = 0.5
J        = 1.0

[model.plugin]
intrcmap = "my_intrcmap.py:build_xonly"
```

Then load normally:

```python
import tomllib
from alice import build_interaction

with open("model.toml", "rb") as f:
    cfg = tomllib.load(f)

interactions, spc, geo = build_interaction(cfg["model"])
```

Relative paths in the plugin spec are resolved relative to the TOML file's directory.

## Combining a Custom Geometry and a Custom Intrcmap

You can replace both stages at once:

```python
interactions, spc, geo = build_interaction(
    config,
    geometry_fn=build_honeycomb_geometry,
    intrcmap_fn=build_honeycomb_intrcmap,
)
```

See [Custom geometry](custom-geometry.md) for how to write `geometry_fn`.

## See Also

- [Geometry](../../api/geometry/geometry.md) — the struct passed to `intrcmap_fn`.
- [build_intrcmap API](../../api/geometry/build-intrcmap.md) — built-in dispatcher.
- [build_interaction](../../api/interaction/build-interaction.md) — where `intrcmap_fn` is wired in.
- [Custom geometry](custom-geometry.md) — replace the geometry stage instead.
