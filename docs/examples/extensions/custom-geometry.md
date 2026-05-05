# Custom Geometry

Alice dispatches geometry construction via `build_geometry`, which reads the `lattice` key from a `[geometry]` config dict. You can plug in your own geometry function in two ways: by passing `geometry_fn` directly to `build_interaction`, or by specifying a plugin file path in the TOML config.

## What a geometry function does

A geometry function receives the `[geometry]` sub-dict and returns a list of `Interaction2Site` objects with:

- `leading_site` and `terminal_site` filled in (0-based site indices).
- `label` set to a list of strings encoding bond topology (e.g. `['NN', 'N2X']`).
- `cpl = 0.0` (coupling is assigned by the model builder, not the geometry builder).
- Tensor fields left as `None`.

## Example: honeycomb lattice

Suppose you want to define a honeycomb lattice traversed with a custom MPS ordering. Here is the structure of a geometry function:

```python
# my_geometry.py
from alice import Interaction2Site

def build_honeycomb(geo: dict) -> list[Interaction2Site]:
    """Build NN interaction map for a honeycomb lattice.

    Expected geo keys:
        lx     — number of unit cells along x
        ly     — number of unit cells along y
        bcx    — 'OBC' or 'PBC'

    Returns a list of Interaction2Site with labels ['NN', 'A'], ['NN', 'B'],
    ['NN', 'C'] for the three bond orientations.
    """
    lx  = geo["lx"]
    ly  = geo["ly"]
    bcx = geo.get("bcx", "OBC").upper()

    interactions = []

    # --- define your MPS site ordering and bonds here ---
    # For each nearest-neighbor bond (i, j) with i < j:
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

## Method 1: Pass `geometry_fn` directly

```python
from alice import build_interaction, build_hamiltonian
from my_geometry import build_honeycomb

config = {
    "geometry": {"lx": 4, "ly": 3, "bcx": "OBC"},
    "model": {"category": "bosonic", "label": "Heisenberg",
              "symmetry": "U1", "spin": 0.5, "J": 1.0},
}

interactions, spc, L = build_interaction(config, geometry_fn=build_honeycomb)
hamiltonian = build_hamiltonian(interactions, L, spc)
```

The `geometry_fn` keyword takes priority over any `[plugin]` section in the config.

## Method 2: TOML plugin spec

Specify the function in the TOML file using `"path/to/file.py:function_name"` syntax:

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
geometry = "my_geometry.py:build_honeycomb"
```

Then load normally:

```python
import tomllib
from alice import build_interaction

with open("honeycomb.toml", "rb") as f:
    cfg = tomllib.load(f)

interactions, spc, L = build_interaction(cfg["honeycomb"])
```

Relative paths in the plugin spec are resolved relative to the TOML file's directory.

## Handling intermediate sites

For long-range bonds where `terminal_site > leading_site + 1`, the model builder must also set `intermid_tnsr`. When writing a custom model for your geometry, make sure to populate `intermid_tnsr` for all such bonds. See [Custom model](custom-model.md) for how to do this.

## See Also

- [build_geometry API](../../api/geometry/build-geometry.md)
- [intrcmap_1dchain](../../api/geometry/intrcmap-1dchain.md), [intrcmap_square](../../api/geometry/intrcmap-square.md) — built-in examples to follow.
- [Custom model](custom-model.md) — pair a custom geometry with a custom model.
