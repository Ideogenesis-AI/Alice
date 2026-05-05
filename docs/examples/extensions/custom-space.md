# Custom Local Space

The three built-in local space builders (`build_bosonic`, `build_fermionic`, `build_conductor`) cover spin, spinless-fermion, and spinful-fermion sites. For other physical systems — bosons with a finite truncation, mixed-valence sites, or entirely custom degrees of freedom — you can provide your own space builder.

## What a space builder does

A space builder calls Nicole's `load_space` (or constructs an `Index` directly) to obtain:

- `spc` — a Nicole `Index` encoding the local Hilbert space (physical quantum numbers and their dimensions).
- `Op` — a dict of named Nicole `Tensor` objects that serve as MPO operator templates.

The operator templates your model builder will use must be pre-built here. The tensor axis conventions are described in the [Local Space API overview](../../api/local-space/index.md).

## Example: spin-1 site with U(1)

The built-in `build_bosonic` already handles this (`spin=1.0`), but here is how you would replicate it from scratch to understand the pattern:

```python
from alice.physics.system import (
    _make_i4, _make_spin_templates,   # internal helpers (not public API)
)
from nicole import Direction, load_space, oplus
from nicole.index import Index, Sector

def build_spin1_u1():
    """Load spin-1 space and build MPO operator templates (U1 symmetry)."""
    spc, Op = load_space('Spin', 'U1', {'J': 1.0})   # spin-1

    # load_space returns Sz without an op axis; insert one.
    Op['Sz'].insert_index(2, direction=Direction.OUT)
    S = Op['Sp'] + Op['Sm'] + Op['Sz']
    Op['S'] = S

    Sdag, S4, S4dag = _make_spin_templates(S)
    Op['Sdag']  = Sdag
    Op['S4']    = S4
    Op['S4dag'] = S4dag

    Sz    = Op['Sz']
    Szdag = Sz.conj().permute([1, 0, 2])
    from alice.physics.system import _make_leading4, _make_terminal4
    Op['Sz4']    = _make_leading4(Sz)
    Op['Sz4dag'] = _make_terminal4(Szdag)

    Op['I4']    = _make_i4(spc)
    from alice.physics.system import _make_intermid4
    from nicole import identity
    Op['I4mid'] = _make_intermid4(identity(spc), Op['S4'])

    return spc, Op
```

## Using a custom space builder

Pass `space_fn` directly to `build_interaction`:

```python
from alice import build_interaction, build_hamiltonian

config = {
    "geometry": {"lattice": "chain", "lx": 10, "bcx": "OBC", "n2x": True},
    "model": {"category": "bosonic", "label": "Heisenberg",
              "symmetry": "U1", "spin": 1.0},
}

interactions, spc, L = build_interaction(config, space_fn=build_spin1_u1)
hamiltonian = build_hamiltonian(interactions, L, spc)
```

Or via TOML plugin spec:

```toml
[spin1_chain.plugin]
space = "my_space.py:build_spin1_u1"
```

The `space_fn` is forwarded to the model builder (e.g. `build_heisenberg`) as a keyword argument, overriding its default space builder.

## Fully custom site (no `load_space`)

For a system that Nicole's `load_space` does not support, construct the `Index` directly:

```python
from nicole import Direction
from nicole.index import Index, Sector

def build_my_site():
    """Example: a qutrit (3-level system) with no symmetry."""
    # Three states with trivial charge 0, each of dimension 1
    spc = Index([Sector(0, 3)], Direction.IN)

    # Build operator templates manually using Nicole tensor operations...
    # Op['I4'] = ...  (identity)
    # Op['A4'] = ...  (your leading-site template)
    # Op['A4dag'] = ... (your terminal-site template)

    return spc, Op
```

!!! note
    For sites without symmetry, use a single `Sector(0, d)` where `d` is the local dimension. Nicole will treat this as a dense (non-block-sparse) site.

## See Also

- [build_bosonic](../../api/local-space/build-bosonic.md), [build_fermionic](../../api/local-space/build-fermionic.md), [build_conductor](../../api/local-space/build-conductor.md) — built-in examples to follow.
- [Custom model](custom-model.md) — pair your space with a custom Hamiltonian.
