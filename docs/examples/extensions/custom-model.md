# Custom Model

A model builder is a function that takes the bare `Interaction` list produced by the geometry stage and fills in `cpl` and tensor fields for each interaction. You can provide a custom model builder to implement any Hamiltonian whose physical operators are defined in an `Op` dictionary from a space builder.

## Signature

Every model builder must have the following signature:

```python
def my_model(
    interactions: list,  # list[Interaction] from the geometry stage
    L: int,              # chain length
    *,
    symmetry: str = 'U1',
    # ... your model parameters ...
    space_fn = None,     # optional override for the space builder
    **_ignored,          # silently ignore extra TOML keys
) -> tuple:              # returns (spc, ops)
```

The function **modifies `interactions` in place** and returns `(spc, ops)` where `spc` is the physical `Index` and `ops` is the operator dict.

## What to Do in a Model Builder

1. Call the space builder (or `space_fn` if provided) to get `(spc, Op)`.
2. Iterate over `interactions`. For each interaction:
   - Set `intr.cpl` to the physical coupling constant.
   - Set the tensor fields from `Op` (using `.clone()` to avoid aliasing).
   - For `Interaction2Site` with `terminal_site > leading_site + 1`, also set `intr.intermid_tnsr`.

## Example: XXZ Model

The Heisenberg model is isotropic \((J_x = J_y = J_z = J)\). An anisotropic XXZ model allows different couplings for the \(S^z S^z\) and transverse \(S^\pm S^\mp\) channels:

\[
H = J_{xy} \sum_{\langle i,j \rangle} (S^+_i S^-_j + S^-_i S^+_j)
  + J_z   \sum_{\langle i,j \rangle} S^z_i S^z_j
\]

```python
# my_model.py
from alice import Interaction2Site
from alice.physics.system import build_bosonic


def build_xxz(
    interactions,
    L: int = 0,
    *,
    symmetry: str = 'U1',
    spin: float = 0.5,
    Jxy: float = 1.0,
    Jz: float = 1.0,
    space_fn=None,
    **_ignored,
):
    """XXZ model: anisotropic Heisenberg Hamiltonian.

    H = Jxy Σ (S+_i S-_j + h.c.) + Jz Σ Sz_i Sz_j
    """
    _space = space_fn if space_fn is not None else build_bosonic
    spc, ops = _space(symmetry=symmetry, spin=spin)

    S4    = ops['S4']
    S4dag = ops['S4dag']
    I4mid = ops['I4mid']

    # Sz templates (available for U1; build_bosonic adds Sz4 / Sz4dag for non-SU2)
    Sz4    = ops.get('Sz4')
    Sz4dag = ops.get('Sz4dag')

    for intr in interactions:
        if not isinstance(intr, Interaction2Site):
            continue
        if 'NN' not in intr.label:
            continue

        # Transverse channel: (S+ S- + S- S+) = S . S (for U1, S4 includes both)
        # For simplicity, use full S4/S4dag and separate Sz contribution.
        intr.cpl           = Jxy           # S.S coupling (transverse part)
        intr.leading_tnsr  = S4.clone()
        intr.terminal_tnsr = S4dag.clone()

        if intr.terminal_site > intr.leading_site + 1:
            intr.intermid_tnsr = I4mid.clone()

    # For anisotropic Jz ≠ Jxy, append separate Sz interactions.
    if Jz != Jxy and Sz4 is not None:
        from alice import Interaction2Site as I2S
        for intr in list(interactions):
            if not isinstance(intr, I2S) or 'NN' not in intr.label:
                continue
            extra = I2S(
                cpl           = Jz - Jxy,      # correction to full S.S
                label         = intr.label,
                leading_site  = intr.leading_site,
                terminal_site = intr.terminal_site,
                leading_tnsr  = Sz4.clone(),
                terminal_tnsr = Sz4dag.clone(),
                intermid_tnsr = I4mid.clone() if intr.terminal_site > intr.leading_site + 1 else None,
            )
            interactions.append(extra)

    return spc, ops
```

## Registering via `geometry_fn`

```python
from alice import build_interaction, build_hamiltonian
from my_model import build_xxz

config = {
    "geometry": {"lattice": "chain", "lx": 16, "bcx": "OBC", "n2x": True},
    "model": {"category": "bosonic", "label": "XXZ",
              "symmetry": "U1", "spin": 0.5, "Jxy": 1.0, "Jz": 2.0},
}

interactions, spc, L = build_interaction(config, model_fn=build_xxz)
hamiltonian = build_hamiltonian(interactions, L, spc)
```

## Registering via TOML Plugin

```toml
[xxz.geometry]
lattice = "chain"
lx      = 16
bcx     = "OBC"
n2x     = true

[xxz.model]
category = "bosonic"
label    = "XXZ"
symmetry = "U1"
spin     = 0.5
Jxy      = 1.0
Jz       = 2.0

[xxz.plugin]
model = "my_model.py:build_xxz"
```

## Tips

- **Always clone tensors** from `Op`: `intr.leading_tnsr = S4.clone()`. Sharing tensor objects between interactions leads to unexpected aliasing bugs.
- **Leave `cpl = 0.0` for inactive interactions** (e.g. NNN bonds when `Jp = 0`). `build_hamiltonian` skips zero-coupling interactions automatically.
- **For long-range bonds**, `intermid_tnsr` must be set whenever `terminal_site > leading_site + 1`. Use the bosonic identity string (`I4mid`) or the fermionic Jordan-Wigner string (`Z4mid`) from the `Op` dict.

## See Also

- [build_heisenberg](../../api/hamiltonian/build-heisenberg.md), [build_free_fermion](../../api/hamiltonian/build-free-fermion.md) — built-in reference implementations.
- [Custom geometry](custom-geometry.md) — pair a custom model with a custom lattice.
- [Custom local space](custom-space.md) — provide custom operator templates.
