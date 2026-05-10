# AutoMPO from TOML

`build_interaction` can load a model config directly from a TOML file, eliminating the need to write Python for standard models. This page walks through every example in `examples/example_config.toml`.

## TOML structure

A config file can contain multiple named sections. Each top-level table must have a `[<name>.geometry]` sub-table and a `[<name>.model]` sub-table:

```toml
[heisenberg_u1.geometry]
lattice = "chain"
lx      = 10
bcx     = "OBC"
n2x     = true

[heisenberg_u1.model]
category = "bosonic"
label    = "Heisenberg"
symmetry = "U1"
spin     = 0.5
J        = 1.0
```

Load a single section with Python's standard `tomllib`:

```python
import tomllib
from alice import build_interaction, build_hamiltonian

with open("example_config.toml", "rb") as f:
    cfg = tomllib.load(f)

interactions, spc, L = build_interaction(cfg["heisenberg_u1"])
hamiltonian = build_hamiltonian(interactions, L, spc)
print(f"L={L}, bonds={hamiltonian.bond_dims}")
```

Alternatively, pass the TOML file path directly (Alice uses `tomllib` internally):

```python
# If the TOML file contains exactly one top-level section, you can pass the
# section's sub-dict directly. For a multi-section file, index by name first.
interactions, spc, L = build_interaction(cfg["heisenberg_su2"])
```

---

## Example 1: Heisenberg chain, U(1)

```toml
[heisenberg_u1.geometry]
lattice = "chain"
lx      = 10
bcx     = "OBC"
n2x     = true

[heisenberg_u1.model]
category = "bosonic"
label    = "Heisenberg"
symmetry = "U1"
spin     = 0.5
J        = 1.0
```

This is the simplest case: a 10-site open chain with nearest-neighbor coupling, conserving \(S_z\) via U(1).

---

## Example 2: Heisenberg chain, SU(2)

```toml
[heisenberg_su2.geometry]
lattice = "chain"
lx      = 10
bcx     = "OBC"
n2x     = true

[heisenberg_su2.model]
category = "bosonic"
label    = "Heisenberg"
symmetry = "SU2"
spin     = 0.5
J        = 1.0
```

Changing `symmetry` to `"SU2"` enables full spin-rotation invariance. The bond dimension in the symmetry-resolved basis is typically 3–5× smaller than U(1) for the same ground-state accuracy.

---

## Example 3: Free-fermion chain

```toml
[free_fermion.geometry]
lattice = "chain"
lx      = 10
bcx     = "OBC"
n2x     = true

[free_fermion.model]
category = "fermionic"
label    = "FreeFermion"
symmetry = "U1"
t        = 1.0
mu       = 0.0
```

`mu = 0.0` corresponds to half-filling for this model.

---

## Example 4: Hubbard chain, U(1)×U(1)

```toml
[hubbard_u1u1.geometry]
lattice = "chain"
lx      = 8
bcx     = "OBC"
n2x     = true

[hubbard_u1u1.model]
category = "conductor"
label    = "Hubbard"
symmetry = "U1,U1"
t        = 1.0
U        = 4.0
mu       = 0.0
```

`"U1,U1"` conserves up-spin and down-spin particle numbers independently. Use `"U1,SU2"` to additionally enforce SU(2) spin symmetry.

---

## Example 5: Heisenberg square lattice (4×4)

```toml
[heisenberg_2d.geometry]
lattice  = "square"
traverse = "serpentine"
lx       = 4
ly       = 4
bcx      = "OBC"
bcy      = "OBC"
n2x      = true   # NN along x
n2y      = true   # NN along y
n3d      = true   # diagonal NNN (J2 coupling)
n3o      = false  # off-diagonal NNN

[heisenberg_2d.model]
category = "bosonic"
label    = "Heisenberg"
symmetry = "U1"
spin     = 0.5
J        = 1.0
Jp       = 0.3    # NNN coupling J'
```

A 4×4 square lattice (16 sites total, traversed in serpentine order). `n3d = true` adds diagonal NNN bonds. The model builder assigns `Jp` to NNN interactions labeled `['NNN', 'N3D']`.

---

## Example 6: Hubbard cylinder (8×8)

```toml
[hubbard_cylinder.geometry]
lattice  = "square"
traverse = "serpentine"
lx       = 8
ly       = 8
bcx      = "OBC"
bcy      = "PBC"    # periodic in y direction (cylinder geometry)
n2x      = true
n2y      = true
n3d      = true
n3o      = true

[hubbard_cylinder.model]
category = "conductor"
label    = "Hubbard"
symmetry = "U1,SU2"
t        = 1.0
tp       = 0.25
U        = 8.0
mu       = 0.0

[hubbard_cylinder.autompo]
trunc         = {thresh = 1e-14}
compact_every = 10
```

An 8×8 Hubbard cylinder with PBC in the y-direction and NNN hopping. The optional `[autompo]` section passes `trunc` and `compact_every` directly to `build_hamiltonian`:

```python
autompo_cfg = cfg["hubbard_cylinder"].get("autompo", {})
interactions, spc, L = build_interaction(cfg["hubbard_cylinder"])
hamiltonian = build_hamiltonian(
    interactions, L, spc,
    trunc         = autompo_cfg.get("trunc"),
    compact_every = autompo_cfg.get("compact_every", 10),
)
```

---

## See Also

- [build_interaction API](../api/interaction/build-interaction.md)
- [build_hamiltonian API](../api/hamiltonian/build-hamiltonian.md)
- [Geometry API](../api/geometry/index.md)
