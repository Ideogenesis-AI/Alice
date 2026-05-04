# build_interaction

Build a fully populated interaction list from a TOML config or dict.

::: alice.build_interaction
    options:
      heading_level: 2

## TOML config structure

A minimal config dict (or TOML section) must contain two sub-tables:

```toml
[model.geometry]
lattice = "chain"
lx      = 10
bcx     = "OBC"
n2x     = true

[model.model]
category = "bosonic"
label    = "Heisenberg"
symmetry = "U1"
spin     = 0.5
J        = 1.0
```

An optional `[model.plugin]` table can override geometry, space, or model callables:

```toml
[model.plugin]
geometry = "my_dir/my_geometry.py:build_my_lattice"
model    = "my_dir/my_model.py:build_my_model"
```

Plugin specs use the format `"path/to/file.py:function_name"`.

## See Also

- [Interaction](interaction.md), [Interaction2Site](interaction-2site.md) — objects returned in the list.
- [build_hamiltonian](../hamiltonian/build-hamiltonian.md) — next step after `build_interaction`.
- [AutoMPO from TOML example](../../examples/autompo-toml.md)
