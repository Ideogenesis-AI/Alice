# build_hubbard

Populate interactions for the Hubbard model.

::: alice.physics.build_hubbard
    options:
      heading_level: 2

## Model parameters

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `symmetry` | str | `"U1,SU2"` | `"U1,U1"`, `"U1,SU2"`, `"Z2,U1"`, or `"Z2,SU2"` |
| `t` | float | `1.0` | NN hopping amplitude |
| `tp` | float | `0.0` | NNN hopping amplitude (requires NNN bonds) |
| `U` | float | `4.0` | On-site Coulomb repulsion |
| `mu` | float | `0.0` | Chemical potential relative to half-filling |

## Hamiltonian

\[
H = -t \sum_{\langle i,j \rangle, \sigma} (c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.})
  + U \sum_i n_{i\uparrow} n_{i\downarrow}
  - \mu \sum_{i,\sigma} n_{i\sigma}
\]

The `mu` parameter is relative to half-filling; the code automatically applies the \(-U/2\) shift for the `U1,SU2` case.

## See Also

- [build_conductor](../local-space/build-conductor.md) — space builder called internally.
- [build_heisenberg](build-heisenberg.md), [build_free_fermion](build-free-fermion.md) — other model builders.
