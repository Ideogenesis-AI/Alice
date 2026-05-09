# build_heisenberg

Populate interactions for a Heisenberg spin model.

::: alice.physics.build_heisenberg
    options:
      heading_level: 2

## Model Parameters

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `symmetry` | str | `"U1"` | `"U1"` or `"SU2"` |
| `spin` | float | `0.5` | Site spin quantum number |
| `J` | float | `1.0` | NN coupling constant |
| `Jp` | float | `0.0` | NNN coupling constant (requires NNN bonds in geometry) |

## Hamiltonian

\[
H = J \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j
  + J' \sum_{\langle\langle i,j \rangle\rangle} \mathbf{S}_i \cdot \mathbf{S}_j
\]

For SU(2) symmetry, the full rotational invariance is preserved. For U(1), the spin-`z` projection is conserved.

## See Also

- [build_bosonic](../local-space/build-bosonic.md) — space builder called internally.
- [build_free_fermion](build-free-fermion.md), [build_hubbard](build-hubbard.md) — fermionic counterparts.
