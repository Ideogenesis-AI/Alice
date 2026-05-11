# build_free_fermion

Populate interactions for a spinless free-fermion (tight-binding) model.

::: alice.physics.build_free_fermion
    options:
      heading_level: 2

## Model Parameters

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `symmetry` | str | `"U1"` | `"U1"` (particle number) or `"Z2"` (fermion parity) |
| `t` | float | `1.0` | NN hopping amplitude |
| `tp` | float | `0.0` | NNN hopping amplitude (requires NNN bonds in geometry) |
| `mu` | float | `0.0` | Chemical potential |

## Hamiltonian

\[
H = -t \sum_{\langle i,j \rangle} (c^\dagger_i c_j + \text{h.c.})
  - t' \sum_{\langle\langle i,j \rangle\rangle} (c^\dagger_i c_j + \text{h.c.})
  - \mu \sum_i n_i
\]

Jordan-Wigner string operators are inserted automatically at intermediate sites for long-range bonds.

## See Also

- [build_fermionic](../local-space/build-fermionic.md) — space builder called internally.
- [build_heisenberg](build-heisenberg.md), [build_hubbard](build-hubbard.md) — other model builders.
