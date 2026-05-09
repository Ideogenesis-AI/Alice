# DMRG

Alice's DMRG algorithm finds the ground state of a Hamiltonian MPO via alternating sweep optimization. Three update schemes are available: 1-site, 2-site, and 1-site-plus (controlled bond expansion).

## API

| Symbol | Description |
|--------|-------------|
| [Options](options.md) | Run options: scheme, bond dimension, tolerances, caching |
| [Summary](summary.md) | Output: energy, MPS state, convergence info |
| [run](run.md) | Top-level entry point |

## Usage Pattern

```python
import alice
from alice import dmrg, build_interaction, build_hamiltonian

interactions, spc, L = build_interaction("config.toml")
mpo  = build_hamiltonian(interactions, L, spc)
opts = dmrg.Options(scheme='2s', n_sweeps=20, max_bond=128)

summary = dmrg.run(mps, mpo, opts)
print(summary.energy)
```

## Update Schemes

| Name | Alias | Description |
|------|-------|-------------|
| `'1s'` | `'1-site'`, `'one-site'` | Single-site update; preserves bond dimension |
| `'2s'` | `'2-site'`, `'two-site'` | Two-site update with SVD; drives bond growth |
| `'1sp'` | `'1-site-plus'`, `'one-site-plus'` | CBE: stability of 1-site + bond flexibility |

## See Also

- [dmrg.run](run.md) — full parameter reference.
- [DMRG examples](../../examples/dmrg/heisenberg.md)
