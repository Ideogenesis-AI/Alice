# Two-Site BUG

Alice's two-site BUG (Basis-Update & Galerkin) integrator evolves an MPS in real or imaginary time under a nearest-neighbour Hamiltonian. It is the faithful rank-adaptive BUG of Ceruti, Kusch & Lubich ([arXiv:2304.05660](https://arxiv.org/abs/2304.05660)): commuting even/odd Trotter sweeps of *local* K/L/S bond updates. Each update augments the left frame from the evolved **K** factor, augments the right frame from the evolved **L** factor, evolves the small core **S** in the augmented bases (Galerkin), then truncates with an SVD — so the bond dimension adapts to the growing entanglement (the basis augmentation). The local substeps exponentiate the *projected* effective Hamiltonian internally (Krylov `expv`); no pre-formed propagator gate is applied, and the update is exact at full rank.

The bond Hamiltonians are reused directly from the [AutoMPO](../interaction/build-interaction.md) interaction list, so any nearest-neighbour model and symmetry that `build_interaction` supports works unchanged.

## API

| Symbol | Description |
|--------|-------------|
| [Options](options.md) | Run options: time step, steps, Trotter order, bond dimension |
| [Summary](summary.md) | Output: evolved MPS, time/norm history, kept and augmented bond dims |
| [run](run.md) | Top-level entry point |

## Usage Pattern

```python
from alice import build_interaction, init_mps
from alice.algorithm import two_site_bug

interactions, spc, geo = build_interaction("config.toml")
mps  = init_mps(geo.L, spc, Op, config=[0, 1] * (geo.L // 2), target_qn=0)
opts = two_site_bug.Options(dt=0.05, n_steps=40, order='strang', max_bond=128)

summary = two_site_bug.run(mps, interactions, opts)
print(summary.max_bond_dims)  # kept bond dimension per step
print(summary.aug_dims)       # proposed (pre-truncation) augmentation per step
```

## Trotter Orders

| Name | Alias | Description |
|------|-------|-------------|
| `'strang'` | `'second'`, `'2'` | Symmetric second-order step `U_even(dt/2) U_odd(dt) U_even(dt/2)` |
| `'lie'` | `'first'`, `'1'` | First-order step `U_even(dt) U_odd(dt)` |

## See Also

- [two_site_bug.run](run.md) — full parameter reference.
- [build_interaction](../interaction/build-interaction.md) — build the `interactions` argument.
