# XTRG

Alice's XTRG algorithm computes finite-temperature thermodynamic properties of
a Hamiltonian MPO via exponential cooling. The thermal density matrix is
repeatedly squared — `ρ(β_{n+1}) ≈ compress(ρ_n ⊗ ρ_n)` — using a
sweep-based variational MPO-MPO compression kernel.

## API

| Symbol | Description |
|--------|-------------|
| [Options](options.md) | Run options: scheme, τ₀, cooling steps, bond dimension, caching, artifacts |
| [Summary](summary.md) | Thermodynamic history: log Z, free energy, internal energy, specific heat, entropy |
| [Artifact](artifact.md) | Density-matrix snapshot `ρ(β)` at one cooling step |
| [run](run.md) | Top-level entry point |

## Usage Pattern

```python
from alice.network import build_interaction, build_hamiltonian
from alice.network.thermal import thermal_mpo
from alice.algorithm import xtrg

interactions, spc, geo = build_interaction(cfg)
H = build_hamiltonian(interactions, geo.L, spc)

opts = xtrg.Options(scheme='2s', n_steps=20, max_bond=64)
rho0 = thermal_mpo(H, opts.tau_0, opts.taylor_order, spc)
summary, artifact = xtrg.run(xtrg.Artifact(rho=rho0, beta=opts.tau_0, step=0), opts)

for beta, f in zip(summary.betas, summary.free_energies):
    print(f"β = {beta:.4f},  f = {f:.6f}")
```

`run()` takes the starting density matrix as an `Artifact` (`rho`, `beta`, `step`). Resuming an
interrupted run is just calling `run()` again with the `Artifact` loaded from `progress.ckpt`:

```python
resumed = xtrg.Artifact.load(ckpt_dir / 'progress.ckpt')
summary, artifact = xtrg.run(resumed, opts)  # thermal.ckpt auto-loaded from opts.checkpoint_dir
```

## Update Schemes

| Name | Alias | Description |
|------|-------|-------------|
| `'1s'` | `'1-site'`, `'one-site'` | 1-site direct contraction; preserves bond dimension |
| `'2s'` | `'2-site'`, `'two-site'` | 2-site SVD with truncation; drives bond growth |
| `'1sp'` | `'1-site-plus'`, `'one-site-plus'` | CBE: 1-site cost with 2-site-like bond flexibility |

## Temperature Grid

XTRG samples an exponentially spaced β grid starting from `τ₀`:

```
β_n = 2^n × τ₀,  n = 0, 1, …, n_steps
```

Thermodynamic observables (u, c_V, S) are derived from log Z using log-β
finite differences, which give uniform O((ln 2)²) discretization error
across the grid.

## See Also

- [xtrg.run](run.md) — full parameter reference.
- [NormalMPO](../../api/network/normal-mpo.md) — the density matrix representation.
- [thermal_mpo](../../api/network/thermal-mpo.md) — used to build the initial `Artifact`.
