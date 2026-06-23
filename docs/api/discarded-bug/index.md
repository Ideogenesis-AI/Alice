# Discarded-Projector BUG

Alice's discarded-projector BUG integrator is a rank-adaptive Basis-Update & Galerkin time integrator derived from the faithful Ceruti–Kusch–Lubich scheme ([arXiv:2304.05660](https://arxiv.org/abs/2304.05660)), differing only in the *local bond update*. Like the [two-site BUG](../two-site-bug/index.md), it evolves an MPS in real or imaginary time under a nearest-neighbour Hamiltonian by commuting even/odd Trotter sweeps of local K/L/S updates, and adapts the bond dimension to the growing entanglement. The two variants share their sweep, their AutoMPO bond Hamiltonians, their Krylov `expv` substeps, and their `Options`/`Summary` records — only the per-bond candidate differs.

## How it differs from the faithful BUG

For a bond state `Θ0 = U0 · S0 · V0`, the faithful update evolves `K0 = U0·S0` under the right-projected generator, orthonormalises `[U0 | K1]` *through an overlap matrix* `M̂`, and transports the core as `Ŝ0 = M̂ S0 N̂`. The discarded variant changes exactly two things:

1. **Project-before.** The discarded (orthogonal-complement) projector is applied to the K/L *generator* before the exponential — `G_K = P⊥_U0 · H_K` with `P⊥_U0 = I − U0 U0†`, and `G_L = H_L · P⊥_V0` with `P⊥_V0 = I − V0† V0`. The projected generator is non-Hermitian, so the K/L substep uses a symmetry-preserving tensor **Arnoldi** exponential rather than the Hermitian Lanczos.
2. **Direct sum, no overlap matrices.** New directions are stacked onto the old isometry by a plain per-sector QR (`Û = [U0 | Qk]`, `V̂ = [V0 ; Ql]`) — no `M̂`/`N̂` is formed. The S-step projects the current two-site tensor straight onto the augmented bases, `Ŝ0 = Û† Θ0 V̂†`, evolves it (Galerkin), and truncates with an SVD.

Everything stays in the symmetry-blocked Nicole representation, so the kept bond dimension respects the U(1) charge sectors throughout. The bond Hamiltonians are reused directly from the [AutoMPO](../interaction/build-interaction.md) interaction list, so any nearest-neighbour model and symmetry that `build_interaction` supports works unchanged.

## API

| Symbol | Description |
|--------|-------------|
| [Options](options.md) | Run options: time step, steps, Trotter order, bond dimension |
| [Summary](summary.md) | Output: evolved MPS, time/norm history, kept and augmented bond dims |
| [run](run.md) | Top-level entry point |

## Usage Pattern

```python
from alice import build_interaction, init_mps
from alice.algorithm import discarded_bug

interactions, spc, geo = build_interaction("config.toml")
mps  = init_mps(geo.L, spc, Op, config=[0, 1] * (geo.L // 2), target_qn=0)
opts = discarded_bug.Options(dt=0.05, n_steps=40, order='strang', max_bond=128)

summary = discarded_bug.run(mps, interactions, opts)
print(summary.max_bond_dims)  # kept bond dimension per step
print(summary.aug_dims)       # proposed (pre-truncation) augmentation per step
```

## Trotter Orders

| Name | Alias | Description |
|------|-------|-------------|
| `'strang'` | `'second'`, `'2'` | Symmetric second-order step `U_even(dt/2) U_odd(dt) U_even(dt/2)` |
| `'lie'` | `'first'`, `'1'` | First-order step `U_even(dt) U_odd(dt)` |

## See Also

- [Two-Site BUG](../two-site-bug/index.md) — the faithful CKL variant this is derived from.
- [discarded_bug.run](run.md) — full parameter reference.
- [build_interaction](../interaction/build-interaction.md) — build the `interactions` argument.
