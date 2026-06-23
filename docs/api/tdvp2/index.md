# Two-Site TDVP

Alice's two-site TDVP (Time-Dependent Variational Principle) integrator evolves an MPS in real or imaginary time under a Hamiltonian MPO. It is the projector-splitting scheme of Haegeman et al. ([arXiv:1408.5056](https://arxiv.org/abs/1408.5056)) with a two-site update so the bond dimension adapts. A forward half-sweep evolves each two-site block forward in time and the carried one-site tensor backward in time (the inverse-free backward correction that removes the double counting of the shared bond); a reverse half-sweep mirrors it; a symmetric step composes the two halves for second-order accuracy. The local substeps exponentiate the *effective Hamiltonian* — the MPS tensor bracketed by the left/right MPO environments — with a Hermitian Krylov `expv`.

TDVP needs the full effective Hamiltonian, so it takes a Hamiltonian MPO built by [`build_hamiltonian`](../hamiltonian/build-hamiltonian.md) — exactly like [DMRG](../dmrg/index.md). It reuses the DMRG environment machinery and effective-Hamiltonian contractions.

## API

| Symbol | Description |
|--------|-------------|
| [Options](options.md) | Run options: time step, steps, bond dimension, real/imaginary time |
| [Summary](summary.md) | Output: evolved MPS, time/norm history, kept bond dims |
| [run](run.md) | Top-level entry point |

## Usage Pattern

```python
from alice import build_interaction, build_hamiltonian, init_mps
from alice.algorithm import tdvp2

interactions, spc, geo = build_interaction("config.toml")
mpo  = build_hamiltonian(interactions, geo.L, spc)
mps  = init_mps(geo.L, spc, Op, config=[0, 1] * (geo.L // 2), target_qn=0)
opts = tdvp2.Options(dt=0.05, n_steps=40, max_bond=128)

summary = tdvp2.run(mps, mpo, opts)
print(summary.max_bond_dims)  # kept bond dimension per step
print(summary.norms)          # norm per step (≈1 for real time; decays for imaginary)
```

## Real vs. Imaginary Time

| `imaginary_time` | Propagator | Use |
|------------------|------------|-----|
| `False` (default) | `exp(-i dt H)` | unitary real-time dynamics; the norm is conserved |
| `True` | `exp(-dt H)` | imaginary-time cooling toward the ground state (pair with `normalize=True`) |

!!! note "Convergence at fixed bond dimension"
    At fixed or adaptively-capped bond dimension, two-site TDVP's error is a
    *manifold-projection* error that does not vanish as `dt → 0` — it plateaus —
    rather than the `O(dt²)` state error of a full-rank propagator. Refine the bond
    dimension (`max_bond`) to reduce the plateau.

## See Also

- [tdvp2.run](run.md) — full parameter reference.
- [build_hamiltonian](../hamiltonian/build-hamiltonian.md) — build the `mpo` argument.
- [DMRG](../dmrg/index.md) — ground-state search sharing the same MPO/environment core.
