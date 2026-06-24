# Discarded-Projector BUG

Alice's discarded-projector BUG integrator is the MPS specialisation of the **Lubich tree-tensor-network BUG** — the rank-adaptive Basis-Update & Galerkin integrator of Ceruti–Lubich–Walach ([arXiv:2304.05660](https://arxiv.org/abs/2304.05660)). Like two-site TDVP and [DMRG](../dmrg/index.md), it evolves an MPS in real or imaginary time under a Hamiltonian **MPO**, exponentiating the two-site effective Hamiltonian with the left/right MPO environments (no Trotter splitting). It adapts the bond dimension to the growing entanglement — a domain-wall quench melts into the full ballistic light cone — and is **inverse-free** (no backward substep, no overlap-matrix inverse).

## The scheme

The reference Lubich TTN-BUG builds its tree by **recursive bisection** of the 1D modes (a balanced binary tree whose leaves are the physical sites). The MPS realisation therefore recursively bisects the chain and performs one **two-site** node update at each bisection bond, with two modifications from the reference:

1. **Two-site node update.** Where the reference updates a single-site node, here each node update is a two-site update of the bisection bond through the two-site effective Hamiltonian (the DMRG `matvec_2s`).
2. **Discarded projector — no overlap matrices.** Where the reference transports the core through augmented overlap matrices `M = Û† U0`, here the augmented frames are read directly off the evolved two-site block and the core is obtained by projecting that block onto them.

### One node update

For a bond window `Θ0 = U0 · S0 · V0`:

1. **Evolve the two-site block** once under the two-site effective Hamiltonian, `Θ1 = exp(τ H₂) Θ0` (Hermitian → Lanczos exponential). Acting with `H` on the window is what creates the new Schmidt direction — a domain-wall interface block has Schmidt rank 2, so the bond grows `1 → 2` in one step.
2. **Grow the frames with the discarded projector.** The augmented left frame is `Û = qr([colspace(Θ1 | link_l, site_l) | U0])` and the augmented right frame is `V̂ = qr([rowspace(Θ1 | link_r, site_r) ; V0])` — the direct sum of the old frame with the evolved block's column/row space, re-orthonormalised by a QR that drops dependent columns. No overlap matrix `M`/`N` is formed; the leading `U0`/`V0` keep the old frame exactly inside.
3. **Galerkin core + truncate.** The core is the projection of the already-evolved block, `S = Û† Θ1 V̂†`, SVD-truncated to `max_bond` / `cutoff` to set the new rank.

### One step

A step recursively bisects the chain: it updates the central bisection bond, then recurses into the left and right half-chains until **every** bond — every tree node — has had its two-site node update. Because every bond is a node, the bond dimension grows along the whole chain (the full light cone) as the wall melts, matching the bond profile of forward two-site TDVP.

The step is **first order** in `dt`; the rank growth / light-cone spread is the validated property. (A second-order symmetric composition is left to future work — a naive node-order-reversed Strang pass does not lift the order, because the per-node basis truncations are not a reversible flow.)

Everything stays in the symmetry-blocked Nicole representation, so the kept bond dimension respects the U(1) charge sectors throughout. The Hamiltonian is a standard [AutoMPO](../hamiltonian/build-hamiltonian.md) MPO, so any model and symmetry that `build_hamiltonian` supports works unchanged.

## API

| Symbol | Description |
|--------|-------------|
| [Options](options.md) | Run options: time step, steps, max bond dimension, cutoff |
| [Summary](summary.md) | Output: evolved MPS, time/norm history, kept bond dims per step |
| [run](run.md) | Top-level entry point |

## Usage Pattern

```python
from alice import build_interaction, build_hamiltonian, init_mps
from alice.algorithm import discarded_bug

interactions, spc, geo = build_interaction("config.toml")
mpo = build_hamiltonian(interactions, geo.L, spc)
mps = init_mps(geo.L, spc, Op, config=[0, 1] * (geo.L // 2), target_qn=0)
opts = discarded_bug.Options(dt=0.05, n_steps=40, max_bond=128)

summary = discarded_bug.run(mps, mpo, opts)
print(summary.bond_dims)       # bond dimensions of the final state (the light cone)
print(summary.max_bond_dims)   # max kept bond dimension per step
```

## See Also

- [discarded_bug.run](run.md) — full parameter reference.
- [build_hamiltonian](../hamiltonian/build-hamiltonian.md) — build the Hamiltonian `MPO` argument.
- [DMRG](../dmrg/index.md) — shares the MPO-environment / two-site-effective-Hamiltonian machinery.
