# VUMPS

Alice's VUMPS algorithm finds the ground state of an infinite chain directly in the thermodynamic limit. The state is a uniform MPS, described by a unit cell of `N` site tensors that tiles the chain, and the Hamiltonian is an [iMPO](../../api/network/impo.md) over the same cell. There is no system size, no boundary, and no finite-size extrapolation: the energy reported is an energy per site.

## API

| Symbol | Description |
|--------|-------------|
| [Options](options.md) | Run options: scheme, bond dimension, tolerances, caching |
| [Summary](summary.md) | Output: energy per site, iMPS state, convergence info |
| [run](run.md) | Top-level entry point |

## Usage Pattern

```python
import alice
from alice import vumps, build_interaction, build_ihamiltonian, init_imps
from nicole import load_space

interactions, spc, geo = build_interaction("config.toml")
impo = build_ihamiltonian(interactions, geo.L, spc)

Spc, Op = load_space('Spin', 'U1', {'J': 0.5})
imps = init_imps(geo.L, Spc, Op, bond_dim=16)

opts = vumps.Options(scheme='2s', n_iters=50, max_bond=32)
summary = vumps.run(imps, impo, opts)
print(summary.energy)       # energy per site
```

## Why not a DMRG sweep

A DMRG sweep visits every site of a finite chain, carrying environments it builds as it goes. An infinite chain has no site to start at and no environment to build up from a boundary, so the environments are fixed points instead: the left block is the leading eigenvector of the transfer operator built from the cell, and the right block likewise from the other side. They are re-solved from scratch after every iteration, because the state they describe has changed.

The unit of work is therefore an iteration, not a sweep. One iteration solves every position of the cell against one frozen set of environments, then re-solves the environments against the updated state. The positions within an iteration are independent of each other, which makes the state uniform: nothing is carried from one position to the next the way a sweep carries its center.

## Update Schemes

| Name | Alias | Description |
|------|-------|-------------|
| `'2s'` | `'2-site'`, `'two-site'` | Two-site update with SVD; grows the bond dimension through the split |
| `'1s'` | `'1-site'`, `'one-site'` | Single-site update with a polar gauge step; preserves bond dimension and charge sectors |
| `'1sp'` | `'1-site-plus'`, `'one-site-plus'` | Single-site update on a bond widened by controlled bond expansion; grows the bond dimension at one-site cost |

The one-site scheme solves two eigenproblems per cell position, one for the center tensor and one for the bond matrix, and glues them back together with a polar decomposition. It discards nothing, so it cannot grow the bond dimension, and it also cannot move weight from one charge sector to another: the sectors the initial state carried are the sectors the converged state carries. On the Heisenberg chain that limit shows up as an energy error around `1e-3` at a bond dimension where the two-site scheme reaches `1e-4`, with a gradient down near `1e-9` reporting, correctly, that the state is stationary within the manifold it is confined to.

The two-site scheme splits a solved pair, so it both widens bonds up to `max_bond` and redistributes weight between sectors. It is the default, and the scheme to reach for unless the state already carries both the bond dimension and the sectors the answer needs.

### Controlled bond expansion

The one-site-plus scheme reaches the same freedom through a different route. Before each center-tensor solve it widens the bond the solve sits on by `expand_k` directions, drawn from a cheap factored approximation of the two-site gradient: the Hamiltonian is applied to a truncated two-site tensor from each side separately, each half is projected onto the complement of what the state already spans, and the leading singular vectors of what survives are the directions the bond is missing. The one-site eigenproblem then fills them, and the split that follows cuts the bond back down to `max_bond`.

So the local problem stays one-site sized — a factor of `d` smaller on each side than a two-site solve — while the bond dimension still grows, by up to `expand_k` per bond per iteration instead of by the factor of `d` a split gives. Two knobs govern it. `expand_k` sets how many directions are offered, and so both the quality of the expanded space and how fast the state can widen. `expand_alpha` truncates the two-site tensor the expansion applies the Hamiltonian to; it keeps the expansion cheap against the Davidson solve it precedes, and `round(sqrt(max_bond))` is a reasonable starting point for both.

The expansion never moves the state: only the neighbor behind the bond is widened, and the center tensor is then re-expressed exactly in the widened basis, so the new directions enter with zero amplitude. A bond with no complement left — saturated, or simply out of the Hamiltonian's reach from the current state — is detected and left alone, and the step degenerates into a plain one-site update instead of expanding on roundoff.

## What the gradient means

Convergence is measured on the gauge mismatch of the state, the norm of the energy gradient on the uniform MPS manifold, maximized over the cell. Each position contributes the larger of `|AC - AL·C|` and `|AC - C·AR|`. Both forms are needed: a scheme that splits writes its answer back as one of them, so that one vanishes by construction at every site the pass solved, and reading it alone would call a sweep whose bonds are still moving converged. The Davidson eigenvalues cannot be used: they belong to regularized local problems, and the energy of an infinite chain is infinite, so the energy density comes from the environment fixed point instead. The gradient keeps falling long after the energy between iterations has flattened out, which makes it the more honest stopping criterion.

Two floors limit how far it can fall. It can never go below `davidson_tol`, since it compares two quantities neither of which is known better than that. And under either scheme that splits, against a bond cap that bites, it settles at a floor set by the discarded weight and does not fall at all: the solve keeps asking for a wider bond than it is allowed. A run that has grown to `max_bond` and stopped improving is converged in every sense that matters.

## Why the environment solver warns

The environment fixed point is reached by absorbing copies of the unit cell until the block stops changing, up to `env_maxiter` of them. The residual falls geometrically, at a rate set by the subleading eigenvalue of the MPS transfer operator, so the budget a tolerance \(\epsilon\) needs is roughly \(\xi \ln(1/\epsilon)\) with the correlation length \(\xi\) counted in unit cells. The default of 200 cells covers \(\xi\) up to about seven, which is comfortable on a gapped model at any bond dimension and too little on a critical one past `D = 16` or so.

Running out logs a warning and continues with the best estimate without abandoning the run. Both `env_maxiter` and `env_tol` resolve it, and which to reach for depends on what the number is for: the energy is converged long before the environment tensor is, while the gradient under `'1s'` tracks the environments directly. [Lessons](lessons.md) works through the measurements behind all of this.

## The unit cell

The cell tiles, so bond `N` is bond `0`, and the bond charge has to return to where it started after `N` sites. A cell therefore carries no net charge: half filling for fermions, zero magnetization for spins. A uniform state at any other filling is not representable on a cell of that length, and [init_imps](../../api/network/init-imps.md) refuses to build one instead of producing a cell that cannot be gauge fixed. Spin-1/2 admits no neutral cell of odd length at all, so `N` has to be even there.

## See Also

- [vumps.run](run.md) — full parameter reference.
- [Lessons](lessons.md) — the environment budget, measured.
- [init_imps](../../api/network/init-imps.md) — build the initial unit cell.
- [build_ihamiltonian](../../api/hamiltonian/build-ihamiltonian.md) — build the Hamiltonian iMPO.
- [VUMPS examples](../../examples/vumps/heisenberg.md)
