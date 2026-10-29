# Heisenberg Chain

Ground-state VUMPS for the spin-1/2 Heisenberg chain, directly in the thermodynamic limit.

\[
H = J \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j, \quad J = 1
\]

The exact energy per site is \(E/N = 1/4 - \ln 2 \approx -0.443147\). Nothing has to be extrapolated away: the state is a uniform MPS on an infinite chain, so the difference from that number is truncation error alone.

## 1. Setup

```python
import alice
alice.configure_logging()

from alice import build_interaction, build_ihamiltonian, init_imps, vumps
from nicole import load_space
```

## 2. Build the Hamiltonian iMPO

The geometry describes the unit cell rather than a chain, so `lx` is the cell length and `bcx` is `IBC`:

```python
config = {
    "geometry": {"lattice": "chain", "lx": 2, "bcx": "IBC", "n2x": True},
    "model": {"category": "bosonic", "label": "Heisenberg",
              "symmetry": "U1", "spin": 0.5, "J": 1.0},
}
interactions, spc, geo = build_interaction(config)
impo = build_ihamiltonian(interactions, geo.L, spc)
print(f"N = {geo.L}, iMPO bond dims: {impo.bond_dims}")
```

## 3. Build an initial unit cell

```python
Spc, Op = load_space('Spin', 'U1', {'J': 0.5})
imps = init_imps(geo.L, Spc, Op, bond_dim=16)
```

A cell of two sites is enough for the uniform ground state. Spin-1/2 admits no charge-neutral cell of odd length, so `init_imps` refuses one; see [init_imps](../../api/network/init-imps.md) for why.

## 4. Run VUMPS

```python
opts = vumps.Options(
    scheme   = '2s',
    n_iters  = 50,
    max_bond = 16,
    grad_tol = 1e-10,
    env_tol  = 1e-8,
)

summary = vumps.run(imps, impo, opts)

import numpy as np
exact = 0.25 - np.log(2)
print(f"VUMPS energy per site: {summary.energy:.10f}")
print(f"Exact energy per site: {exact:.10f}")
print(f"Error:                 {abs(summary.energy - exact):.2e}")
print(f"Gradient:              {summary.gradients[-1]:.2e}")
```

At `max_bond = 16` the energy per site comes out around `5e-5` above the exact value. Widening to 32 or 64 brings it down further; the gradient settles at a floor set by the discarded weight rather than falling to `grad_tol`, which is what a two-site run against a bond cap looks like when it has converged.

The chain is critical, so `env_tol = 1e-8` is deliberate: the environment fixed point converges at a rate set by a correlation length that the bond dimension itself cuts off, and reaching `1e-12` takes more cells than `env_maxiter` allows. The energy is unaffected either way, so loosening the tolerance is the cheaper of the two fixes. See [Lessons](../../algorithms/vumps/lessons.md#the-environment-budget).

Running under `'1s'` instead would strand the state: the one-site scheme can neither grow the bond dimension nor move weight between charge sectors, so from a random initial cell it converges to a stationary point of the sectors that cell happened to carry, roughly an order of magnitude worse in energy.

## Command-line script

```bash
uv run python examples/examples_vumps/vumps_heisenberg.py --cell 2 --bond-dim 16
uv run python examples/examples_vumps/vumps_heisenberg.py --cell 2 --bond-dim 32 --history
uv run python examples/examples_vumps/vumps_heisenberg.py --cell 2 --bond-dim 16 --init product
```

## See Also

- [Free fermion](free-fermion.md)
- [VUMPS overview](../../algorithms/vumps/index.md)
- [init_imps API](../../api/network/init-imps.md)
- [DMRG on the finite chain](../dmrg/heisenberg.md)
