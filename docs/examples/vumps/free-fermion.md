# Free Fermion

Ground-state VUMPS for a spinless tight-binding chain, directly in the thermodynamic limit.

\[
H = -t \sum_{\langle i,j \rangle} (c^\dagger_i c_j + \text{h.c.}), \quad t = 1
\]

The exact energy per site at half filling is \(E/N = -2t/\pi \approx -0.636620\), with no finite-size correction to subtract.

## 1. Setup

```python
import alice
alice.configure_logging()

from alice import build_interaction, build_ihamiltonian, init_imps, vumps
from nicole import load_space
```

## 2. Build the Hamiltonian iMPO

```python
config = {
    "geometry": {"lattice": "chain", "lx": 2, "bcx": "IBC", "n2x": True},
    "model": {"category": "fermionic", "label": "FreeFermion",
              "symmetry": "U1", "t": 1.0},
}
interactions, spc, geo = build_interaction(config)
impo = build_ihamiltonian(interactions, geo.L, spc)
```

## 3. Build an initial unit cell

```python
Spc, Op = load_space('Ferm', 'U1')
imps = init_imps(geo.L, Spc, Op, bond_dim=16)
```

Half filling is not a setting here, it is what a unit cell is. The bond charge has to return to itself after `N` sites, so the cell carries no net charge, and a uniform state at any other filling is not representable on a cell of that length. A doped state needs a longer cell, which fixes the filling at a rational number of its own.

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
exact = -2.0 / np.pi
print(f"VUMPS energy per site: {summary.energy:.10f}")
print(f"Exact energy per site: {exact:.10f}")
print(f"Error:                 {abs(summary.energy - exact):.2e}")
```

At `max_bond = 16` the error is around `9e-5`. The chain is gapless, so the entanglement grows with the bond dimension rather than saturating, and the error falls slowly with it, exactly as on the Heisenberg chain.

## 5. Growing from a product state

A cell of bond dimension one is a deterministic alternating occupied and empty pattern, and the two-site scheme widens it from the first iteration:

```python
imps = init_imps(geo.L, Spc, Op, bond_dim=1)
summary = vumps.run(imps, impo, vumps.Options(scheme='2s', n_iters=50, max_bond=16))
print(summary.bond_dims)
```

It reaches the same energy as a run started at width 16, which is a useful check that neither start is sitting in a local minimum of its own.

## Command-line script

```bash
uv run python examples/examples_vumps/vumps_freefermion.py --cell 2 --bond-dim 16
uv run python examples/examples_vumps/vumps_freefermion.py --cell 2 --bond-dim 32 --history
uv run python examples/examples_vumps/vumps_freefermion.py --cell 2 --bond-dim 16 --init product
```

## See Also

- [Heisenberg chain](heisenberg.md)
- [VUMPS overview](../../algorithms/vumps/index.md)
- [init_imps API](../../api/network/init-imps.md)
- [DMRG on the finite chain](../dmrg/free-fermion.md)
