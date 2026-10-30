# Free Fermion

Ground-state DMRG for a spinless tight-binding chain.

\[
H = -t \sum_{\langle i,j \rangle} (c^\dagger_i c_j + \text{h.c.}) - \mu \sum_i n_i, \quad t = 1, \; \mu = 0
\]

At half-filling (`µ = 0`), the exact ground-state energy for a chain of length `L` with OBC is
\(E_0 = -t \sum_{k=1}^{L/2} 2\cos\bigl(\pi k / (L+1)\bigr)\).

## 1. Setup

```python
import alice
alice.configure_logging()

from alice import build_interaction, build_hamiltonian, init_mps, dmrg
from nicole import load_space
```

## 2. Build the Hamiltonian

```python
config = {
    "geometry": {"lattice": "chain", "lx": 20, "bcx": "OBC", "n2x": True},
    "model": {"category": "fermionic", "label": "FreeFermion",
              "symmetry": "U1", "t": 1.0, "mu": 0.0},
}
interactions, spc, geo = build_interaction(config)
L = geo.L
hamiltonian = build_hamiltonian(interactions, L, spc)
print(f"L = {L}, MPO bond dims: {hamiltonian.bond_dims}")
```

## 3. Build an initial MPS

`init_mps` builds the state from the physical space and operators of the
model. With `bond_dim=1` and an explicit `config` it gives a product state;
here alternating occupied (sector 1) and empty (sector 0) sites put the chain
at half-filling:

```python
Spc, Op = load_space("Ferm", "U1")
mps = init_mps(L, Spc, Op, bond_dim=1, config=[1 if i % 2 == 0 else 0 for i in range(L)])
```

Omitting `config` lets `init_mps` pick a balanced half-filled configuration
itself, and `bond_dim > 1` gives a random MPS instead. A product state does
not carry the bond sectors of the ground state, so pair it with the 2-site
scheme below, which adapts them as it sweeps; the 1-site scheme never changes
them and would converge above the true energy.

## 4. Run DMRG

```python
opts = dmrg.Options(
    scheme   = '2s',
    n_sweeps = 10,
    max_bond = 32,
    e_tol    = 1e-10,
)

summary = dmrg.run(mps, hamiltonian, opts)

import numpy as np
exact = -2 * sum(np.cos(np.pi * k / (L + 1)) for k in range(1, L // 2 + 1))
print(f"DMRG energy:   {summary.energy:.10f}")
print(f"Exact energy:  {exact:.10f}")
print(f"Error:         {abs(summary.energy - exact):.2e}")
```

The error should be below `1e-10` for `max_bond = 32` on a free-fermion chain of length 20.

## See Also

- [Heisenberg chain](heisenberg.md), [Hubbard model](hubbard.md)
- [build_free_fermion API](../../api/hamiltonian/build-free-fermion.md)
- [build_fermionic API](../../api/local-space/build-fermionic.md)
