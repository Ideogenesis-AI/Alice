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

from alice import build_interaction, build_hamiltonian, MPS, dmrg
```

## 2. Build the Hamiltonian

```python
config = {
    "geometry": {"lattice": "chain", "lx": 20, "bcx": "OBC", "n2x": True},
    "model": {"category": "fermionic", "label": "FreeFermion",
              "symmetry": "U1", "t": 1.0, "mu": 0.0},
}
interactions, spc, L = build_interaction(config)
hamiltonian = build_hamiltonian(interactions, L, spc)
print(f"L = {L}, MPO bond dims: {hamiltonian.bond_dims}")
```

## 3. Build an initial MPS

For a half-filled chain with U(1) symmetry, alternate occupied (charge 1) and empty (charge 0) sites:

```python
from nicole import isometry, Direction
from nicole.index import Index, Sector

ndigits = max(2, len(str(L)))
bond_0 = Index([Sector(0, 1)], Direction.OUT)

tensors = []
for i in range(L):
    charge = 1 if i % 2 == 0 else 0   # half-filling product state
    phys = Index([Sector(charge, 1)], Direction.IN)
    t = isometry(bond_0.flip(), phys)
    t.insert_index(1, direction=Direction.OUT, itag=f'A{i+1:0{ndigits}d}')
    t.retag(0, f'A{i:0{ndigits}d}')
    t.retag(2, f's{i:02d}')
    tensors.append(t)

mps = MPS(tensors)
```

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

## 5. With NNN hopping

```python
config_nnn = {
    "geometry": {"lattice": "chain", "lx": 20, "bcx": "OBC",
                 "n2x": True, "n3d": True},   # n3d enables NNN on a 2D lattice
    "model": {"category": "fermionic", "label": "FreeFermion",
              "symmetry": "U1", "t": 1.0, "tp": 0.3, "mu": 0.0},
}
```

!!! note
    For a 1D chain, NNN bonds require the `n3d` flag. The geometry builder detects `ly=1` and delegates to `intrcmap_1dchain`, but NNN along a 1D chain is not supported by that builder. Use the square lattice builder with `ly=1` to enable NNN: set `lattice = "square"` and `ly = 1` in the TOML.

## See Also

- [Heisenberg chain](heisenberg.md), [Hubbard model](hubbard.md)
- [build_free_fermion API](../../api/hamiltonian/build-free-fermion.md)
- [build_fermionic API](../../api/local-space/build-fermionic.md)
