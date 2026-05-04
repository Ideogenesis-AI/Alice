# Heisenberg Chain

Ground-state DMRG for the spin-1/2 Heisenberg chain.

\[
H = J \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j, \quad J = 1
\]

The exact ground-state energy per site in the thermodynamic limit (Bethe ansatz) is
\(E_\infty / L = \tfrac{1}{4} - \ln 2 \approx -0.4431\).

## 1. Setup

```python
import alice
alice.configure_logging()  # optional: write diagnostics to .logging/

from alice import build_interaction, build_hamiltonian, MPS, dmrg
```

## 2. Build the Hamiltonian from TOML

Save the following as `heisenberg.toml`:

```toml
[model.geometry]
lattice = "chain"
lx      = 20
bcx     = "OBC"
n2x     = true

[model.model]
category = "bosonic"
label    = "Heisenberg"
symmetry = "U1"
spin     = 0.5
J        = 1.0
```

Then load it:

```python
interactions, spc, L = build_interaction("heisenberg.toml")
hamiltonian = build_hamiltonian(interactions, L, spc)
print(f"L = {L}, MPO bond dims: {hamiltonian.bond_dims}")
```

Or equivalently, pass a config dict directly (no file needed):

```python
config = {
    "geometry": {"lattice": "chain", "lx": 20, "bcx": "OBC", "n2x": True},
    "model": {"category": "bosonic", "label": "Heisenberg",
              "symmetry": "U1", "spin": 0.5, "J": 1.0},
}
interactions, spc, L = build_interaction(config)
hamiltonian = build_hamiltonian(interactions, L, spc)
```

## 3. Build an initial MPS

Alice does not include a general MPS initializer in its public API (use the examples or your own initialization). Here we build a simple Néel-state product MPS:

```python
from nicole import isometry, Direction
from nicole.index import Index, Sector

ndigits = max(2, len(str(L)))
bond_trivial_out = Index([Sector(0, 1)], Direction.OUT)
bond_trivial_in  = bond_trivial_out.flip()

tensors = []
for i in range(L):
    # Alternate up (charge +1) and down (charge -1) to target Sz=0
    charge = 1 if i % 2 == 0 else -1
    phys = Index([Sector(charge, 1)], Direction.IN)
    t = isometry(bond_trivial_out.flip(), phys)
    t.insert_index(1, direction=Direction.OUT, itag=f'A{i+1:0{ndigits}d}')
    t.retag(0, f'A{i:0{ndigits}d}')
    t.retag(2, f's{i:02d}')
    tensors.append(t)

mps = MPS(tensors)
```

## 4. Run DMRG

```python
opts = dmrg.Options(
    scheme   = '2s',    # 2-site update with SVD bond growth
    n_sweeps = 20,
    max_bond = 64,
    e_tol    = 1e-8,
)

summary = dmrg.run(mps, hamiltonian, opts)

print(f"Energy:        {summary.energy:.10f}")
print(f"Energy/site:   {summary.energy / L:.10f}")
print(f"Converged:     {summary.converged}")
print(f"Sweeps done:   {summary.n_sweeps}")
print(f"Max bond dim:  {max(summary.bond_dims)}")
```

Expected output for `L=20`, `max_bond=64`:

```
Energy:        -8.6824...
Energy/site:   -0.4341...
Converged:     True
Sweeps done:   ...
Max bond dim:  64
```

## 5. SU(2) symmetry

To use full SU(2) spin-rotation symmetry, change `symmetry` to `"SU2"`:

```python
config["model"]["symmetry"] = "SU2"
interactions, spc, L = build_interaction(config)
hamiltonian = build_hamiltonian(interactions, L, spc)
```

With SU(2), the bond dimension in the symmetry-resolved basis is much smaller for the same physical accuracy. The DMRG run is identical.

## 6. Save and reload

```python
import torch

torch.save(summary.serialize(), "heisenberg_result.pt")

# Later:
data    = torch.load("heisenberg_result.pt", weights_only=True)
summary = dmrg.Summary.deserialize(data)
print(f"Reloaded energy: {summary.energy:.10f}")
```

## See Also

- [Free fermion](free-fermion.md), [Hubbard model](hubbard.md)
- [dmrg.Options API](../../api/dmrg/options.md)
- [build_heisenberg API](../../api/hamiltonian/build-heisenberg.md)
