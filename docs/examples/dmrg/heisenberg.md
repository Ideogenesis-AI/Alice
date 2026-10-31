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

from alice import build_interaction, build_hamiltonian, init_mps, dmrg
from nicole import load_space
```

## 2. Build the Hamiltonian from TOML

Save the following as `heisenberg.toml`:

```toml
[geometry]
lattice = "chain"
lx      = 20
bcx     = "OBC"
n2x     = true

[model]
category = "bosonic"
label    = "Heisenberg"
symmetry = "U1"
spin     = 0.5
J        = 1.0
```

Then load it:

```python
interactions, spc, geo = build_interaction("heisenberg.toml")
L = geo.L
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
interactions, spc, geo = build_interaction(config)
L = geo.L
hamiltonian = build_hamiltonian(interactions, L, spc)
```

## 3. Build an initial MPS

`init_mps` builds a symmetry-aware initial MPS from the physical space and
operators of the model. A bond-dimension-1 product state is the simplest
choice and pairs well with the 2-site scheme, which grows the bond dimension
as it sweeps:

```python
Spc, Op = load_space("Spin", "U1", {"J": 0.5})
mps = init_mps(L, Spc, Op, bond_dim=1)
```

Passing `bond_dim > 1` instead gives a random MPS whose bond sectors come from
a search around the center-bond charge. Note that neither state carries the
bond sectors of the ground state, so the 1-site scheme — which never changes
them — would converge above the true energy; see
[init_mps](../../api/network/init-mps.md).

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
interactions, spc, geo = build_interaction(config)
L = geo.L
hamiltonian = build_hamiltonian(interactions, L, spc)

Spc, Op = load_space("Spin", "SU2", {"J": 0.5})
mps = init_mps(L, Spc, Op, bond_dim=1)
```

With SU(2), the bond dimension in the symmetry-resolved basis is much smaller for the same physical accuracy. Only the initial MPS has to be rebuilt from the SU(2) space; the DMRG call itself is identical.

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
- [dmrg.Options API](../../algorithms/dmrg/options.md)
- [build_heisenberg API](../../api/hamiltonian/build-heisenberg.md)
