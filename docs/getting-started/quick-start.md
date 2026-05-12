# Quick Start

This page walks you through a complete DMRG calculation for the spin-1/2 Heisenberg chain in about 25 lines of Python.

## Prerequisites

Make sure Alice is installed:

```bash
pip install alice-net
```

## Step-by-step

### 1. Configure logging

```python
import alice
alice.configure_logging()   # writes a timestamped .log file under .logging/
```

### 2. Define the model via TOML

Create a file `heisenberg.toml`:

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

### 3. Build interactions and Hamiltonian

```python
from alice import build_interaction, build_hamiltonian

interactions, spc, L = build_interaction("heisenberg.toml")
hamiltonian = build_hamiltonian(interactions, L, spc)
```

### 4. Create an initial MPS

`init_mps` constructs a symmetry-aware initial MPS from the physical space
returned by Nicole's `load_space`. It works for all particle types without a
separate symmetry argument.

```python
from nicole import load_space
from alice import init_mps

Spc, Op = load_space('Spin', 'U1', {'J': 0.5})

# Product state (bond_dim=1) — exact charge targeting, best for CBE / 2-site DMRG
mps = init_mps(L, Spc, Op, bond_dim=1)

# Random MPS (bond_dim>1) — group-derived bond sectors, more friendly to 1-site DMRG
# mps = init_mps(L, Spc, Op, bond_dim=32)
```

See [init_mps](../api/network/init-mps.md) for the full signature and all
supported symmetry groups.

### 5. Run DMRG

```python
from alice import dmrg, MPS

opts = dmrg.Options(
    scheme    = '2s',
    n_sweeps  = 20,
    max_bond  = 64,
    e_tol     = 1e-8,
)

summary = dmrg.run(mps, hamiltonian, opts)
print(f"Ground-state energy: {summary.energy:.10f}")
print(f"Converged: {summary.converged} after {summary.n_sweeps} sweeps")
```

### 6. Save the result

```python
summary.save("ground_state.ckpt")
```

Reload later:

```python
summary = dmrg.Summary.load("ground_state.ckpt")
```

## Expected Output

For a 20-site Heisenberg chain with OBC, U(1) symmetry, and bond dimension 64, the run converges in 2 sweeps. Alice logs the sweep progress at INFO level:

```
sweep  1 / 20: E = -8.68247313627, |ΔE| = inf,        dw = 1.65e-11
sweep  2 / 20: E = -8.68247313627, |ΔE| = 1.51e-12,   dw = 1.55e-11
converged after 2 sweep(s)
```

The final energy and per-site energy printed by the script:

```
Sweeps performed : 2  (converged)
Ground-state E   : -8.6824731363
Energy per site  : -0.4341236568

Exact E/N (N→∞)  : -0.4431471806  (thermodynamic limit)
DMRG  E/N        : -0.4341236568
Difference       : +9.0235e-03  (finite-size + truncation error)
```

The ~0.009 gap is expected: for L=20 with OBC the finite-size correction alone accounts for most of the difference with the thermodynamic-limit Bethe-ansatz value of \(E/L = 1/4 - \ln 2 \approx -0.4431\).

## Next Steps

- See the [full DMRG examples](../examples/dmrg/heisenberg.md) for complete initialization code.
- Learn how to load all options from a TOML file in [AutoMPO from TOML](../examples/autompo-toml.md).
- Explore the [API Reference](../api/index.md) for full signatures and options.
