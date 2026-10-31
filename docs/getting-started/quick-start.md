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

The two tables are top level here because the file path goes straight to
`build_interaction`. To keep several models in one file, nest them under a
name each (`[heisenberg.geometry]`, `[heisenberg.model]`), load the file with
`tomllib`, and pass the section you want.

### 3. Build interactions and Hamiltonian

```python
from alice import build_interaction, build_hamiltonian

interactions, spc, geo = build_interaction("heisenberg.toml")
L = geo.L
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

# Random MPS (bond_dim>1) — group-derived bond sectors
# mps = init_mps(L, Spc, Op, bond_dim=32)
```

Neither state carries the bond sectors of the ground state, so pair it with
the 2-site (`'2s'`) or CBE (`'1sp'`) scheme below. The 1-site scheme never
changes those sectors, and would converge above the true energy.

See [init_mps](../api/network/init-mps.md) for the full signature and all
supported symmetry groups.

### 5. Run DMRG

```python
from alice import dmrg

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

For a 20-site Heisenberg chain with OBC, U(1) symmetry, and bond dimension 64,
the run converges in 4 sweeps from the product state. Alice logs the sweep
progress at INFO level:

```
sweep  1 / 20: E = -8.60547015119, |ΔE| = inf, dw = 0.0000e+00
sweep  2 / 20: E = -8.68225912573, |ΔE| = 7.6789e-02, dw = 0.0000e+00
sweep  3 / 20: E = -8.6824731357, |ΔE| = 2.1401e-04, dw = 0.0000e+00
sweep  4 / 20: E = -8.68247313627, |ΔE| = 5.7109e-10, dw = 6.6697e-14
converged after 4 sweep(s)
```

And the script itself prints:

```
Ground-state energy: -8.6824731363
Converged: True after 4 sweeps
```

That is `E/L = -0.4341236568` per site, about 0.009 above the
thermodynamic-limit Bethe-ansatz value \(E/L = 1/4 - \ln 2 \approx -0.4431\).
The gap is expected: for L=20 with OBC the finite-size correction alone
accounts for most of it.

## Next Steps

- See the [full DMRG examples](../examples/dmrg/heisenberg.md) for complete initialization code.
- Learn how to load all options from a TOML file in [AutoMPO from TOML](../examples/autompo-toml.md).
- Explore the [API Reference](../api/index.md) for full signatures and options.
