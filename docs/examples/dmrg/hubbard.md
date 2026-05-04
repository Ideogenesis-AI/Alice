# Hubbard Model

Ground-state DMRG for the 1D Hubbard model with U(1)×SU(2) symmetry.

\[
H = -t \sum_{\langle i,j \rangle, \sigma} (c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.})
    + U \sum_i n_{i\uparrow} n_{i\downarrow}
\]

At half-filling (`µ = 0`) the Hubbard model is a Mott insulator for any `U > 0`. Its ground-state energy is known exactly via the Bethe ansatz.

## 1. Setup

```python
import alice
alice.configure_logging()

from alice import build_interaction, build_hamiltonian, MPS, dmrg
```

## 2. Build the Hamiltonian

U(1)×SU(2) symmetry conserves total particle number and total spin simultaneously, achieving the best compression ratio:

```python
config = {
    "geometry": {"lattice": "chain", "lx": 8, "bcx": "OBC", "n2x": True},
    "model": {"category": "conductor", "label": "Hubbard",
              "symmetry": "U1,SU2", "t": 1.0, "U": 4.0, "mu": 0.0},
}
interactions, spc, L = build_interaction(config)
hamiltonian = build_hamiltonian(interactions, L, spc)
print(f"L = {L}, MPO bond dims: {hamiltonian.bond_dims}")
```

Alternatively, use U(1)×U(1) to conserve spin-up and spin-down counts separately:

```python
config["model"]["symmetry"] = "U1,U1"
```

## 3. Build an initial MPS

For U(1)×SU(2), the product-state initialization targets the half-filled singlet sector (total particle number `L`, total spin `S=0`). Building the initial MPS correctly for non-Abelian symmetry is non-trivial; it is typically done via iterative diagonalization (see `examples/examples_dmrg/dmrg_heisenberg.py` for a reference implementation).

A simpler approach is to start from a 2-site DMRG sweep with a small random MPS:

```python
# For a demonstration, use the 2-site scheme to grow bonds from a product state.
opts = dmrg.Options(
    scheme   = '2s',
    n_sweeps = 30,
    max_bond = 128,
    e_tol    = 1e-8,
)
```

## 4. Run DMRG

```python
summary = dmrg.run(mps, hamiltonian, opts)

print(f"Energy:        {summary.energy:.10f}")
print(f"Energy/site:   {summary.energy / L:.10f}")
print(f"Converged:     {summary.converged}")
print(f"Max bond dim:  {max(summary.bond_dims)}")
```

For `L=8`, `U=4`, `t=1`, OBC, the ground-state energy is approximately `E ≈ -3.67`.

## 5. Environment caching for large chains

For long chains where all environment blocks do not fit in RAM, enable disk caching:

```python
opts = dmrg.Options(
    scheme        = '2s',
    n_sweeps      = 20,
    max_bond      = 256,
    env_cache_dir = "/tmp/alice_envs",   # write blocks to disk
    env_window    = 4,                    # keep 4 blocks in memory
    env_async_io  = True,                 # overlap I/O with computation
)
```

The `env_cache_dir` directory is created automatically. Each block is stored as a `.pt` file and re-loaded transparently as the sweep progresses.

## See Also

- [Heisenberg chain](heisenberg.md), [Free fermion](free-fermion.md)
- [build_hubbard API](../../api/hamiltonian/build-hubbard.md)
- [build_conductor API](../../api/local-space/build-conductor.md)
- [dmrg.Options](../../api/dmrg/options.md)
