# Hubbard Model

Finite-temperature thermodynamics via XTRG for the 1D Hubbard chain with
Z(2)×SU(2) symmetry.

\[
H = -t \sum_{\langle i,j \rangle, \sigma} (c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.})
    + U \sum_i n_{i\uparrow} n_{i\downarrow}
\]

At `U = 0` the spin-up and spin-down channels decouple, and each channel is
identical to a spinless free-fermion chain at the same chemical potential, so

\[
\log Z_{\text{spinful}}(\beta) = 2 \times \log Z_{\text{spinless}}(\beta)
\]

giving an exact reference to validate against. For `U ≠ 0` there is no
closed-form `log Z`, so only the `U=0` point can be checked this way — see
`examples/examples_xtrg/xtrg_spinful.py` for a runnable script that also
reports thermodynamics at finite `U`.

## 1. Setup

```python
import alice
alice.configure_logging()  # optional: write diagnostics to .logging/

from alice.network import build_hamiltonian, build_interaction
from alice.algorithm import xtrg
```

## 2. Build the Hamiltonian

Z(2)×SU(2) conserves fermion parity and total spin, giving the best
compression ratio for this model:

```python
config = {
    "geometry": {"lattice": "chain", "lx": 6, "bcx": "OBC", "n2x": True},
    "model": {"category": "conductor", "label": "Hubbard",
              "symmetry": "Z2,SU2", "t": 1.0, "U": 0.0, "mu": 0.0},
}
interactions, spc, geo = build_interaction(config)
hamiltonian = build_hamiltonian(interactions, geo.L, spc)
print(f"L = {geo.L}, MPO bond dims: {hamiltonian.bond_dims}")
```

## 3. Run XTRG

```python
opts = xtrg.Options(
    scheme       = '2s',
    tau_0        = 2 ** -12,
    n_steps      = 10,
    taylor_order = 10,
    max_bond     = 32,
    n_sweeps     = 4,
)

summary = xtrg.run(hamiltonian, spc, opts)
```

## 4. Compare with the exact `U=0` solution

```python
import math

def exact_log_z_spinless(L, t, mu, beta):
    eps = [-2.0 * t * math.cos(k * math.pi / (L + 1)) for k in range(1, L + 1)]
    return sum(math.log1p(math.exp(-beta * (e - mu))) for e in eps)

def exact_log_z_spinful(L, t, mu, beta):
    return 2.0 * exact_log_z_spinless(L, t, mu, beta)

for beta, lz in zip(summary.betas, summary.log_z):
    lz_exact = exact_log_z_spinful(geo.L, 1.0, 0.0, beta)
    rel_err = abs(lz - lz_exact) / abs(lz_exact)
    print(f"β = {beta:8.4f},  log Z = {lz:12.8f},  exact = {lz_exact:12.8f},  rel err = {rel_err:.2e}")
```

## 5. Interacting case (`U ≠ 0`)

```python
config["model"]["U"] = 4.0
interactions, spc, geo = build_interaction(config)
hamiltonian = build_hamiltonian(interactions, geo.L, spc)

summary_u4 = xtrg.run(hamiltonian, spc, opts)
# No closed-form log Z reference exists for U != 0; compare thermodynamics
# (free energy, internal energy, entropy) against the U=0 point for context.
```

## See Also

- [Free fermion](free-fermion.md)
- [xtrg.Options API](../../api/xtrg/options.md)
- [build_hubbard API](../../api/hamiltonian/build-hubbard.md)
- [build_conductor API](../../api/local-space/build-conductor.md)
