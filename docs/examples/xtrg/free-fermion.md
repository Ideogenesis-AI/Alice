# Free Fermion

Finite-temperature thermodynamics via XTRG for a spinless tight-binding chain.

\[
H = -t \sum_{\langle i,j \rangle} (c^\dagger_i c_j + \text{h.c.}) - \mu \sum_i n_i, \quad t = 1, \; \mu = 0
\]

XTRG computes \(\rho(\beta) = e^{-\beta H}\) by repeated squaring, starting from a
small \(\tau_0\) and doubling \(\beta\) at each cooling step. For this
free-fermion chain, the exact grand-canonical \(\log Z\) is known in closed form
and provides a direct accuracy check:

\[
\log Z(\beta) = \sum_{k=1}^{L} \log\bigl(1 + e^{-\beta (\varepsilon_k - \mu)}\bigr),
\quad \varepsilon_k = -2t \cos\!\left(\frac{k\pi}{L+1}\right)
\]

## 1. Setup

```python
import alice
alice.configure_logging()  # optional: write diagnostics to .logging/

from alice.network import build_hamiltonian, build_interaction
from alice.algorithm import xtrg
```

## 2. Build the Hamiltonian

```python
config = {
    "geometry": {"lattice": "chain", "lx": 8, "bcx": "OBC", "n2x": True},
    "model": {"category": "fermionic", "label": "FreeFermion",
              "symmetry": "U1", "t": 1.0, "mu": 0.0},
}
interactions, spc, geo = build_interaction(config)
hamiltonian = build_hamiltonian(interactions, geo.L, spc)
print(f"L = {geo.L}, MPO bond dims: {hamiltonian.bond_dims}")
```

## 3. Run XTRG

```python
opts = xtrg.Options(
    scheme       = '2s',        # 2-site update with SVD bond growth
    tau_0        = 2 ** -12,
    n_steps      = 12,
    taylor_order = 10,
    max_bond     = 32,
    n_sweeps     = 4,
)

summary = xtrg.run(hamiltonian, spc, opts)
```

## 4. Compare with the exact solution

```python
import math

def exact_log_z(L, t, mu, beta):
    eps = [-2.0 * t * math.cos(k * math.pi / (L + 1)) for k in range(1, L + 1)]
    return sum(math.log1p(math.exp(-beta * (e - mu))) for e in eps)

for beta, lz in zip(summary.betas, summary.log_z):
    lz_exact = exact_log_z(geo.L, 1.0, 0.0, beta)
    rel_err = abs(lz - lz_exact) / abs(lz_exact)
    print(f"β = {beta:8.4f},  log Z = {lz:12.8f},  exact = {lz_exact:12.8f},  rel err = {rel_err:.2e}")
```

For `L=8`, `max_bond=32`, the relative error should stay below `1e-8` across the
whole cooling range — at this modest chain length the untruncated bond
dimension never exceeds `~16`, so `max_bond=32` amounts to essentially exact
compression.

## See Also

- [Hubbard model](hubbard.md)
- [xtrg.Options API](../../algorithms/xtrg/options.md)
- [build_free_fermion API](../../api/hamiltonian/build-free-fermion.md)
- [NormalMPO](../../api/network/normal-mpo.md), [thermal_mpo](../../api/network/thermal-mpo.md)
