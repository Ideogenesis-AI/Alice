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
from alice.network.thermal import thermal_mpo
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

rho0 = thermal_mpo(hamiltonian, opts.tau_0, opts.taylor_order, spc)
summary, artifact = xtrg.run(xtrg.Artifact(rho=rho0, beta=opts.tau_0, step=0), opts)
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

## 5. Environment caching for large chains

Each squaring step fits a compressed MPO against two factor MPOs, so XTRG holds
three networks at once and runs out of RAM at shorter chain lengths than DMRG
does. When the environment blocks no longer fit, spill them to disk:

```python
opts = xtrg.Options(
    scheme        = '2s',
    tau_0         = 2 ** -12,
    n_steps       = 20,
    max_bond      = 256,
    env_cache_dir = "/tmp/alice_envs",   # write blocks to disk
    env_window    = 4,                    # keep 4 blocks in memory
    env_async_io  = True,                 # overlap I/O with computation
)
```

The `env_cache_dir` directory is created automatically. Each run then creates
its own uniquely-named subdirectory inside it, so several jobs dispatched to the
same node can share one cache root without overwriting each other's blocks. That
subdirectory is reused by every squaring step of the run and removed when
`run()` finishes or raises, leaving `env_cache_dir` itself empty.

Peak memory then scales with `env_window` rather than with chain length, at the
cost of one serialize/deserialize round trip per block. Leave `env_async_io`
enabled so those writes overlap with the variational sweeps.

## See Also

- [Hubbard model](hubbard.md)
- [xtrg.Options API](../../algorithms/xtrg/options.md)
- [build_free_fermion API](../../api/hamiltonian/build-free-fermion.md)
- [NormalMPO](../../api/network/normal-mpo.md), [thermal_mpo](../../api/network/thermal-mpo.md)
