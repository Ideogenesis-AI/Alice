# Free Fermion

Finite-temperature thermodynamics via TgTRG for a spinless tight-binding chain.

\[
H = -t \sum_{\langle i,j \rangle} (c^\dagger_i c_j + \text{h.c.}) - \mu \sum_i n_i, \quad t = 1, \; \mu = 0
\]

TgTRG cools \(\rho(\beta) = e^{-\beta H}\) along the tangent space of the MPO
manifold, one sweep per step, starting from a small \(\tau_0\). For this
free-fermion chain the exact grand-canonical solution is known in closed form
and provides a direct accuracy check:

\[
\log Z(\beta) = \sum_{k=1}^{L} \log\bigl(1 + e^{-\beta (\varepsilon_k - \mu)}\bigr),
\quad \varepsilon_k = -2t \cos\!\left(\frac{k\pi}{L+1}\right)
\]

\[
u(\beta) = \frac{1}{L} \sum_k \frac{\varepsilon_k - \mu}{1 + e^{\beta (\varepsilon_k - \mu)}},
\qquad
c_V(\beta) = \frac{\beta^2}{L} \sum_k \frac{(\varepsilon_k - \mu)^2}{4 \cosh^2\!\bigl(\tfrac{1}{2}\beta(\varepsilon_k - \mu)\bigr)}
\]

The full script is
[`examples/examples_tgtrg/tgtrg_spinless.py`](https://github.com/ideogenesis-ai/Alice/blob/main/examples/examples_tgtrg/tgtrg_spinless.py).

## 1. Setup

```python
import alice
alice.configure_logging()  # optional: write diagnostics to .logging/

from alice.network import build_hamiltonian, build_interaction
from alice.network.thermal import thermal_mpo
from alice.algorithm import tgtrg
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

## 3. Cool in two phases

The exponential phase gets out of the high-temperature regime cheaply; the
linear phase then takes the chain cold in controlled steps. Stacking the two
keeps the step size bounded — see
[step size](../../algorithms/tgtrg/lessons.md#how-large-a-step-may-be).

```python
shared = dict(scheme='2s', order=2, tau_0=2 ** -12, taylor_order=10,
              max_bond=64, trunc_thresh=1e-12)

rho0 = thermal_mpo(hamiltonian, 2 ** -12, 10, spc)
state = tgtrg.Artifact(rho=rho0, beta=2 ** -12, step=0)

# Phase 1: double beta from tau_0 up to 2.
summary, artifact = tgtrg.run(
    state, hamiltonian, tgtrg.Options(grid='exp', n_steps=13, **shared))

# Phase 2: continue in steps of 0.5 up to beta = 8.
summary, artifact = tgtrg.run(
    artifact, hamiltonian,
    tgtrg.Options(grid='lin', tau=0.5, n_steps=25, **shared))
```

The returned `summary` spans both phases: the second call recovers the first
one's history from `thermal.ckpt` in the checkpoint directory.

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

At `L=8` the untruncated bond dimension stays well inside `max_bond=64`, so
the relative error on `log Z` holds near `1e-13` across the whole range. The
free energy, computed from `log Z` directly, is just as good.

## 5. Observables beyond log Z

`u`, `c_V`, and `S` are finite differences on the β grid, so they carry its
spacing as an error — visibly so for `c_V`, and worst at the junction between
the two phases, where the spacing jumps. Switch on `obs_exact` to get the
internal energy measured directly instead:

```python
opts = tgtrg.Options(grid='lin', tau=0.5, n_steps=25, obs_exact=True, **shared)
summary, artifact = tgtrg.run(artifact, hamiltonian, opts)

for beta, u_fd, u_direct in zip(summary.betas, summary.energies,
                                summary.exact_energies):
    print(f"β = {beta:8.4f},  u (differences) = {u_fd:10.6f},  "
          f"u (direct) = {u_direct:10.6f}")
```

The gap between the two columns *is* the discretization error, which makes
this the cheapest way to decide whether a grid is fine enough.

## 6. Environment caching for large chains

Each step sweeps across environment blocks built from
`(bra_ρ, H, ket_ρ)`. When they no longer fit in RAM, spill them to disk:

```python
opts = tgtrg.Options(
    scheme        = '2s',
    grid          = 'lin',
    tau           = 0.5,
    n_steps       = 60,
    max_bond      = 256,
    env_cache_dir = "/tmp/alice_envs",   # write blocks to disk
    env_window    = 4,                    # keep 4 blocks in memory
    env_async_io  = True,                 # overlap I/O with computation
)
```

The `env_cache_dir` directory is created automatically, and each run gets its
own uniquely-named subdirectory inside it, so several jobs on one node can
share a cache root. Peak memory then scales with `env_window` instead of chain
length, at the cost of one serialize/deserialize round trip per block.

## See Also

- [Hubbard model](hubbard.md)
- [tgtrg overview](../../algorithms/tgtrg/index.md), [tgtrg.Options API](../../algorithms/tgtrg/options.md)
- [XTRG free fermion](../xtrg/free-fermion.md) — the same chain by repeated squaring.
- [NormalMPO](../../api/network/normal-mpo.md), [thermal_mpo](../../api/network/thermal-mpo.md)
