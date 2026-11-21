# Hubbard Model

Finite-temperature thermodynamics via TgTRG for the 1D Hubbard chain with
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
closed-form `log Z`, so only the `U=0` point can be checked this way. The full
script is
[`examples/examples_tgtrg/tgtrg_spinful.py`](https://github.com/ideogenesis-ai/Alice/blob/main/examples/examples_tgtrg/tgtrg_spinful.py),
which also reports thermodynamics at finite `U`.

## 1. Setup

```python
import alice
alice.configure_logging()  # optional: write diagnostics to .logging/

from alice.network import build_hamiltonian, build_interaction
from alice.network.thermal import thermal_mpo
from alice.algorithm import tgtrg
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

## 3. Cool in two phases

```python
shared = dict(scheme='2s', order=2, tau_0=2 ** -12, taylor_order=10,
              max_bond=64, trunc_thresh=1e-12, obs_exact=True)

rho0 = thermal_mpo(hamiltonian, 2 ** -12, 10, spc)
state = tgtrg.Artifact(rho=rho0, beta=2 ** -12, step=0)

# Phase 1: double beta from tau_0 up to 2.
summary, artifact = tgtrg.run(
    state, hamiltonian, tgtrg.Options(grid='exp', n_steps=13, **shared))

# Phase 2: continue in steps of 0.5 up to beta = 6.
summary, artifact = tgtrg.run(
    artifact, hamiltonian,
    tgtrg.Options(grid='lin', tau=0.5, n_steps=21, **shared))
```

`obs_exact` is worth switching on here: a doubled physical space fills a bond
faster than the spinless chain does, so the run reaches `max_bond` at a
warmer temperature, and the direct `Tr[ρH]` measurement is then the honest
internal energy to quote.

## 4. Compare with the exact `U=0` solution

```python
import math

def exact_log_z_spinless(L, t, mu, beta):
    eps = [-2.0 * t * math.cos(k * math.pi / (L + 1)) for k in range(1, L + 1)]
    return sum(math.log1p(math.exp(-beta * (e - mu))) for e in eps)

def exact_log_z_spinful(L, t, mu, beta):
    return 2.0 * exact_log_z_spinless(L, t, mu, beta)

for beta, lz, D in zip(summary.betas, summary.log_z, summary.max_bonds):
    lz_exact = exact_log_z_spinful(geo.L, 1.0, 0.0, beta)
    rel_err = abs(lz - lz_exact) / abs(lz_exact)
    print(f"β = {beta:8.4f},  log Z = {lz:12.8f},  exact = {lz_exact:12.8f},  "
          f"rel err = {rel_err:.2e},  D = {D}")
```

Printing `max_bonds` alongside the error shows the two regimes: while the bond
is free to grow, the error sits at roundoff; once it saturates at `max_bond`,
the error tracks the discarded weight instead. `summary.discarded_weights` is
the quantity to watch from that point on.

## 5. Interacting case (`U ≠ 0`)

```python
config["model"]["U"] = 4.0
interactions, spc, geo = build_interaction(config)
hamiltonian = build_hamiltonian(interactions, geo.L, spc)

rho0_u4 = thermal_mpo(hamiltonian, 2 ** -12, 10, spc)
summary_u4, artifact_u4 = tgtrg.run(
    tgtrg.Artifact(rho=rho0_u4, beta=2 ** -12, step=0), hamiltonian,
    tgtrg.Options(grid='exp', n_steps=13, **shared))
```

No closed-form `log Z` reference exists for `U != 0`. Convergence has to be
established from the run itself: `discarded_weights` against `max_bond`, and
the agreement between the finite-difference `energies` and the directly
measured `exact_energies`.

## See Also

- [Free fermion](free-fermion.md)
- [tgtrg overview](../../algorithms/tgtrg/index.md), [tgtrg.Options API](../../algorithms/tgtrg/options.md)
- [XTRG Hubbard model](../xtrg/hubbard.md) — the same chain by repeated squaring.
- [build_hubbard API](../../api/hamiltonian/build-hubbard.md)
- [build_conductor API](../../api/local-space/build-conductor.md)
