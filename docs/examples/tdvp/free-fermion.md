# Free Fermion

Real-time propagation of a Gaussian wavepacket on a spinless tight-binding
chain — the quantitative accuracy benchmark for TDVP, since the exact answer is
available in closed form.

\[
H = -t \sum_{\langle i,j \rangle} (c^\dagger_i c_j + \text{h.c.}), \quad t = 1
\]

## Why a single particle is the right benchmark

A one-particle state on a free chain is a vector in an \(L\)-dimensional
single-particle space, so the full many-body evolution reduces to
exponentiating an \(L \times L\) hopping matrix:

\[
\psi(t) = e^{-iht}\,\psi(0), \qquad h_{j,j\pm1} = -t, \qquad n_j(t) = |\psi_j(t)|^2
\]

That reference is exact to machine precision at any time, with no bond
dimension and no step size, so every digit of disagreement belongs to TDVP.

The choice of a *moving* packet rather than a static configuration is
deliberate. Conservation laws and overlap magnitudes are blind to the sign of
the propagator: `exp(+iHt)` conserves energy and norm just as well. A packet
with mean momentum \(k_0 > 0\) travels right under `exp(-iHt)` and left under
`exp(+iHt)`, at group velocity

\[
v = \frac{\partial \varepsilon}{\partial k} = 2t \sin k_0,
\qquad \varepsilon(k) = -2t\cos k
\]

so the direction of motion is itself a test of the sign convention.

## 1. Setup

```python
import alice
alice.configure_logging()  # optional: write diagnostics to .logging/

from alice.network import build_hamiltonian, build_interaction
from alice.algorithm import tdvp
```

## 2. Build the Hamiltonian

```python
config = {
    "geometry": {"lattice": "chain", "lx": 24, "bcx": "OBC", "n2x": True},
    "model": {"category": "fermionic", "label": "FreeFermion",
              "symmetry": "U1", "t": 1.0, "mu": 0.0},
}
interactions, spc, geo = build_interaction(config)
hamiltonian = build_hamiltonian(interactions, geo.L, spc)
```

## 3. Build the wavepacket

A single-particle superposition \(\sum_j \psi_j c^\dagger_j |0\rangle\) is an
MPS of bond dimension exactly 2 — one bond state carrying "the particle is
still ahead" and one carrying "the particle is already behind". The
`wavepacket_mps` helper in
[`freefermion_exact.py`](https://github.com/Ideogenesis-AI/Alice/blob/main/examples/examples_tdvp/freefermion_exact.py)
builds it by hand:

```python
from freefermion_exact import gaussian_wavepacket, wavepacket_mps

psi = gaussian_wavepacket(L=24, x0=6.0, k0=0.25 * 3.141592653589793, sigma=2.0)
state = wavepacket_mps(psi, spc, ops)
```

Bond dimension 2 is not an approximation and does not grow: a one-particle
state stays a one-particle state, so TDVP has nothing to truncate here and the
only error source is the time step.

## 4. Evolve

```python
opts = tdvp.Options(scheme='2s', dt=0.05, n_steps=60, order=2, max_bond=32)
summary, artifact = tdvp.run(
    tdvp.Artifact(state=state, time=0.0, step=0), hamiltonian, opts)
```

## 5. Compare with the exact propagator

```python
from freefermion_exact import (exact_single_particle, mps_occupations,
                               occupation_mpos)

obs = occupation_mpos(geo.L, spc, ops)
n_tdvp = mps_occupations(artifact.state, obs)
n_exact = exact_single_particle(psi, 1.0, artifact.time)  # |psi_j(t)|^2

print(f"max |n_j - exact| = {abs(n_tdvp - n_exact).max():.3e}")
```

## Running the script

```bash
uv run python examples/examples_tdvp/tdvp_freefermion.py --L 24 --time 3.0 --dt 0.05
```

Abbreviated output:

```text
Group velocity : +1.414214  (packet moves right)
Initial MPS bond dims: [2, 2, 2, ..., 2]

Initial state check
  n_j vs |psi_j|^2 : 2.165e-15
  energy           : -1.328522537394  (exact -1.328522537394)

      t       <x>     exact     n_j err     E drift        N-1  density profile
------------------------------------------------------------------------------------
  0.750    6.9964    6.9964    1.06e-06   -1.73e-13   +1.5e-13  |    .-#%*-.             |
  1.500    7.9928    7.9928    3.28e-06   -2.40e-13   +1.9e-13  |     .-*%*-.            |
  2.250    8.9893    8.9892    6.20e-06   -2.47e-13   +2.0e-13  |     .:-*#*=:.          |
  3.000    9.9857    9.9856    9.44e-06   -2.39e-13   +1.9e-13  |      .:-+**=:.         |

Max |n_j - exact| over all sites : 9.438e-06
Center of mass  TDVP : 9.985681
Center of mass  exact: 9.985635
Energy  : -1.328522537395  (exact -1.328522537394, conserved)
Bond dims : [2, 2, 2, ..., 2]
```

The packet moves right at the predicted velocity, the center of mass tracks the
exact value to five digits, and the bond dimension stays at 2 throughout. The
ASCII density profile makes the motion and the spreading visible at a glance.

!!! note "The ballistic estimate is only approximate"

    The script also prints `x0 + v·t`, which at \(t = 3\) reads 10.243 against
    a measured 9.986. The gap is physical, not an error: a packet of finite
    width samples a range of momenta, and the slower components drag the mean
    position behind the peak velocity. The exact propagator agrees with TDVP,
    not with the estimate.

## Time-step convergence

Since there is nothing to truncate, this example isolates the integrator error
cleanly:

```bash
uv run python examples/examples_tdvp/tdvp_freefermion.py --dt-scan --L 20 --time 1.0
```

```text
  steps         dt    max |n_j - exact|    ratio
--------------------------------------------------
     10    0.10000            6.857e-06
     20    0.05000            1.713e-06     4.00
     40    0.02500            4.281e-07     4.00
     80    0.01250            1.070e-07     4.00
```

Ratios of exactly 4.00 across three halvings confirm the `O(dt²)` global error
of the second-order integrator.

This clean scaling depends on the manifold being genuinely restricted. At full
bond dimension the tangent-space projector is the identity and TDVP is exact at
any step size, so the same scan there returns noise at the 1e-13 level with no
meaningful ratio.

## See Also

- [Domain wall](domain-wall.md) — a many-body state where entanglement actually grows.
- [tdvp.Options](../../algorithms/tdvp/options.md) — parameter reference.
- [DMRG free fermion](../dmrg/free-fermion.md) — ground state of the same Hamiltonian.
