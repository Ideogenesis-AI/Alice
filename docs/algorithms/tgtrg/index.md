# TgTRG

Alice's TgTRG algorithm computes finite-temperature thermodynamic properties
of a Hamiltonian MPO by cooling along the tangent space of the MPO manifold.
Read as an imaginary-time TDVP for the density matrix: the exact flow

\[
\frac{\mathrm{d}\rho}{\mathrm{d}\beta} = -H \rho
\]

is projected onto the tangent space at the current \(\rho\), and the
projection splits into local propagators that are applied in a sweep —
\(e^{-\tau H^{(1)}}\) on each site tensor forward, \(e^{+\tau H^{(0)}}\) on
each bond backward.

Where [XTRG](../xtrg/index.md) reaches \(\beta\) by squaring, and so never
refers back to \(H\) after the initial Taylor state, TgTRG takes one sweep per
step and contracts against \(H\) throughout. One sweep costs \(O(D^3)\),
against XTRG's \(O(D^4)\) fit, and the \(\beta\) grid is free: it need not
double.

## API

| Symbol | Description |
|--------|-------------|
| [Options](options.md) | Run options: scheme, grid mode, \(\tau_0\), steps, bond dimension, caching, artifacts |
| [Summary](summary.md) | Thermodynamic history: \(\log Z\), free energy, internal energy, specific heat, entropy |
| [Artifact](artifact.md) | Density-matrix snapshot \(\rho(\beta)\) at one cooling step |
| [run](run.md) | Top-level entry point |

## Usage Pattern

```python
from alice.network import build_interaction, build_hamiltonian
from alice.network.thermal import thermal_mpo
from alice.algorithm import tgtrg

interactions, spc, geo = build_interaction(cfg)
H = build_hamiltonian(interactions, geo.L, spc)

opts = tgtrg.Options(scheme='2s', grid='exp', n_steps=13, max_bond=64)
rho0 = thermal_mpo(H, opts.tau_0, opts.taylor_order, spc)
summary, artifact = tgtrg.run(
    tgtrg.Artifact(rho=rho0, beta=opts.tau_0, step=0), H, opts)

for beta, f in zip(summary.betas, summary.free_energies):
    print(f"β = {beta:.4f},  f = {f:.6f}")
```

Unlike `xtrg.run()`, `tgtrg.run()` takes the Hamiltonian MPO as its second
argument — every step needs it.

Resuming an interrupted run is calling `run()` again with the `Artifact`
loaded from `tgtrg.ckpt`:

```python
resumed = tgtrg.Artifact.load(ckpt_dir / 'tgtrg.ckpt')
summary, artifact = tgtrg.run(resumed, H, opts)  # thermal.ckpt auto-loaded
```

`n_steps` is the absolute step index to stop at, counted from the Taylor
state, not a number of additional steps to perform.

## Update Schemes

| Name | Alias | Description |
|------|-------|-------------|
| `'1s'` | `'1-site'`, `'one-site'` | 1-site propagation with a backward bond update; preserves bond dimension |
| `'2s'` | `'2-site'`, `'two-site'` | 2-site propagation with an SVD split; lets the bond grow |
| `'1sp'` | `'1-site-plus'`, `'one-site-plus'` | 1-site propagation on a bond expanded beforehand by controlled bond expansion; lets the bond grow at close to 1-site cost |

`'1s'` cannot change a bond dimension, so the rank it runs at is the rank of
the state it is handed. That is also how `max_bond` reaches it: the initial
state is compacted to the cap before the first step.

### When to reach for `'1sp'`

`'2s'` is the default and the right choice for most cooling runs. Reach for
`'1sp'` when `'2s'` is too expensive at the rank \(\rho(\beta)\) wants: it
expands each bond ahead of the update by up to `expand_k` directions drawn
from the complement of the space that bond currently spans, cools over the
expanded space, and truncates back. The rank adapts as it does under `'2s'`
while every local propagator stays 1-site sized.

Cost and name are all it shares with `'1s'`. It adapts the rank, truncates,
and reports a discarded weight, and it inherits none of the fixed-rank
exactness that makes `'1s'` worth having — group it with `'2s'` when
reasoning about what it guarantees. The one setting where the two coincide is
`expand_k = 0`, which empties the complement and reduces the sweep to `'1s'`.

| Option | Meaning | Typical value |
|--------|---------|---------------|
| `expand_k` | Complement directions added per bond per half-sweep | \(\mathrm{round}(\sqrt{D})\) |
| `expand_alpha` | Internal bond of the cheap factorization of \(\Theta\) inside the search | \(\approx\) `expand_k` |

`expand_alpha` is what keeps the search below 2-site cost; `None` makes it
exact and no cheaper than `'2s'` per bond. Neither option is consulted by the
other two schemes.

One detail of the implementation is worth knowing about if the expansion is
ever ported or modified: it re-gauges each expanded bond, a step the
real-time version does without. The complement is ill-conditioned wherever a
bond's kept space already nearly exhausts what \(H\) can reach, and a cooling
run starts there. See
[Why `'1sp'` re-gauges after expanding](lessons.md#why-1sp-re-gauges-after-expanding).

## Integrator Order

| `order` | Composition | Error per step |
|---------|-------------|----------------|
| `1` | one forward half-sweep | \(O(\tau^2)\) |
| `2` | forward then backward half-sweep | \(O(\tau^3)\) |

At full rank — when the manifold is the whole space — the projector-splitting
integrator is *exact*, at either order and for any \(\tau\). The step size
only matters once the rank is limited, or once it runs into the arithmetic
below. Even under a rank limit it saturates quickly: the projection error
accrues per unit of \(\beta\), not per step, so refining \(\tau\) reaches a
floor that only `max_bond` can lower.
[Lessons](lessons.md#step-size-versus-bond-dimension) has the numbers.

## Temperature Grid

Two modes, chosen with `Options.grid`. They are never combined within a single
run; a schedule that needs both is two `run()` calls.

| `grid` | Alias | Step | \(\beta\) after step \(n\) |
|--------|-------|------|------------------|
| `'exp'` | `'exponential'`, `'doubling'` | \(\tau = \beta\) | \(2^n \tau_0\) |
| `'lin'` | `'linear'`, `'uniform'` | `Options.tau` | \(\beta_0 + n \tau\) |

The exponential grid covers the high-temperature regime in a handful of steps,
the way XTRG does. The linear grid places steps where the physics is, the one
thing TgTRG buys over squaring.

### Stacking the two grids

A run normally uses both, one after the other: the exponential grid out of
the high-temperature regime where the Taylor state lives, the linear grid
from there on. The exponential step size *is* \(\beta\), so carrying that
mode to low temperature means steps too large for the backward bond update to
resolve, and `run()` warns past a soft limit of \(\tau = 8\).
[Lessons](lessons.md#how-large-a-step-may-be) has the criterion, the
measurements behind it, and the regime the warning does not cover.

```python
shared = dict(scheme='2s', tau_0=2 ** -12, max_bond=64,
              checkpoint_dir=str(ckpt_dir))

# Phase 1: double β from τ₀ up to 2.
summary, artifact = tgtrg.run(
    tgtrg.Artifact(rho=rho0, beta=2 ** -12, step=0), H,
    tgtrg.Options(grid='exp', n_steps=13, **shared))

# Phase 2: continue in steps of 0.5 to β = 12.
summary, artifact = tgtrg.run(
    artifact, H,
    tgtrg.Options(grid='lin', tau=0.5, n_steps=33, **shared))
```

The second call resumes from the first call's `Artifact` and recovers its
history from `thermal.ckpt`, so the returned `Summary` spans both phases.
`Summary.taus` tells the phases apart after the fact.

## Observables

\(\log Z(\beta)\) comes from `NormalMPO.log_trace()` at every step, which is
cheap, and the free energy follows from it directly. Both are accurate to
roundoff when the rank suffices.

The internal energy, specific heat, and entropy are finite differences of
\(\log Z\) on the \(\beta\) grid, so they are only as good as its spacing —
\(c_V\), a second derivative, least of all, and especially across a phase
boundary where the spacing jumps. Setting `Options.obs_exact` adds a direct
measurement of

\[
u = \frac{\mathrm{Tr}[\rho H]}{L \, \mathrm{Tr}[\rho]}
\]

at each step, free of discretization error; it is off by default so that the
cooling pays for nothing it does not need.

## See Also

- [tgtrg.run](run.md) — full parameter reference.
- The method as published: PRL **130**, 226502 (2023), [arXiv:2212.11973](https://arxiv.org/abs/2212.11973).
- [XTRG](../xtrg/index.md) — the squaring alternative, and the source of the thermal data structures.
- [TDVP](../tdvp/index.md) — the real-time algorithm this projects the same way.
- [Free fermion](../../examples/tgtrg/free-fermion.md), [Hubbard model](../../examples/tgtrg/hubbard.md) — worked examples against exact solutions.
- [Scheme comparison](../../examples/tgtrg/scheme-comparison.md) — the three schemes cooled side by side, and how to size `expand_k`.
- [NormalMPO](../../api/network/normal-mpo.md) — the density matrix representation.
- [thermal_mpo](../../api/network/thermal-mpo.md) — used to build the initial `Artifact`.
