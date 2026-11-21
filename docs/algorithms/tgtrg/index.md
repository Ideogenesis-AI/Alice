# TgTRG

Alice's TgTRG algorithm computes finite-temperature thermodynamic properties
of a Hamiltonian MPO by cooling along the tangent space of the MPO manifold.
Read as an imaginary-time TDVP for the density matrix: the exact flow

```
dρ/dβ = -H ρ
```

is projected onto the tangent space at the current ρ, and the projection
splits into local propagators that are applied in a sweep — `exp(-τ H⁽¹⁾)` on
each site tensor forward, `exp(+τ H⁽⁰⁾)` on each bond backward.

Where [XTRG](../xtrg/index.md) reaches β by squaring, and so never refers back
to `H` after the initial Taylor state, TgTRG takes one sweep per step and
contracts against `H` throughout. One sweep costs `O(D³)`, against XTRG's
`O(D⁴)` fit, and the β grid is free: it need not double.

## API

| Symbol | Description |
|--------|-------------|
| [Options](options.md) | Run options: scheme, grid mode, τ₀, steps, bond dimension, caching, artifacts |
| [Summary](summary.md) | Thermodynamic history: log Z, free energy, internal energy, specific heat, entropy |
| [Artifact](artifact.md) | Density-matrix snapshot `ρ(β)` at one cooling step |
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
summary, artifact = tgtrg.run(resumed, H, opts)  # cooling.ckpt auto-loaded
```

`n_steps` is the absolute step index to stop at, counted from the Taylor
state, not a number of additional steps to perform.

## Update Schemes

| Name | Alias | Description |
|------|-------|-------------|
| `'1s'` | `'1-site'`, `'one-site'` | 1-site propagation with a backward bond update; preserves bond dimension |
| `'2s'` | `'2-site'`, `'two-site'` | 2-site propagation with an SVD split; lets the bond grow |

`'1s'` cannot change a bond dimension, so the rank it runs at is the rank of
the state it is handed. That is also how `max_bond` reaches it: the initial
state is compacted to the cap before the first step.

`'1sp'` (1-site-plus / controlled bond expansion), as in XTRG and TDVP, is
accepted by `Options` for forward compatibility and rejected by `run()` with
`NotImplementedError`; it lands in a follow-up release.

## Integrator Order

| `order` | Composition | Error per step |
|---------|-------------|----------------|
| `1` | one forward half-sweep | `O(τ²)` |
| `2` | forward then backward half-sweep | `O(τ³)` |

At full rank — when the manifold is the whole space — the projector-splitting
integrator is *exact*, at either order and for any τ. The step size only
matters once the rank is limited, or once it runs into the arithmetic below.
Even under a rank limit it saturates quickly: the projection error accrues per
unit of β, not per step, so refining τ reaches a floor that only `max_bond`
can lower. [Lessons](lessons.md#step-size-versus-bond-dimension) has the
numbers.

## Temperature Grid

Two modes, chosen with `Options.grid`. They are never combined within a single
run; a schedule that needs both is two `run()` calls.

| `grid` | Alias | Step | β after step `n` |
|--------|-------|------|------------------|
| `'exp'` | `'exponential'`, `'doubling'` | `τ = β` | `2^n τ₀` |
| `'lin'` | `'linear'`, `'uniform'` | `τ = Options.tau` | `β₀ + n τ` |

The exponential grid covers the high-temperature regime in a handful of steps,
the way XTRG does. The linear grid places steps where the physics is, the one
thing TgTRG buys over squaring.

### Step size, and the stacked schedule

The backward bond update propagates by `exp(+τ H⁽⁰⁾)`, amplifying the
high-energy end of the local spectrum. Once that amplification exceeds the
inverse of float64 resolution, what it amplifies is roundoff. Since `order=2`
propagates in half-steps, the criterion is on the exponent of one local
propagation:

```
(τ / 2) ΔE  <  log(1 / ε)  ≈  36
```

with `ΔE` the spectral range of the Hamiltonian — so the limit is on the
exponent, not on τ and not on β, and a chain with a wider spectrum reaches it
at a proportionally smaller step. The exponential grid walks into it on its
own, its step size *being* β.

Measured on an `L = 4` free-fermion chain: reaching β = 64 from β = 32 in one
τ = 32 step loses eight digits of log Z, while four τ = 8 steps over the same
interval are exact to 1e-14. Nothing downstream flags the bad step; the Krylov
expansions converge to 1e-13 either way. `run()` warns past a fixed soft limit
of τ = 8, which is calibrated for a spectral range of order a few and is not
universal — [Lessons](lessons.md#how-large-a-step-may-be) has the measurements
and the regime the warning does not cover.

So the schedule to use is the stacked one — exponential out of the
high-temperature regime, linear from there on:

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
history from `cooling.ckpt`, so the returned `Summary` spans both phases.
`Summary.taus` tells the phases apart after the fact.

## Observables

`log Z(β)` comes from `NormalMPO.log_trace()` at every step, which is cheap,
and the free energy follows from it directly. Both are accurate to roundoff
when the rank suffices.

The internal energy, specific heat, and entropy are finite differences of
`log Z` on the β grid, so they are only as good as its spacing — `c_V`, a
second derivative, least of all, and especially across a phase boundary where
the spacing jumps. Setting `Options.obs_exact` adds a direct measurement of
`u = Tr[ρH] / (Tr[ρ] L)` at each step, free of discretization error; it is off
by default so that the cooling pays for nothing it does not need.

## See Also

- [tgtrg.run](run.md) — full parameter reference.
- The method as published: PRL **130**, 226502 (2023), [arXiv:2212.11973](https://arxiv.org/abs/2212.11973).
- [XTRG](../xtrg/index.md) — the squaring alternative, and the source of the thermal data structures.
- [TDVP](../tdvp/index.md) — the real-time algorithm this projects the same way.
- [Free fermion](../../examples/tgtrg/free-fermion.md), [Hubbard model](../../examples/tgtrg/hubbard.md) — worked examples against exact solutions.
- [NormalMPO](../../api/network/normal-mpo.md) — the density matrix representation.
- [thermal_mpo](../../api/network/thermal-mpo.md) — used to build the initial `Artifact`.
