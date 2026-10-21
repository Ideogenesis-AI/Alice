# Options

TDVP run options.

::: alice.algorithm.tdvp.Options
    options:
      heading_level: 2

## TOML Loading

`Options` can be loaded directly from a TOML section:

```python
import tomllib
from alice.algorithm import tdvp

with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

opts = tdvp.Options.from_toml(cfg["quench"]["algorithm"])
```

Example TOML block:

```toml
[quench.algorithm]
scheme               = "2s"
dt                   = 0.05
n_steps              = 40
order                = 2
max_bond             = 64
trunc_thresh         = 1e-12
expand_k             = 8
expand_alpha         = 8
krylov_dim           = 16
save_artifacts_every = 4
```

## Choosing `dt` and `max_bond`

The two parameters fail in opposite directions and are best converged
separately.

Truncation error is set by `max_bond` and is independent of `dt`, so a run
limited by bond dimension shows no improvement when the step is halved.
Integration error scales as `dt²` for `order=2` and is independent of
`max_bond`. A scan that halves `dt` and sees the error drop by four is
integration-limited; one that sees no change is truncation-limited and needs a
larger `max_bond`.

Entanglement grows roughly linearly in time after a quench, so the bond
dimension needed grows exponentially. This, not the step size, is what sets the
reachable time horizon.

## Choosing `expand_k`

`expand_k` matters only for `scheme='1sp'`, where it caps how many directions a
bond may gain per local update. The cap is an upper bound and frequently not
the binding one: what a bond receives is however many directions its complement
holds above the scale the discarded-space projection cancelled on, and for a
short-range Hamiltonian that is typically one or two. A typical choice for the
cap is `round(sqrt(max_bond))`.

Too small a value is visible as a `max_bonds` history that climbs step by step
over a stretch where the physics is not changing that fast, and as observables
that improve when `expand_k` is raised at fixed `max_bond` and `dt`. Too large
a value costs a slightly bigger SVD per update and changes nothing else — if
raising it does not move the answer, the complement, not the cap, was the
limit. Setting `expand_k = 0` disables the expansion and reproduces `'1s'`
exactly, which is a useful control when attributing an error to the expansion
rather than to the integrator.

`expand_alpha` truncates the two-site tensor the complement search is built
from, which is what keeps the search cheaper than a 2-site step; `None` leaves
the bond intact, making the search exact and the step no cheaper than `'2s'`.
A typical choice is `round(sqrt(max_bond))`, the same scale as `expand_k`.

## See Also

- [Summary](summary.md) — evolution history.
- [Artifact](artifact.md) — state snapshots.
- [run](run.md) — pass `Options` here.
