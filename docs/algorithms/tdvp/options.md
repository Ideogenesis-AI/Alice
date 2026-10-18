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

## See Also

- [Summary](summary.md) — evolution history.
- [Artifact](artifact.md) — state snapshots.
- [run](run.md) — pass `Options` here.
