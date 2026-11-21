# Options

TgTRG run options.

::: alice.algorithm.tgtrg.Options
    options:
      heading_level: 2

## TOML Loading

`Options` can be loaded directly from a TOML section:

```python
import tomllib
from alice.algorithm import tgtrg

with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

opts = tgtrg.Options.from_toml(cfg["free_fermion"]["algorithm"])
```

Every option is a scalar or a string, so a stacked schedule is two sections:

```toml
[free_fermion.cooling_exp]
scheme       = "2s"
grid         = "exp"
tau_0        = 0.000244140625   # 2^{-12}
n_steps      = 13               # -> beta = 2
order        = 2
max_bond     = 64
trunc_thresh = 1e-12

[free_fermion.cooling_lin]
scheme       = "2s"
grid         = "lin"
tau_0        = 0.000244140625
tau          = 0.5
n_steps      = 33               # -> beta = 2 + 31 * 0.5
order        = 2
max_bond     = 64
trunc_thresh = 1e-12
```

`tau_0` is carried into the linear section because the recovered history was
built on it; `n_steps` counts from the Taylor state in both.

## See Also

- [Overview](index.md) — grid modes, step size, and the stacked schedule.
- [Lessons](lessons.md) — how large a step may be, and why `max_bond` is the knob that moves the floor.
- [Summary](summary.md) — thermodynamic history.
- [Artifact](artifact.md) — density-matrix snapshots.
- [run](run.md) — pass `Options` here.
