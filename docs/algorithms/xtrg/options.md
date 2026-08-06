# Options

XTRG run options.

::: alice.algorithm.xtrg.Options
    options:
      heading_level: 2

## TOML Loading

`Options` can be loaded directly from a TOML section:

```python
import tomllib
from alice.algorithm import xtrg

with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

opts = xtrg.Options.from_toml(cfg["free_fermion"]["algorithm"])
```

Example TOML block:

```toml
[free_fermion.algorithm]
scheme       = "2s"
tau_0        = 0.000244140625   # 2^{-12}
n_steps      = 20
max_bond     = 64
trunc_thresh = 1e-12
n_sweeps     = 4
```

## See Also

- [Summary](summary.md) — thermodynamic history.
- [Artifact](artifact.md) — density-matrix snapshots.
- [run](run.md) — pass `Options` here.
