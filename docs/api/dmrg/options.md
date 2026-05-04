# Options

DMRG run options.

::: alice.algorithm.dmrg.Options
    options:
      heading_level: 2

## TOML loading

`Options` can be loaded directly from an `[algorithm]` TOML section:

```python
import tomllib
from alice import dmrg

with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

opts = dmrg.Options.from_toml(cfg["heisenberg"]["algorithm"])
```

Example TOML block:

```toml
[heisenberg.algorithm]
scheme        = "2s"
n_sweeps      = 30
max_bond      = 128
trunc_thresh  = 1e-10
e_tol         = 1e-8
```

## See Also

- [Summary](summary.md) — output dataclass.
- [run](run.md) — pass `Options` here.
