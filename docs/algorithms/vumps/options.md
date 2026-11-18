# Options

VUMPS run options.

::: alice.algorithm.vumps.Options
    options:
      heading_level: 2

## TOML Loading

`Options` can be loaded directly from an `[algorithm]` TOML section:

```python
import tomllib
from alice import vumps

with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

opts = vumps.Options.from_toml(cfg["heisenberg"]["algorithm"])
```

Example TOML block:

```toml
[heisenberg.algorithm]
scheme        = "2s"
n_iters       = 50
max_bond      = 32
trunc_thresh  = 1e-15
grad_tol      = 1e-10
env_tol       = 1e-8
```

Under `scheme = "1sp"`, two further keys govern the bond expansion:

```toml
[heisenberg.algorithm]
scheme        = "1sp"
max_bond      = 32
expand_k      = 6
expand_alpha  = 6
```

## See Also

- [Summary](summary.md) — output dataclass.
- [run](run.md) — pass `Options` here.
