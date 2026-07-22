# Options

Two-site TDVP run options.

::: alice.algorithm.tdvp2.Options
    options:
      heading_level: 2

## TOML Loading

`Options` can be loaded directly from an `[algorithm]` TOML section:

```python
import tomllib
from alice.algorithm import tdvp2

with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

opts = tdvp2.Options.from_toml(cfg["heisenberg"]["algorithm"])
```

Example TOML block:

```toml
[heisenberg.algorithm]
dt              = 0.05
n_steps         = 40
max_bond        = 128
cutoff          = 1e-12
imaginary_time  = false
```

## See Also

- [Summary](summary.md) — output dataclass.
- [run](run.md) — pass `Options` here.
