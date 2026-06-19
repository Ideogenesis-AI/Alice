# Options

Two-site BUG run options.

::: alice.algorithm.two_site_bug.Options
    options:
      heading_level: 2

## TOML Loading

`Options` can be loaded directly from an `[algorithm]` TOML section:

```python
import tomllib
from alice.algorithm import two_site_bug

with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

opts = two_site_bug.Options.from_toml(cfg["heisenberg"]["algorithm"])
```

Example TOML block:

```toml
[heisenberg.algorithm]
dt              = 0.05
n_steps         = 40
order           = "strang"
max_bond        = 128
trunc_thresh    = 1e-12
imaginary_time  = false
```

## See Also

- [Summary](summary.md) — output dataclass.
- [run](run.md) — pass `Options` here.
