# Summary

XTRG output summary containing the final density matrix and thermodynamic
observables at each cooling step.

::: alice.algorithm.xtrg.Summary
    options:
      heading_level: 2

## Serialization

`Summary` supports save/load via `torch.save` / `torch.load`:

```python
# Save after a run.
summary.save("xtrg_result.ckpt")

# Load later.
from alice.algorithm.xtrg import Summary
summary = Summary.load("xtrg_result.ckpt")
```

Mid-run checkpoints written by `run()` (`xtrg.ckpt` under `Options.checkpoint_dir`,
or the current working directory when that option is unset) use the same
`Summary` format and are loadable the same way.

## See Also

- [Options](options.md) — configuration for the run that produced this summary.
- [run](run.md) — returns a `Summary`.
- [NormalMPO](../../api/network/normal-mpo.md) — type of `summary.rho`.
