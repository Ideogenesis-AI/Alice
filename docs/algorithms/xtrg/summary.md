# Summary

XTRG thermodynamic summary: β grid, log Z, and derived observables at each
cooling step. Density matrices are returned separately as
[`Artifact`](artifact.md).

::: alice.algorithm.xtrg.Summary
    options:
      heading_level: 2

## Serialization

`Summary` supports save/load via `torch.save` / `torch.load`:

```python
# Save after a run.
summary.save("thermal_result.ckpt")

# Load later.
from alice.algorithm.xtrg import Summary
summary = Summary.load("thermal_result.ckpt")
```

During `run()`, the latest summary is written to `thermal.ckpt` under
`Options.checkpoint_dir` (or the current working directory when that option
is unset). Serialization version is `2` (no density matrix in the payload).

`thermal.ckpt` is also where a resumed or continued run recovers its β/log Z
history from. A continued run appends to that history, so a single summary may
merge segments computed under different options (a larger `max_bond`, say). The
merged history records no marker of where the options changed, and the `u` /
`c_V` finite differences spanning the junction mix both accuracies — copy the
checkpoint directory beforehand to keep the original series for comparison.

## See Also

- [Artifact](artifact.md) — density-matrix snapshot at one cooling step.
- [Options](options.md) — configuration for the run that produced this summary.
- [run](run.md) — returns `(Summary, Artifact)`.
