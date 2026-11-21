# Summary

TgTRG thermodynamic summary: β grid, log Z, and derived observables at each
cooling step. Density matrices are returned separately as
[`Artifact`](artifact.md).

::: alice.algorithm.tgtrg.Summary
    options:
      heading_level: 2

## Serialization

`Summary` supports save/load via `torch.save` / `torch.load`:

```python
# Save after a run.
summary.save("cooling_result.ckpt")

# Load later.
from alice.algorithm.tgtrg import Summary
summary = Summary.load("cooling_result.ckpt")
```

During `run()`, the latest summary is written to `cooling.ckpt` under
`Options.checkpoint_dir` (or the current working directory when that option is
unset). Serialization version is `1` (no density matrix in the payload).

`cooling.ckpt` is also where a resumed or continued run recovers its β/log Z
history from, and `finished` is `False` on the mid-run snapshots. A continued
run appends to that history, so a single summary may merge segments computed
under different options — the two phases of a stacked exponential-then-linear
schedule, or a second pass at a larger `max_bond`. `taus` is the one column
that records where the step size changed; the `u` and `c_V` finite differences
spanning a junction mix both spacings, and are correspondingly worse there.

## Which columns to trust

| Column | Source | Accuracy |
|--------|--------|----------|
| `log_z` | `NormalMPO.log_trace()` | roundoff, when the rank suffices |
| `free_energies` | `log_z` directly | same as `log_z` |
| `energies` | first differences of `log_z` | limited by grid spacing |
| `specific_heats` | second differences | limited by grid spacing, worst of the three |
| `entropies` | `β (u - f)` | inherits the `energies` error |
| `exact_energies` | `Tr[ρH] / (Tr[ρ] L)` | roundoff; empty unless `Options.obs_exact` |

Comparing `energies` against `exact_energies` on the same run measures the
grid's discretization error directly.

## See Also

- [Artifact](artifact.md) — density-matrix snapshot at one cooling step.
- [Lessons](lessons.md) — what limits each column, and what a converged Krylov error does not promise.
- [Options](options.md) — configuration for the run that produced this summary.
- [run](run.md) — returns `(Summary, Artifact)`.
