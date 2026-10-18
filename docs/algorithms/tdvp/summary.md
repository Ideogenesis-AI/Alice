# Summary

TDVP evolution history: the time grid and the per-step energy, norm, bond
dimension, discarded weight, and Krylov error estimate. States themselves are
returned separately as [`Artifact`](artifact.md).

::: alice.algorithm.tdvp.Summary
    options:
      heading_level: 2

## Reading the history

The lists come in two lengths, and mixing them up misaligns the time axis.
Quantities measured *on a state* include the initial one at `t = 0` and so have
`n_steps + 1` entries; quantities produced *by a step* have `n_steps`:

| Length | Fields |
|--------|--------|
| `n_steps + 1` | `times`, `energies`, `norms`, `max_bonds` |
| `n_steps` | `discarded_weights`, `krylov_errors` |

So energies pair directly with times, while the step diagnostics need an offset
of one:

```python
for t, e in zip(summary.times, summary.energies):
    print(f"t = {t:.3f},  E = {e:.10f}")

for t, dw in zip(summary.times[1:], summary.discarded_weights):
    print(f"t = {t:.3f},  discarded weight = {dw:.3e}")
```

`bond_dims` is not a history at all — it is the final bond-dimension profile,
one entry per bond.

`energies` and `norms` are the conserved quantities and should stay flat;
drift in either points at the integrator or the Krylov dimension, not at the
physics. `discarded_weights` is the truncation error of the `'2s'` scheme and
is the one to watch for physical accuracy — a run whose values keep climbing is
outgrowing its `max_bond`. For `'1s'` it is identically zero, since nothing is
ever truncated.

## Serialization

`Summary` supports save/load via `torch.save` / `torch.load`:

```python
# Save after a run.
summary.save("evolution_result.ckpt")

# Load later.
from alice.algorithm.tdvp import Summary
summary = Summary.load("evolution_result.ckpt")
```

During `run()`, the latest summary is written to `evolution.ckpt` under
`Options.checkpoint_dir` (or the current working directory when that option is
unset). Serialization version is `1` (no state in the payload).

`evolution.ckpt` is also where a resumed or continued run recovers its time and
energy history from. A continued run appends to that history, so a single
summary may merge segments computed under different options (a larger
`max_bond`, say) with no marker of where they join — copy the checkpoint
directory beforehand to keep the original series for comparison.

## See Also

- [Artifact](artifact.md) — state snapshot at one time step.
- [Options](options.md) — configuration for the run that produced this summary.
- [run](run.md) — returns `(Summary, Artifact)`.
