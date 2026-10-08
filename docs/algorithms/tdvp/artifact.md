# Artifact

State snapshot at one TDVP time step (`state`, `time`, `step`).

::: alice.algorithm.tdvp.Artifact
    options:
      heading_level: 2

`Artifact` is both the input and the output of [`run`](run.md): it carries the
initial state in, and the final state out. This is what makes a resumed run
indistinguishable from a fresh one.

## On-disk layout

Under the checkpoint directory (`Options.checkpoint_dir`, or cwd) and the
artifacts directory (`Options.artifacts_dir`, or `artifacts/` under the
checkpoint directory):

```text
checkpoint_dir/
  evolution.ckpt       # latest Summary (kept after success)
  tdvp.ckpt            # latest Artifact while running (removed on success)
  artifacts/           # or Options.artifacts_dir, when set
    step_00000.ckpt    # when Options.save_artifacts is True
    step_00004.ckpt    # spacing set by Options.save_artifacts_every
    ...
```

- Step `0` is the initial state at `t = 0`.
- Step `k` (`1 … n_steps`) is the state at `t = k · dt`.
- Files use five-digit zero padding (`step_{k:05d}.ckpt`), so a long run's
  archives still sort lexicographically.
- `tdvp.ckpt` is overwritten each step and deleted when `run()` finishes
  successfully; a crash leaves it on disk for inspection.

```python
from alice.algorithm.tdvp import Artifact

artifact = Artifact.load("artifacts/step_00020.ckpt")
state = artifact.state
print(artifact.time, artifact.step)
```

Archived states are how time-dependent observables are measured: TDVP records
no observables of its own, so anything beyond energy and norm is computed
afterwards from the snapshots. Saving states is also far more expensive than
saving a thermodynamic history, and a real-time run is typically hundreds to
thousands of steps long — hence `save_artifacts_every`, which thins the archive
to a sparse, evenly spaced sample of the trajectory.

## The grid

Step `k` is archived when `k % save_artifacts_every == 0`. The stride is keyed
on the absolute step index rather than on where the run started, so a
trajectory stitched together from several resumed calls lands on exactly the
same grid as one that ran straight through.

A run never evolves past the last grid point. When `n_steps` is not a multiple
of the stride, [`run`](run.md) stops at the largest multiple that is not
greater than it and warns:

```text
n_steps = 1000 is not a multiple of save_artifacts_every = 8; stopping at
step 992 instead so the final state lands on the artifact grid
```

The reason is that `tdvp.ckpt` is removed on success, so an off-grid endpoint
would finish the run with no snapshot of its own final state anywhere on disk.
Only the endpoint is constrained — a run may *start* at any step, which is why
resuming from a crash at an off-grid step still works.

## See Also

- [Summary](summary.md) — evolution history without the states.
- [Options](options.md) — `artifacts_dir` / `save_artifacts` / `save_artifacts_every`.
- [run](run.md) — takes and returns an `Artifact`.
- [MPS](../../api/network/mps.md) — type of `artifact.state`.
- [observe](../../api/network/observe.md) — measuring an archived state.
