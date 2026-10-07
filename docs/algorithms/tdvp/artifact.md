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
    step_00.ckpt       # when Options.save_artifacts is True
    step_01.ckpt
    ...
```

- Step `0` is the initial state at `t = 0`.
- Step `k` (`1 … n_steps`) is the state at `t = k · dt`.
- Files use two-digit zero padding (`step_{k:02d}.ckpt`).
- `tdvp.ckpt` is overwritten each step and deleted when `run()` finishes
  successfully; a crash leaves it on disk for inspection.

```python
from alice.algorithm.tdvp import Artifact

artifact = Artifact.load("artifacts/step_20.ckpt")
state = artifact.state
print(artifact.time, artifact.step)
```

Keeping every step is how time-dependent observables are measured: TDVP records
no observables of its own, so anything beyond energy and norm is computed
afterwards from the archived states. Saving states is also far more expensive
than saving a thermodynamic history — `save_artifacts_since` exists to skip the
early steps when only late times matter.

## See Also

- [Summary](summary.md) — evolution history without the states.
- [Options](options.md) — `artifacts_dir` / `save_artifacts` / `save_artifacts_since`.
- [run](run.md) — takes and returns an `Artifact`.
- [MPS](../../api/network/mps.md) — type of `artifact.state`.
- [observe](../../api/network/observe.md) — measuring an archived state.
