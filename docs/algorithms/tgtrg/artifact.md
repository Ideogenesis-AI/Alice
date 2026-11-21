# Artifact

Density-matrix snapshot at one TgTRG cooling step (`ρ(β)`, `beta`, `step`).

::: alice.algorithm.tgtrg.Artifact
    options:
      heading_level: 2

## On-disk layout

Under the checkpoint directory (`Options.checkpoint_dir`, or cwd) and the
artifacts directory (`Options.artifacts_dir`, or `artifacts/` under the
checkpoint directory):

```text
checkpoint_dir/
  thermal.ckpt         # latest Summary (kept after success)
  tgtrg.ckpt           # latest Artifact while running (removed on success)
  artifacts/           # or Options.artifacts_dir, when set
    step_00.ckpt       # when Options.save_artifacts is True
    step_01.ckpt
    ...
```

- Step `0` is after Taylor init (`ρ(τ₀)`).
- Step `k` (`1 … n_steps`) is after the `k`-th cooling sweep.
- Files use two-digit zero padding (`step_{k:02d}.ckpt`).
- `Options.save_artifacts_every` thins the archive: a stride of `k` keeps
  every `k`-th step, plus the final one regardless of the stride.
- `tgtrg.ckpt` is overwritten each step and deleted when `run()` finishes
  successfully; a crash leaves it on disk for inspection.

```python
from alice.algorithm.tgtrg import Artifact

artifact = Artifact.load("artifacts/step_05.ckpt")
rho = artifact.rho
print(artifact.beta, artifact.step)
```

An `Artifact` is also the handoff between the phases of a stacked schedule —
the one `run()` returns is passed straight into the next `run()` call. See the
[overview](index.md#stacking-the-two-grids).

## See Also

- [Summary](summary.md) — thermodynamic history without `ρ`.
- [Options](options.md) — `artifacts_dir` / `save_artifacts` / `save_artifacts_every`.
- [run](run.md) — takes a starting `Artifact`, returns the final one.
- [Lessons](lessons.md#run-cools-in-place) — why an `Artifact` is consumed by the run it is given to.
- [NormalMPO](../../api/network/normal-mpo.md) — type of `artifact.rho`.
