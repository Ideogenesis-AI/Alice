# Artifact

Density-matrix snapshot at one XTRG cooling step (`ρ(β)`, `beta`, `step`).

::: alice.algorithm.xtrg.Artifact
    options:
      heading_level: 2

## On-disk layout

Under the checkpoint directory (`Options.checkpoint_dir`, or cwd) and the
artifacts directory (`Options.artifacts_dir`, or `artifacts/` under the
checkpoint directory):

```text
checkpoint_dir/
  thermal.ckpt         # latest Summary (kept after success)
  xtrg.ckpt            # latest Artifact while running (removed on success)
  artifacts/           # or Options.artifacts_dir, when set
    step_00.ckpt       # when Options.save_artifacts is True
    step_01.ckpt
    ...
```

- Step `0` is after Taylor init (`ρ(τ₀)`).
- Step `k` (`1 … n_steps`) is after the `k`-th squaring.
- Files use two-digit zero padding (`step_{k:02d}.ckpt`).
- `xtrg.ckpt` is overwritten each step and deleted when `run()` finishes
  successfully; a crash leaves it on disk for inspection.

```python
from alice.algorithm.xtrg import Artifact

artifact = Artifact.load("artifacts/step_05.ckpt")
rho = artifact.rho
print(artifact.beta, artifact.step)
```

## See Also

- [Summary](summary.md) — thermodynamic history without `ρ`.
- [Options](options.md) — `artifacts_dir` / `save_artifacts` / `save_artifacts_since`.
- [run](run.md) — returns `(Summary, Artifact)`.
- [NormalMPO](../../api/network/normal-mpo.md) — type of `artifact.rho`.
