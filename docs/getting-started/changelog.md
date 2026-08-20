# Changelog

## [0.2.6] - 2026-08-20

**Continuable Cooling, Faster Thermal Measurement**

Generalizes XTRG resumption: `run()` now accepts a starting state from *any* archived
`artifacts/step_XX.ckpt`, not just the step the previous run stopped at, so a finished
run can be cooled further or a segment re-cooled under different options. Separately,
`observe()` on a thermal `NormalMPO` is rewritten as a transfer-matrix environment
sweep instead of forming and compressing the MPO product `ρ · O`. No breaking API
changes.

### Continuing a Finished XTRG Run

- `run()` accepts a state at any step covered by `thermal.ckpt`, including one *before*
  the end of that history; the later entries are truncated (with a `WARNING`), then
  recomputed and overwritten on disk together with their `step_XX.ckpt` archives.
  Previously the history had to end exactly at `state.step`.
- History recovery moves into a new `_resume_history` helper, with validation anchored
  at `state.step` — β is compared at `history.betas[state.step]` rather than at the end
  of the history, which is rejected only if it *stops before* `state.step`.
- New checks: `history.betas[0]` must match `opts.tau_0`, and `state.step` must not be
  past `opts.n_steps` (which is the absolute step index to stop at, counted from τ₀,
  not a number of additional steps).
- A resumed run's startup banner reports the starting step and steps remaining, and
  derives `beta_max` from `state.beta` instead of τ₀.

### Thermal `observe()` via Environment Sweep

- `observe(rho, O)` for a `NormalMPO` now evaluates `Tr[ρ O] / Tr[ρ]` with a
  left-to-right sweep accumulating a `(ρ_bond, O_bond)` environment, mirroring the MPS
  path, instead of forming `ρ · O` at bond dimension `χ_ρ · χ_O` and calling `compact()`
  before the trace.
- The ratio is still combined in log-space, so it stays correct when either trace alone
  would overflow float64; the boundary scalar accounts for the Bridge (`intw`) weight,
  covering generic symmetry groups.
- A length mismatch between `rho` and `observable` now raises `ValueError`.

### Tests

- New `TestResume` cases in `test_xtrg.py` for continuing a finished run from
  `step_02.ckpt`, re-cooling from an earlier step (asserting the truncation warning),
  and the three new error conditions.
- New thermal `observe()` coverage in `test_observe.py`, shared via
  `_ObserveThermalTests` and run against both U(1) and SU(2) realizations of the same
  Heisenberg chain, including an SU(2)-vs-U(1) cross-check of `⟨H⟩_β`; new session-scoped
  `heisenberg_mpo_u1`/`heisenberg_mpo_su2` fixtures in `tests/network/conftest.py`.

### Documentation

- `xtrg/index.md` gains a "Continuing a finished run" section; the module and `run()`
  docstrings document the same, including the new `ValueError` conditions.
- `xtrg/summary.md` warns that a continued run's summary may merge segments computed
  under different options, with `u`/`c_V` finite differences at the junction mixing both
  accuracies.

### Statistics

- **968 tests** across 29 test modules (up from 954 / 29 modules in v0.2.5).
- **8 commits** since v0.2.5.
- **8 files changed**, 498 insertions, 46 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** none.
- **Behavioral Changes:** a `thermal.ckpt` reaching past `state.step` is now accepted
  and truncated rather than rejected, while a mismatched τ₀ and a `state.step` past
  `opts.n_steps` are now rejected; a continued run overwrites the `step_XX.ckpt` and
  `thermal.ckpt` entries past its starting step. Thermal `observe()` results may differ
  in the last few digits, since the environment sweep skips the `compact()` compression
  the MPO-product path applied.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.5] - 2026-08-19

**Independent Artifact Storage for DMRG and XTRG**

Decouples archived-artifact storage from the checkpoint directory in both
`dmrg.run()` and `xtrg.run()`, via a new `Options.artifacts_dir`. DMRG gains
final-state archiving to match XTRG's existing per-step archiving; XTRG's mid-run
`Artifact` file is renamed from `progress.ckpt` to `xtrg.ckpt`. Two breaking changes,
both to on-disk file names.

### `artifacts_dir`: Decoupled from `checkpoint_dir`

- New `Options.artifacts_dir` on both `dmrg.Options` and `xtrg.Options` (default
  `None`, resolving to `artifacts/` under `checkpoint_dir`); an explicit path archives
  artifacts elsewhere, independent of where checkpoints themselves live.
- XTRG's `save_artifacts`/`save_artifacts_since` now write `step_XX.ckpt` under
  `artifacts_dir` instead of always under `checkpoint_dir/artifacts`.
- Both `run()`s log the resolved artifacts directory in their startup banner.

### DMRG Final-State Archiving

- **Breaking:** `dmrg.run()` now archives its final `Summary` as `state.ckpt` under
  `artifacts_dir` on success, then removes the per-sweep `dmrg.ckpt`/`dmrg_lock.ckpt`,
  since they are redundant with the archived artifact. Previously `dmrg.ckpt` was left
  on disk after a successful run.
- Unlike the per-sweep checkpoint, the archived `Summary` carries the correct
  `converged` flag, since it is built after the sweep loop exits.

### XTRG `xtrg.ckpt` Rename

- **Breaking:** the mid-run `Artifact` checkpoint is renamed from `progress.ckpt` to
  `xtrg.ckpt` (lock file `progress_lock.ckpt` → `xtrg_lock.ckpt`), aligning the name
  with DMRG's `dmrg.ckpt` as each algorithm's own per-run checkpoint file. Behavior is
  unchanged: written every step, removed on success.

### Tests

- New `TestArtifact` class in `test_dmrg.py`: `state.ckpt` creation, lock-file cleanup,
  round-trip loading with a correct `converged` flag, default placement, and
  `artifacts_dir` decoupled from `checkpoint_dir`.
- New `artifacts_dir` default/TOML-round-trip tests in both `test_dmrg.py` and
  `test_xtrg.py`; `test_progress_removed_after_success` renamed to
  `test_checkpoint_removed_after_success` and updated for `xtrg.ckpt`.

### Documentation

- `artifact.md`'s on-disk layout diagram gains `artifacts_dir` and the `xtrg.ckpt`
  rename; `index.md`'s resumption example and both example scripts updated from
  `progress.ckpt` to `xtrg.ckpt`.

### Statistics

- **954 tests** across 29 test modules (up from 946 / 29 modules in v0.2.4).
- **11 commits** since v0.2.4.
- **9 files changed**, 214 insertions, 69 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** `dmrg.run()` no longer leaves `dmrg.ckpt` on disk after a
  successful run — read `artifacts_dir/state.ckpt` instead. XTRG's mid-run
  `progress.ckpt` is renamed to `xtrg.ckpt`; callers loading it by a hardcoded path
  must update the filename.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.4] - 2026-08-18

**XTRG: Resumption from Interrupted Runs**

Allows an interrupted XTRG run to resume from its last checkpointed step rather than
restarting from `ρ(τ₀)`. This required moving state construction out of `run()`: it now
accepts a starting `Artifact` (`rho`, `beta`, `step`) supplied by the caller, instead of
building `ρ(τ₀)` internally via `thermal_mpo()`. Breaking change to the signature of
`run()`.

### Resumable `run()`

- **Breaking:** `run(H, spc, opts)` becomes `run(state: Artifact, opts=None)`; the chain
  length `L` is read from `state.rho.L`, and the internal `thermal_mpo` call is removed.
- At `state.step == 0`, `run()` records the initial `(beta, log Z)` grid point directly
  from `state`, as before. When resuming (`state.step > 0`), it loads `thermal.ckpt` to
  recover the prior `betas`/`log_z`/`discarded_weights` history, validates it against
  `state.step` and `state.beta`, and continues squaring from `state.step` onward.
- Callers now build the initial state explicitly — `thermal_mpo(...)` wrapped in an
  `Artifact` — and resume a crashed run by passing the `Artifact` from `progress.ckpt`
  back into `run()`.

### Tests

- New `_initial_state` helper builds the step-zero `Artifact`; all existing XTRG tests
  updated to use it.
- New `TestResume` (5 tests): a resumed run matches an uninterrupted one's thermodynamics
  to `rel_tol=1e-10` while only computing the remaining steps; missing or inconsistent
  checkpoints raise `FileNotFoundError`/`ValueError`; resuming past `n_steps` is a no-op.

### Documentation

- `xtrg_spinless`, `xtrg_spinful`, and the XTRG documentation and worked examples
  (free-fermion, Hubbard) updated to build the initial `Artifact` via `thermal_mpo`
  before calling `run()`. The module docstring and `index.md` gain a resumption example.

### Statistics

- **946 tests** across 29 test modules (up from 940 / 29 modules in v0.2.3).
- **9 commits** since v0.2.3.
- **8 files changed**, 295 insertions, 82 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** `xtrg.run()` no longer accepts `(H, spc, opts)`; its signature is
  now `run(state: Artifact, opts=None)`. Callers must build the initial state themselves,
  e.g. `xtrg.Artifact(rho=thermal_mpo(H, opts.tau_0, opts.taylor_order, spc), beta=opts.tau_0, step=0)`.
  `progress.ckpt` and `thermal.ckpt` from earlier versions remain loadable for resumption.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.3] - 2026-08-17

**XTRG: Convergence-Based Early Termination**

Lets XTRG's variational compression terminate before exhausting its sweep budget once it
has converged, via a new `Options.z_tol` tolerance on `‖C‖` between sweeps.
`Summary.converged` is renamed to `Summary.finished` to say what it actually means.
One breaking change to `Summary`'s fields.

### Early-Termination Compression Fit

- New `Options.z_tol` (default `1e-10`): `_fit_mpo` terminates sweeping once
  `|‖C‖ − ‖C_prev‖| < z_tol`, measured at the orthogonality center after each full
  sweep; `n_sweeps` becomes a maximum rather than a fixed count.
- The criterion is exact: at the least-squares optimum, `⟨C, A·B⟩ = ‖C‖²`, so
  `‖C − A·B‖²_F = ‖A·B‖² − ‖C‖²` with `‖A·B‖` fixed across sweeps — `‖C‖` convergence is
  equivalent to residual convergence, read from the already-isometric center tensor.
- `z_tol=0.0` restores the previous fixed-sweep-count behavior exactly. `run()`'s
  startup banner now logs the configured fit convergence threshold.

### `Summary.finished` Rename

- **Breaking:** `Summary.converged` is renamed to `Summary.finished`; the field was
  always about whether the summary came from a completed `run()` call, not fit
  convergence. `serialize()`/`deserialize()` updated; serialization version stays at 2.
- A v0.2.2 `thermal.ckpt` still loads, but reloads as `finished=True` unconditionally,
  since the old `converged` key is no longer consulted.

### Tests

- New `TestFitMpoZTol`: a loose `z_tol` stops early, `z_tol=0.0` always runs the full
  sweep budget, and an early-terminated fit reproduces the `log_z` of a full-sweep run.

### Statistics

- **940 tests** across 29 test modules (up from 937 / 29 modules in v0.2.2).
- **7 commits** since v0.2.2.
- **3 files changed**, 119 insertions, 20 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** `Summary.converged` is renamed to `Summary.finished`; a
  `thermal.ckpt` from v0.2.2 or earlier still loads but reloads as `finished=True`
  unconditionally.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.2] - 2026-08-14

**XTRG: Specific Heat and Cache Isolation**

Fixes the sign of XTRG's specific heat, which was reported negative for every physical
Hamiltonian, and isolates XTRG's environment disk cache per run so that concurrent jobs
sharing one `env_cache_dir` no longer overwrite each other's blocks. The remainder is a
consistency pass over the codebase: American English spelling, "index" in place of "leg",
lowercase builtin type annotations, and role-based einsum index names. No breaking
changes.

### Specific Heat Sign Fix

- **Bug fixed:** `_compute_observables` computed `c_V[n] = β_n Δu / ln 2`, dropping the
  minus sign in `c_V = ∂u/∂T = −β ∂u/∂(ln β)`. Since `u` decreases with β, the reported
  specific heat was negative wherever the true value is positive.
- Both the interior and the trailing one-sided stencil now carry the correct sign;
  `free_energies`, `energies`, and `entropies` are unaffected.

### XTRG Environment Cache Isolation

- `xtrg.run()` now creates a unique subdirectory (first 8 hex characters of a UUID4,
  e.g. `{env_cache_dir}/a1b2c3d4/`) inside `env_cache_dir` per invocation, reused by
  every squaring step and removed in a `finally` block on return or exception.
- Cache-directory resolution moved from `_fit_mpo` to `run()`; `_fit_mpo` gains a
  `cache_dir` parameter and no longer reads `opts.env_cache_dir`.
- The resolved cache path is logged in `run()`'s startup banner.

### Consistency Pass

- All comments, docstrings, and documentation pages converted to American English;
  "leg" replaced by "index" or an ordinal rank ("2nd-order" instead of "2-leg").
- XTRG drops `typing.Dict`/`typing.List` for builtin `dict`/`list`; `sweep` gains real
  `Options` annotations under `TYPE_CHECKING`.
- `_observe_mps`'s contraction renamed from `einsum('ace,abg,cdgh,efh->bdf', …)` to
  `einsum('aob,acr,oprs,bds->cpd', …)`, following the project's index-naming convention.

### Tests

- New `TestComputeObservables` (specific-heat sign on a two-level system) and
  `TestEnvCache` (per-run subdirectory creation and removal, distinct subdirectories
  across runs, cached run matching the in-memory run, TOML round trip).
- `test_scheme_1s`'s idempotency test now builds mixed-canonical environments in the
  compressed MPO's own canonical frame, where the 1-site update is a true variational
  optimum.

### Documentation

- New **Environment caching for large chains** section in the XTRG free-fermion example;
  the DMRG Hubbard example's caching note updated for the per-run subdirectory.

### Statistics

- **937 tests** across 29 test modules (up from 930 / 29 modules in v0.2.1).
- **44 commits** since v0.2.1.
- **48 files changed**, 528 insertions, 313 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.2.1. Checkpoints from
  v0.2.0/v0.2.1 still load, but their `specific_heats` entries must be negated.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.1] - 2026-08-07

**XTRG: Density-Matrix Artifacts**

Splits XTRG's density matrix out of `Summary` into a new `Artifact` dataclass, and
reworks checkpointing around that split: `thermal.ckpt` tracks the thermodynamic history
alone, `progress.ckpt` protects the latest density matrix against a mid-run crash, and an
optional `artifacts/` archive keeps a caller-chosen range of per-step density matrices on
disk. `xtrg.run()` now returns `(Summary, Artifact)`. Breaking change to `xtrg.run()`'s
return signature and `Summary`'s fields.

### `Artifact` and Checkpointing

- New `Artifact(AlgorithmSummary)` dataclass holding `rho` (`NormalMPO`), `beta`, and
  `step`; exported from `alice.algorithm.xtrg` alongside `Options`, `Summary`, and `run`.
- `Summary` drops its `rho` and `rho_log_scale` fields entirely; `serialize`/
  `deserialize` bump to version 2 and reject version-1 (rho-carrying) payloads with a
  `ValueError`.
- `thermal.ckpt` replaces `xtrg.ckpt` (rho-free `Summary`, written every step);
  `progress.ckpt` is new (latest `Artifact`, deleted on successful completion).
- New `Options.save_artifacts` (default `True`) and `Options.save_artifacts_since`
  (default `0`) archive per-step `Artifact` files under `artifacts/step_XX.ckpt`.
- `checkpoint_dir` now defaults to the current working directory instead of disabling
  checkpointing when unset.

### Examples

- `xtrg_spinless` and `xtrg_spinful` return `(Summary, Artifact)` and accept
  `save_artifacts` / `save_artifacts_since`; CLI gains `--no-save-artifacts` and
  `--save-artifacts-since K`.

### Documentation

- New `docs/algorithms/xtrg/artifact.md`; other XTRG algorithm pages updated for the
  `Summary`/`Artifact` split and the `(Summary, Artifact)` return tuple.

### Statistics

- **930 tests** across 29 test modules (up from 916 / 29 modules in v0.2.0).
- **19 commits** since v0.2.0.
- **14 files changed**, 545 insertions, 142 deletions.
- **26 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** `xtrg.run()` returns `(Summary, Artifact)` instead of a single
  `Summary`; `Summary` no longer has `rho`/`rho_log_scale` fields; `xtrg.ckpt` is
  replaced by `thermal.ckpt` and `progress.ckpt`.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.0] - 2026-08-01

**XTRG: Finite-Temperature Thermodynamics**

Introduces XTRG (eXponential Tensor Renormalization Group), Alice's second algorithm
alongside DMRG: a finite-temperature solver that computes `ρ(β) = e^{-βH}` by repeated
squaring, sharing the `1s`/`2s`/`1sp` update schemes with DMRG. `NormalMPO` is upgraded to
track its physical magnitude in log form, and `observe` computes thermal expectation-value
ratios in log-space, keeping both stable arbitrarily deep into a cooling run. One breaking
change to `NormalMPO.__init__`.

### XTRG Algorithm

- New `alice.algorithm.xtrg` package exporting `Options`, `Summary`, and `run`, mirroring
  the `alice.algorithm.dmrg` interface.
- `xtrg.run(H, spc, opts)` initializes `ρ(τ₀)` via `thermal_mpo`, then repeatedly squares
  it — `ρ(2β) ≈ compress(ρ(β) ⊗ ρ(β))` — to reach `β_max = 2^n_steps × τ₀`, sampling an
  exponentially spaced β grid.
- Three update schemes for the inner variational MPO-MPO compression `C ≈ A · B`:
  **1-site** (`1s`), **2-site** (`2s`) with SVD truncation, and **1-site-plus** (`1sp`)
  controlled bond expansion (CBE), adapted from DMRG's `'1sp'` scheme to the linear
  (non-eigenvalue) fitting problem.
- `Summary` reports `betas`, `log_z`, `free_energies`, `energies`, `specific_heats`, and
  `entropies` per cooling step; `u`, `c_V`, and `S` are derived from `log_z` via log-β
  finite differences for uniform accuracy across the exponential grid.
- `Options.checkpoint_dir` atomically checkpoints `ρ` and thermodynamic history after
  every cooling step, matching DMRG's checkpoint pattern.

### `NormalMPO`: Log-Scale Representation

- **Breaking:** `NormalMPO.__init__` keyword argument renamed from `scale` to
  `log_scale` (`log_scale = log(scale)`).
- New `log_trace()` method returns `(log|Tr[ρ]|, sign)` without ever materializing the
  raw trace, which can reach `~10^500` deep into an XTRG run; `trace()` is now a thin
  wrapper over it. New `scale_by(log_scale_delta)` for in-place log-magnitude updates.
- `__matmul__`, `__add__`, and `__mul__` combine `log_scale` by addition/subtraction
  instead of multiplying raw floats; the sign of the physical operator is folded into
  site 0's tensor data instead.

### `observe`: Log-Space Thermal Ratios

- `_observe_thermal` now computes `Tr[ρ O] / Tr[ρ]` via `log_trace()` on both numerator
  and denominator, combined as `(sign_num · sign_den) × exp(log_num − log_den)`, so the
  well-behaved O(1) ratio remains correct even when either trace overflows float64.

### Documentation

- New `docs/algorithms/` section (replacing algorithm pages formerly under `docs/api/`)
  with per-algorithm subdirectories (`algorithms/dmrg/`, `algorithms/xtrg/`).
- New `docs/examples/xtrg/` pages for free-fermion and Hubbard worked examples, validated
  against exact grand-canonical solutions.
- README's **Algorithms** section gains an XTRG subsection alongside the existing DMRG
  one, plus a new **Upcoming** list (tanTRG, TDVP, TaSK).

### Statistics

- **916 tests** across 29 test modules (up from 814 / 23 modules in v0.1.6).
- **66 commits** since v0.1.6.
- **49 files changed**, 6,172 insertions, 92 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** `NormalMPO.__init__` keyword argument renamed from `scale` to
  `log_scale`; `NormalMPO.from_mpo()`, `thermal_mpo()`, and the `scale` read-only
  property are unaffected.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.1.6] - 2026-06-10

**MPS Initialization for Odd Chains**

Extends `init_mps` with a `target_qn` parameter for explicit quantum-number targeting,
fixes a silent zero-MPS bug in random initialization for odd-L chains, and replaces
`warnings.warn` with structured `logging` throughout `automps`. No breaking changes.

### `init_mps`: `target_qn` Parameter

- New `target_qn` keyword on `init_mps`: the desired right-boundary charge `Q[L]` (total
  quantum number of the chain). Defaults to `Q_vac` (half-filling), preserving behavior
  for even-L even-filling cases.
- When `target_qn` is given and the auto-config cannot reach it (e.g. `target_qn=0` for
  odd-L spin-½), `init_mps` raises a `ValueError` in both modes rather than silently
  producing an MPS in the wrong sector.

### Odd-L Random MPS Fix

- **Bug fixed:** `_random_mps` previously pinned both boundary indices to `Op['vac']`
  (charge `Q_vac`). For odd L, where no config can return to `Q_vac`, every charge block
  of the last tensor was forbidden and the MPS canonicalized to zero.
- The right boundary is now constructed with `Q[L]` (the actual charge reached by the
  auto-config path) when `target_qn` was not given explicitly. For even L this is still
  `Q_vac`; for odd L it is the correct non-vacuum charge.

### Warning System Upgrade

- `init_mps` no longer calls `warnings.warn(UserWarning)` for odd-L chains where
  `Q_vac` is unreachable. The message is now emitted via `logging.getLogger(__name__)`
  and recommends passing `target_qn` explicitly.

### `_auto_config` Generalization

- `_auto_config` now accepts `target_qn` and all three internal strategies (single-sector
  fill, period-2 alternation, greedy fallback) target `target_qn` instead of `Q_vac`.
- The function is now a pure helper with no side effects; the sole warning site is
  `init_mps`.

### Documentation

- `init-mps.md` gains a **Logic Overview** decision diagram, an **Odd-chain lengths**
  section with a spin-½ example, and a corrected **Bond Sectors in Random Mode** table
  note (BFS is seeded from `Q_c = Q[L//2]`, not from `Q_vac`).

### Statistics

- **814 tests** across 23 test modules (up from ~795 / 23 modules in v0.1.5).
- **5 commits** since v0.1.5.
- **4 files changed**, 423 insertions, 89 deletions.
- **22 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.5.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.1.5] - 2026-06-08

**Cache Isolation and Thermal MPO**

Fixes DMRG environment-block caching for concurrent runs and corrects the early-stopping
criterion in `thermal_mpo` to account for the operator norm of `H`. No breaking changes.

### DMRG Environment Cache Isolation

- `dmrg.run()` now creates a unique subdirectory (first 8 hex characters of a UUID4,
  e.g. `{env_cache_dir}/a1b2c3d4/`) inside `env_cache_dir` per invocation. Previously,
  concurrent runs sharing the same `env_cache_dir` wrote to the same `left/` and
  `right/` paths and could corrupt each other's cached blocks.
- The unique subdirectory is removed automatically in a `finally` block on return or
  exception, leaving no stale files behind.
- `Options.env_cache_dir` docstring updated to document the subdirectory scheme and
  automatic cleanup.

### `thermal_mpo` Early-Stopping Correction

- The early-stopping guard now tests `|β^n / n!| × ‖H^n‖_F < coeff_thresh` rather
  than the bare coefficient `|β^n / n!|` alone. For Hamiltonians with large operator
  norm, `‖H^n‖_F` can be much larger than 1, causing the old criterion to exit
  prematurely before the series had converged.
- The conditional that skipped the power-update step when the next coefficient was small
  has been removed; `H_pow` is now always advanced when `n < order`.
- `order` and `coeff_thresh` parameter docstrings updated to describe the corrected
  criterion.

### Statistics

- **~795 tests** across 23 test modules (up from ~740 / 23 modules in v0.1.4).
- **3 commits** since v0.1.4.
- **4 files changed**, 141 insertions, 19 deletions.
- **22 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.4.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.1.4] - 2026-05-31

**Thermal Density Matrix**

Introduces `NormalMPO`, an `MPO` subclass with a separately tracked scale factor, and
`thermal_mpo`, which approximates `ρ(β) = exp(−βH)` via a truncated Taylor series in MPO
arithmetic. The `observe` function is extended to accept `NormalMPO` as the state
argument, enabling finite-temperature expectation values within the existing workflow.
No breaking changes.

### `NormalMPO`

- New `NormalMPO` class in `alice.network.thermal`: represents an operator as
  `scale × mpo_unit`, keeping the internal MPO at unit Frobenius norm and carrying the
  physical magnitude in a separate `_scale` attribute.
- `*` (scalar multiply): only `_scale` is updated; site tensors are not modified.
  `@` (MPO product) and `+` (MPO sum) produce new site tensors with `_scale` set to
  the combined physical magnitude. `compact()` folds the extracted SVD norm into
  `_scale` after each compression sweep and restores standard bond-arrow directions
  via `capcup`.
- Class method `NormalMPO.from_mpo(mpo)` constructs a `NormalMPO` from any plain `MPO`.
- Exported from `alice.network` and the `alice` top-level namespace.

### `thermal_mpo`

- New `thermal_mpo(H, beta, order, spc)` function: computes
  `ρ(β) ≈ Σ (−β)^n/n! · H^n` via MPO arithmetic, calling `compact()` after every
  addition and power step to control bond growth. Early exit when the Taylor coefficient
  drops below `coeff_thresh` (default `1e-15`).
- The returned `NormalMPO` carries `_scale ≈ Tr[ρ]` (the partition function, up to the
  MPO norm).
- Exported from `alice.network` and the `alice` top-level namespace.

### `observe` Updated for Thermal States

- `observe` now accepts a `NormalMPO` as the state argument: forms the MPO product
  `ρ @ O`, compresses it with `compact()`, and returns `Tr[ρ O] / Tr[ρ]` — the
  normalized thermal expectation value.

### Documentation

- New API reference pages for `NormalMPO` and `thermal_mpo`; `alice.network` index and
  `observe` reference updated to document the new types.

### Statistics

- **~740 tests** across 23 test modules (up from ~700 / 22 modules in v0.1.3).
- **13 commits** since v0.1.3.
- **9 files changed**, 1,192 insertions, 10 deletions.
- **22 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.3.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.1.3] - 2026-05-13

**MPS Init and DMRG Checkpointing**

Introduces `init_mps`, a universal MPS initializer that replaces the per-example
`_random_mps` helpers with a single particle-type–agnostic function. Adds atomic
per-sweep checkpointing to DMRG via `dmrg.Options.checkpoint_dir`, and exposes
`Network.bond_states` for counting physical states at each bond in non-Abelian
simulations. No breaking changes.

### Universal MPS Initializer

- New `alice.network.automps` module with `init_mps`: constructs an initial MPS for
  DMRG from any `(Spc, Op)` pair returned by `load_space`. Works for bosonic,
  fermionic, and conductor sites without a `particle_type=` argument.
- `bond_dim=1`: deterministic product state with exact charge targeting, one sector per
  bond; best paired with CBE (`scheme='1sp'`) or 2-site (`scheme='2s'`) DMRG.
- `bond_dim>1`: random MPS with group-derived bond sectors; reachable charges are
  selected by BFS from the center-bond charge to depth 2.
- Auto-balanced `config=None` heuristic covers all standard even-*L* half-filled cases:
  alternating high/low for 2-sector spaces, single neutral sector for 3-sector spaces,
  alternating neutral-pair for 4-sector spaces, SU(2) dimer path for pure-SU(2) spaces.
- Exported from `alice` top-level namespace and from `alice.network`.
- DMRG example scripts (`dmrg_heisenberg`, `dmrg_freefermion`, `dmrg_conductor`)
  updated to accept an `init` parameter (`'iter_diag'` or `'random'`) using `init_mps`;
  per-example `_random_mps` helpers removed.

### DMRG Checkpointing

- New `checkpoint_dir` option in `dmrg.Options`: after every completed sweep the
  current state is serialized to `dmrg.ckpt` via an atomic write (write to
  `dmrg_lock.ckpt`, then rename). On POSIX systems the rename is atomic, so a crash
  during serialization cannot corrupt the previous checkpoint.
- Checkpoint file is in PyTorch format, loadable via `dmrg.Summary.load`.
- Defaults to the current working directory at `run()` call time (matching `.logging`).

### `Network.bond_states`

- New `bond_states` property on `Network`: returns the number of physical states per
  internal bond (length `L - 1`). Equal to `bond_dims` for Abelian groups; larger for
  SU(2) due to multiplet degeneracy (*2j+1* states per multiplet of spin *j*).

### Documentation

- New API reference page for `init_mps` with parameter table, charge-convention notes,
  and runnable examples; `Network` reference updated with `bond_states`.
- Quick-start guide updated to use `init_mps` and the `Summary` checkpoint interface.

### Statistics

- **~700 tests** across 22 test modules (up from ~650 / 21 modules in v0.1.2).
- **16 commits** since v0.1.2.
- **17 files changed**, 1,319 insertions, 187 deletions.
- **21 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.2.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.6.

---

## [0.1.2] - 2026-05-11

**Geometry Expansion and Refactor**

Introduces Kagome lattice support, a full refactor of the geometry subsystem around a
`Geometry` dataclass, a unified traversal-order naming scheme, and ASCII text diagrams
for MPS and MPO chains. Also adds `Sp4`/`Sm4` operator templates to `build_bosonic` and
`build_conductor`. Several breaking changes to the geometry and `build_interaction` APIs;
the DMRG, Hamiltonian, and algorithm APIs are unchanged.

### Kagome Lattice

- New `alice.physics.kagome` module with `intrcmap_kagome`: nearest-neighbor bond
  generation for Kagome lattices, covering N2U (upward-triangle: A–B, A–C, B–C within
  each unit cell) and N2D (downward-triangle: bonds between adjacent unit cells), with
  OBC/PBC boundary conditions along both axes.
- Traversal orders `'sequential'` (column-major, default) and `'serpentine'`
  (column-major with alternating row direction) for Kagome lattices.
- `intrcmap_kagome` is exported from `alice.physics` alongside `intrcmap_1dchain`
  and `intrcmap_square`.

### Geometry Module Refactor

- New `Geometry` dataclass returned by `build_geometry`; carries `cfg`, `ord_map`,
  and `latt` as a single typed object with derived properties `lattice`, `traverse`,
  `lx`, `ly`, `L`, and helpers `to_1d`/`to_2d`.
- All `intrcmap_*` functions now take a `Geometry` as input (previously a plain dict)
  and continue to return `List[Interaction2Site]`.
- `build_interaction` now returns `(interactions, spc, geo)` — the third element
  changed from an `int` (site count) to the `Geometry` instance.
- New `build_intrcmap(geo)` dispatcher: reconstructs the bond list from any `Geometry`
  instance, decoupling lattice construction from bond enumeration.
- New `build_traversal` helpers in all three lattice modules (`chain.py`, `square.py`,
  `kagome.py`); 1D chain and square lattice geometries separated into their own modules.
- `ord_map` keys changed from integer flat indices to coordinate tuples: `(row, col)`
  for chain and square, `(row, col, u)` for Kagome (where `u` is the sublattice index
  0=A, 1=B, 2=C), making coordinate lookups explicit for all lattice types.

### Traversal Order Naming

- `'snake'` is renamed to `'serpentine'` (same behavior: columns alternate
  direction, even columns top→bottom, odd columns bottom → top).
- `'sequential'` is a new order (all columns top → bottom, no reversal) and is
  now the default for square and Kagome lattices.
- All documentation, example configs, and tests updated to the new naming.

### MPS/MPO Text Display

- New `alice.network.display` module with `network_summary`: Unicode ASCII-art chain
  diagrams showing tensor nodes, bond dimensions, center site, and summary info.
- `MPS.__repr__` and `MPO.__repr__` delegate to `network_summary`, giving readable
  representations in REPLs and notebooks.

### New Operator Templates

- `build_bosonic` and `build_conductor` gain `Sp4`/`Sp4dag`/`Sm4`/`Sm4dag` templates
  (U(1)-only), enabling AutoMPO construction for models where raising and lowering
  channels must be handled separately.

### Documentation

- New API reference pages for `intrcmap_kagome` and `build_intrcmap`; geometry and
  traversal pages updated to reflect the refactored two-stage construction pipeline.
- All documentation page headings standardized to title case.

### Statistics

- **~650 tests** across 21 test modules (up from 538 / 19 modules in v0.1.0).
- **119 commits** since v0.1.1.
- **65 files changed**, 3,380 insertions, 824 deletions.
- **20 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:**
    - `build_geometry` return type changed from `List[Interaction2Site]` to `Geometry`.
    - All `intrcmap_*` functions now take a `Geometry` as input instead of a plain dict;
      return type `List[Interaction2Site]` is unchanged.
    - `build_interaction` third return value changed from `int` (site count) to `Geometry`.
    - `geometry_fn` callable override signature changed from
      `(geo: dict) -> List[Interaction2Site]` to `(geo_cfg: dict) -> Geometry`; a new
      `intrcmap_fn` override was added with signature
      `(geo: Geometry) -> List[Interaction2Site]`.
    - The public function `generate_snake_order` is removed; its traversal logic is now
      internal.
    - The TOML key `traverse: 'snake'` is deprecated; `'snake'` was renamed to
      `'serpentine'`, and `'sequential'` is the new default order.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.6.

---

## [0.1.1] - 2026-05-06

**Documentation Website**

Introduces the complete Alice documentation site: Material-themed MkDocs with a full API
reference, worked DMRG examples, and getting-started guides. Also
renames the PyPI distribution to `alice-net`. No changes to the Alice project; fully
backward compatible with v0.1.0.

### Documentation Infrastructure

- Custom `alice` color scheme, three-tab navigation, dark/light/system theme switching, search with suggestions and shareable links
- Content features: code copy, inline annotations, tabbed content, in-page edit and view actions
- Custom home page template (`docs/overrides/home.html`) with hero image and feature cards; custom stylesheet (`docs/stylesheets/extra.css`)
- Build hook (`docs/hooks.py`) for table-cell bullet-list post-processing
- Plugins: **mkdocstrings** (NumPy-style API docs), **markdown-exec** (configured for future live execution), **git-revision-date-localized**, **git-committers**
- Extensions: MathJax via `arithmatex`, Mermaid via `superfences`, tabbed content, FontAwesome/Material emoji

### Getting Started (7 pages)

- **What is Alice**: philosophy, relationship with Nicole, supported symmetry groups and algorithms
- **Installation**: `pip install alice-net` / `uv add alice-net`, development setup, optional dependency groups
- **Core Concepts**: MPS/MPO block-sparse structure, symmetry sectors, DMRG sweep logic
- **Quick Start**: end-to-end Heisenberg DMRG example from site definition through energy output
- **Contributing**: branch model, coding conventions, test requirements
- **Git Control**: tagging, branching, and release workflow for the Alice project
- **Changelog**: version history beginning with v0.1.0

### API Reference (30 pages)

- **Network** — `Network`, `MPS`, `MPO`: construction, canonicalization, norm, serialization, SVD compression, norm redistribution; `observe`: expectation-value sweep with SU(2) Bridge weight support
- **Interaction** — `Interaction`, `Interaction1Site`, `Interaction2Site` dataclass references; `build_interaction`: TOML-configured builder with plugin section documentation
- **Geometry** — `generate_snake_order`, `intrcmap_1dchain`, `intrcmap_square`, `build_geometry`: function references with parameter tables and usage notes
- **Local Space** — `build_bosonic`, `build_fermionic`, `build_conductor`: site Hilbert space constructors with symmetry-mode tables
- **Hamiltonian** — `build_hamiltonian`: MPO assembler; `build_heisenberg`, `build_free_fermion`, `build_hubbard`: model-specific builders
- **DMRG** — `dmrg.Options`: parameter reference with TOML key mapping; `dmrg.Summary`: output fields and serialization; `dmrg.run`: sweep logic, update schemes (`1s`, `2s`, `1sp`), convergence criteria
- **Logging** — `configure_logging`: handler configuration, log levels, output file naming

### Examples (8 pages)

- **Heisenberg chain**: ground-state energy of a spin-1/2 chain, U(1) and SU(2) symmetry comparison
- **Free fermion**: tight-binding chain benchmark against exact diagonalization
- **Hubbard model**: charge and spin sector targeting in a single-band system
- **AutoMPO from TOML**: `[[interaction]]` tables, `[plugin]` sections for user-defined models, built-in presets
- **Custom geometry**: implementing a user-defined lattice traversal and registering it with `build_geometry`
- **Custom local space**: defining a new site Hilbert space outside the built-in presets
- **Custom model**: wrapping a user-defined Hamiltonian function as an Alice-compatible builder

### Packaging

- PyPI distribution renamed from `alice` to `alice-net`; `pyproject.toml` updated with wheel target and project URLs; README and logging banner updated
- **Install:** `pip install alice-net` or `uv add alice-net`; the import namespace `alice` is unchanged

### Statistics

- **45 documentation pages**; all public symbols documented
- **39 commits** since v0.1.0
- **112 files changed**, 3,161 insertions, 63 deletions

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.0; import paths, function signatures, and TOML configuration formats are unchanged
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.6

---

## [0.1.0] - 2026-05-04

Initial stable release of Alice.

### Tensor Network Infrastructure

- `MPS`, `MPO`, and `Network` classes: block-sparse matrix product states and operators inheriting Nicole's exact symmetry engine with full support for U(1), Z(2), SU(2), and product symmetry groups.
- `Network.canonical()`: left/right QR canonicalization with configurable bond truncation.
- `Network.norm()` and `normalize()`: Frobenius norm via the center tensor or a full transfer-matrix sweep.
- `Network.serialize()` / `Network.deserialize()`: `torch.save`-compatible serialization preserving all index metadata.
- `MPO.compact()` and `MPO.redistribute_norm()`: SVD compression and norm redistribution.
- `observe()`: expectation-value computation via MPS–MPO–MPS transfer-matrix sweep, with SU(2) Bridge weight support.

### AutoMPO Construction

- `build_interaction()`: TOML-configured interaction map builder with built-in Heisenberg, free-fermion, and Hubbard model presets and support for user-defined model functions via `[plugin]` sections.
- `build_hamiltonian()`: term-by-term MPO assembler using Nicole's `oplus`; compresses the result with two canonical sweeps; handles non-Abelian SU(2).
- `Interaction`, `Interaction1Site`, `Interaction2Site` dataclasses.
- Incremental `compact_every` parameter to limit bond dimension growth during large builds.

### Lattice Geometries

- `intrcmap_1dchain()`: nearest-neighbor 1D chain with optional PBC.
- `intrcmap_square()`: 2D square lattice with NN and NNN bonds, PBC support, and snake-order traversal.
- `generate_snake_order()`: boustrophedon traversal for rectangular lattices.
- `build_geometry()`: TOML dispatcher for lattice and traversal selection.

### Physical Spaces

- `build_bosonic()`: spin-`s` site; U(1) and SU(2) symmetry.
- `build_fermionic()`: spinless-fermion site; U(1) and Z(2) symmetry.
- `build_conductor()`: spinful-fermion (Band) site; U(1)×U(1), Z(2)×U(1), U(1)×SU(2), Z(2)×SU(2) symmetry.

### Physics Models

- `build_heisenberg()`: Heisenberg spin model with NN coupling `J` and optional NNN `J'`.
- `build_free_fermion()`: spinless tight-binding chain with hopping `t`, optional NNN `t'`, and chemical potential `µ`.
- `build_hubbard()`: Hubbard model with hopping `t`, on-site `U`, optional NNN `t'`, and `µ` at half-filling.

### DMRG Algorithm

- `dmrg.Options` dataclass: all run parameters with TOML loading via `Options.from_toml()`.
- `dmrg.Summary` dataclass: energy, optimized MPS, convergence info, and serialization.
- `dmrg.run()`: alternating sweep optimization with three update schemes (`1s`, `2s`, `1sp`).
- Davidson eigensolver with thick restart operating directly in the symmetry-block-sparse space.
- `Environment` class with optional disk-spilling (sliding window + async I/O).

### Logging

- `configure_logging()`: one-call setup of Alice's two-handler logging scheme (console + timestamped file). Emits startup banner, model specification summary, sweep-by-sweep diagnostics, and geometry ASCII diagrams.

### Statistics

- 538 tests across 19 test modules.
- 196 commits across 10+ feature branches.
- 64 files, ~16,000 lines of code.
- 16 source modules in three subpackages: `alice.network`, `alice.physics`, `alice.algorithm.dmrg`.

[0.2.6]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.6
[0.2.5]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.5
[0.2.4]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.4
[0.2.3]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.3
[0.2.2]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.2
[0.2.1]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.1
[0.2.0]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.0
[0.1.6]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.6
[0.1.5]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.5
[0.1.4]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.4
[0.1.3]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.3
[0.1.2]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.2
[0.1.1]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.1
[0.1.0]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.0
