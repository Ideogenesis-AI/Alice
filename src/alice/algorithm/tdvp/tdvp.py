# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Alice project.
#
# Alice is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Alice is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Alice. If not, see <https://www.gnu.org/licenses/>.


"""Time-Dependent Variational Principle (TDVP) driver.

TDVP evolves an MPS in real time by projecting the Schrödinger equation onto
the tangent space of the variational manifold:

    i d|ψ⟩/dt = P_{T_ψ} H |ψ⟩

and integrating the result by projector splitting. Each local update is a
Krylov matrix exponential (`krylov.py`) applied to the effective Hamiltonian
assembled from the environment blocks (`environ.py`), with the 1-site and
2-site updates defined in `scheme_1s.py` and `scheme_2s.py` and ordered into
sweeps by `sweep.py`.

`run()` operates purely on state: it takes a starting MPS (bundled with its
`time` and `step` in an `Artifact`) and advances it by `opts.dt` repeatedly.
Preparing the initial state is the caller's responsibility:

    import alice.algorithm.tdvp as tdvp
    from alice import init_mps

    psi0 = init_mps(L, spc, ops, bond_dim=1, config=config,
                    dtype=torch.complex128)
    opts = tdvp.Options(dt=0.05, n_steps=40)
    summary, artifact = tdvp.run(tdvp.Artifact(state=psi0, time=0.0, step=0),
                                 mpo, opts)

Resuming an interrupted run is calling `run()` again with the `Artifact`
written to `tdvp.ckpt`; the matching time/energy history is recovered
automatically from `evolution.ckpt` in `opts.checkpoint_dir`:

    resumed = tdvp.Artifact.load(ckpt_dir / 'tdvp.ckpt')
    summary, artifact = tdvp.run(resumed, mpo, opts)

Continuing a run that already finished works the same way, starting from any
archived `artifacts/step_XXXXX.ckpt` and new options — e.g. evolving further,
at a larger bond dimension:

    state = tdvp.Artifact.load(ckpt_dir / 'artifacts' / 'step_00040.ckpt')
    opts = tdvp.Options(dt=0.05, n_steps=80, max_bond=512,
                        checkpoint_dir=str(ckpt_dir))
    summary, artifact = tdvp.run(state, mpo, opts)

`n_steps` counts steps from the start of the history, not additional steps to
perform, and `dt` must match the step the recovered history was built on.
Starting from a step before the end of that history is allowed: the entries
past that step are truncated, then recomputed and overwritten by this run.

Which states get archived is set by `save_artifacts_every`: step `k` is
written when `k % save_artifacts_every == 0`. A long trajectory rarely wants
every step on disk, and the stride is keyed on the absolute step index so a
resumed run archives the same grid as an uninterrupted one.

Choosing a scheme
-----------------
`'2s'` is the default and the right choice for a quench. `'1s'` is cheaper and
strictly norm- and energy-conserving, but it cannot change bond dimensions,
and across a bond of dimension 1 it cannot transport correlation at all. A
product state has every bond at dimension 1, so a `'1s'` quench from one does
not evolve — silently, since the conserved quantities stay conserved. `run()`
warns when it is handed such a state.
"""

from __future__ import annotations

import logging
import math
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from alice.network import MPO, MPS, Network, observe

from ..interface import AlgorithmOptions, AlgorithmSummary
from .environ import (
    Environment,
    build_left_envs,
    build_right_envs,
    left_env_boundary,
    right_env_boundary,
)
from .sweep import time_step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheme alias resolution
# ---------------------------------------------------------------------------

_SCHEME_ALIASES: dict[str, str] = {
    '1s': '1s',
    '1-site': '1s',
    'one-site': '1s',
    '2s': '2s',
    '2-site': '2s',
    'two-site': '2s',
    '1sp': '1sp',
    '1-site-plus': '1sp',
    'one-site-plus': '1sp',
}

_IMPLEMENTED_SCHEMES = {'1s', '2s'}


def _resolve_scheme(alias: str) -> str:
    """Normalize a scheme alias to its canonical name.

    Parameters
    ----------
    alias:
        User-provided scheme string (from TOML or direct construction).

    Returns
    -------
    str
        Canonical scheme name (`'1s'`, `'2s'`, or `'1sp'`).

    Raises
    ------
    ValueError
        If `alias` is not a recognized scheme name.
    """
    canonical = _SCHEME_ALIASES.get(alias.lower())
    if canonical is None:
        known = ', '.join(sorted(_SCHEME_ALIASES))
        raise ValueError(
            f"unknown TDVP scheme {alias!r}; recognized values are: {known}"
        )
    return canonical


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class Options(AlgorithmOptions):
    """TDVP run options.

    All fields have sensible defaults so `Options()` is a valid minimal
    configuration. Use `Options.from_toml` to load from a TOML section, or
    `Options.load_toml` to read directly from a file.

    Parameters
    ----------
    scheme:
        Update scheme per time step. Canonical values and aliases:

        - `'2s'` / `'2-site'` / `'two-site'`: 2-site update with SVD
          truncation. The default, because it is the only scheme that can
          change bond dimensions, and therefore the only one that can follow
          a quench from a low-entanglement state.
        - `'1s'` / `'1-site'` / `'one-site'`: 1-site update. Cheaper, and
          conserves norm and energy to roundoff, but holds every bond
          dimension fixed. See the module docstring for the product-state
          trap this creates.
    dt:
        Time increment per step. May be negative to evolve backwards, which
        is how a run is reversed to check accumulated error.
    n_steps:
        Target step index counted from the start of the history, not a number
        of additional steps to perform. A run from `step=0` to `n_steps=40`
        reaches `t = 40 dt`; resuming at `step=20` with the same `n_steps`
        performs the remaining 20.
    order:
        Integrator order, `1` or `2`.

        - `2` (default, Lubich symmetric): forward half-sweep at `dt/2`, then
          a backward half-sweep at `dt/2`. The step is its own adjoint and
          leaves the orthogonality center back at site 0.
        - `1` (Lie-Trotter): a single half-sweep at `dt`, alternating
          direction between steps.

        Both are globally second order at equal cost — two consecutive
        `order=1` steps are exactly one `order=2` step of twice the size. The
        difference is that every `order=2` snapshot lands at the end of a
        complete forward-and-backward sweep pair, whereas `order=1` snapshots
        after an odd step carry first-order error.
    max_bond:
        Maximum bond dimension. `None` means unlimited. Meaningful for `'2s'`
        only; the `'1s'` update never changes a bond dimension, so no
        truncation is applied there at all.
    trunc_thresh:
        Singular value truncation threshold (relative to the largest singular
        value per charge sector). As for `max_bond`, `'2s'` only.
    krylov_dim:
        Maximum Krylov subspace dimension per local exponential. The solver
        stops earlier on happy breakdown or once `krylov_tol` is met.
    krylov_tol:
        Convergence tolerance for the local matrix exponential, measured as
        the weight the propagator places on the newest Krylov direction.
    renormalize:
        If `True`, rescale the state to unit norm after every step. `False`
        by default so that norm loss from truncation stays visible in
        `Summary.norms` instead of being silently absorbed; the `'1s'` scheme
        conserves the norm regardless.
    env_cache_dir:
        Root directory for environment disk caching. When set, `run()`
        creates a unique subdirectory inside it (first 8 hex characters of a
        UUID4, e.g. `{env_cache_dir}/a1b2c3d4/`) so that concurrent runs
        sharing the same config do not overwrite each other's blocks. Inside
        that subdirectory, `left/{i:05d}.pt` and `right/{i:05d}.pt` files are
        written, and are reused across all time steps of the run. The unique
        subdirectory is removed automatically when `run()` returns (or
        raises). `None` (default) keeps all blocks in memory. Stored as `str`
        for TOML compatibility.
    env_async_io:
        If `True` (default), disk writes are submitted asynchronously so they
        overlap with computation. Has no effect when `env_cache_dir` is
        `None`.
    env_window:
        Number of environment blocks to keep in memory at once when disk
        caching is enabled.
    checkpoint_dir:
        Directory for checkpoint files. After every time step an
        `evolution.ckpt` file (PyTorch format, loadable via
        `tdvp.Summary.load`) is written using an atomic write
        (`evolution_lock.ckpt` → rename). Mid-run progress is stored as
        `tdvp.ckpt` (an `Artifact`) and removed when `run()` finishes
        successfully. `None` (default) resolves to `Path.cwd()` at the time
        `run()` is called, mirroring `.logging`. Stored as `str` for TOML
        compatibility.
    artifacts_dir:
        Directory for archived `Artifact` files (`step_XXXXX.ckpt`), written
        when `save_artifacts` is `True`. `None` (default) resolves to
        `artifacts/` under `checkpoint_dir`. Stored as `str` for TOML
        compatibility.
    save_artifacts:
        If `True` (default), write `Artifact` files under `artifacts_dir` for
        every step on the `save_artifacts_every` grid. Step `0` is the
        initial state; step `k` is after the `k`-th time step. These
        snapshots are the only record of the state along the trajectory,
        since `Summary` deliberately carries no MPS.
    save_artifacts_every:
        Stride between archived steps: step `k` is written when
        `k % save_artifacts_every == 0`. Must be `>= 1`; `1` (the default)
        archives every step. The stride is keyed on the absolute step index
        rather than on an offset from where the run started, so a resumed run
        archives the same grid as an uninterrupted one.

        A run never evolves past the last grid point. When `n_steps` is not a
        multiple of the stride, `run()` stops at the largest multiple that is
        not greater than it and warns, which keeps `artifacts_dir` an exact
        grid whose final entry is also the run's resumable endpoint. No
        clamping happens when `save_artifacts` is `False`.
    """

    scheme: str = '2s'
    dt: float = 0.05
    n_steps: int = 20
    order: int = 2
    max_bond: Optional[int] = None
    trunc_thresh: float = 1e-12
    krylov_dim: int = 16
    krylov_tol: float = 1e-12
    renormalize: bool = False
    env_cache_dir: Optional[str] = None
    env_async_io: bool = True
    env_window: int = 2
    checkpoint_dir: Optional[str] = None
    artifacts_dir: Optional[str] = None
    save_artifacts: bool = True
    save_artifacts_every: int = 1

    def __post_init__(self) -> None:
        self.scheme = _resolve_scheme(self.scheme)
        if self.order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {self.order}")
        if self.dt == 0.0:
            raise ValueError("dt must be nonzero; a zero step advances no time")
        if self.krylov_dim < 1:
            raise ValueError(f"krylov_dim must be >= 1, got {self.krylov_dim}")
        # The matching clamp of `n_steps` onto this grid lives in `run()`, not
        # here: rewriting a field in __post_init__ would make the options
        # silently disagree with the TOML they were loaded from.
        if self.save_artifacts_every < 1:
            raise ValueError(
                f"save_artifacts_every must be >= 1, got {self.save_artifacts_every}"
            )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class Summary(AlgorithmSummary):
    """TDVP trajectory summary (no state).

    States are returned separately as `Artifact` (and optionally archived
    under `artifacts/` when `Options.save_artifacts` is enabled).

    Every list indexed "per step" has one entry per *recorded* step including
    the initial one, so `times`, `energies`, and `norms` all have length
    `n_steps + 1` for a run started from step 0. The diagnostics produced by
    the stepping itself — `discarded_weights` and `krylov_errors` — have one
    entry per step actually performed, so they are one shorter.

    Parameters
    ----------
    times:
        Physical time at each recorded step.
    energies:
        `⟨ψ|H|ψ⟩` at each recorded step. Conserved exactly by `'1s'`; drifts
        under `'2s'` only through truncation.
    norms:
        `‖ψ‖` at each recorded step. Constant unless `'2s'` truncation
        discards weight, which makes this the most direct measure of
        accumulated truncation error over the trajectory.
    max_bonds:
        Largest bond dimension of the state at each recorded step.
    discarded_weights:
        Discarded weight at the mid-chain bond for each step performed
        (`'2s'` only; `0.0` entries for `'1s'`).
    krylov_errors:
        Largest local Krylov error estimate over each step performed. A value
        approaching `Options.krylov_tol` from above across many steps means
        `krylov_dim` is the binding constraint on local accuracy.
    bond_dims:
        Bond dimension profile of the final state, one entry per bond
        (length `L - 1`).
    finished:
        `True` only for the summary returned by a completed `run()` call.
        `False` for mid-run `evolution.ckpt` snapshots written after an
        intermediate step (e.g. if the process is interrupted).
    n_steps:
        Number of time steps reflected in this summary.
    """

    times: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    norms: list[float] = field(default_factory=list)
    max_bonds: list[int] = field(default_factory=list)
    discarded_weights: list[float] = field(default_factory=list)
    krylov_errors: list[float] = field(default_factory=list)
    bond_dims: list[int] = field(default_factory=list)
    finished: bool = True
    n_steps: int = 0

    def serialize(self) -> dict:
        """Serialize the summary to a plain dict compatible with `torch.save`.

        Returns
        -------
        dict
            Serialized summary (version 1; no state).
        """
        return {
            'version': 1,
            'times': self.times,
            'energies': self.energies,
            'norms': self.norms,
            'max_bonds': self.max_bonds,
            'discarded_weights': self.discarded_weights,
            'krylov_errors': self.krylov_errors,
            'bond_dims': self.bond_dims,
            'finished': self.finished,
            'n_steps': self.n_steps,
        }

    @classmethod
    def deserialize(cls, data: dict, device: str = 'cpu') -> Summary:
        """Reconstruct a `Summary` from a dict produced by `serialize`.

        Parameters
        ----------
        data:
            Dict previously returned by `serialize`.
        device:
            Unused for version-1 summaries (no tensors). Accepted for API
            compatibility with `AlgorithmSummary.load`.

        Returns
        -------
        Summary
            Reconstructed trajectory summary.

        Raises
        ------
        ValueError
            If `data["version"]` is not `1`.
        """
        del device  # no tensors in version 1
        version = data.get('version', 1)
        if version != 1:
            raise ValueError(f"Unsupported Summary serialization version: {version!r}")

        return cls(
            times=list(data['times']),
            energies=list(data['energies']),
            norms=list(data['norms']),
            max_bonds=list(data['max_bonds']),
            discarded_weights=list(data.get('discarded_weights', [])),
            krylov_errors=list(data.get('krylov_errors', [])),
            bond_dims=list(data.get('bond_dims', [])),
            finished=bool(data.get('finished', True)),
            n_steps=int(data.get('n_steps', 0)),
        )


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

@dataclass
class Artifact(AlgorithmSummary):
    """State snapshot at one TDVP time step.

    Parameters
    ----------
    state:
        Evolved MPS `|ψ(t)⟩`.
    time:
        Physical time of this snapshot.
    step:
        Step index: `0` for the initial state; `k` after the `k`-th time step.
    """

    state: MPS
    time: float
    step: int

    def serialize(self) -> dict:
        """Serialize the artifact to a plain dict compatible with `torch.save`.

        Returns
        -------
        dict
            Serialized artifact.
        """
        return {
            'version': 1,
            'step': self.step,
            'time': self.time,
            'state': self.state.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict, device: str = 'cpu') -> Artifact:
        """Reconstruct an `Artifact` from a dict produced by `serialize`.

        Parameters
        ----------
        data:
            Dict previously returned by `serialize`.
        device:
            Device to place all tensor blocks on. Defaults to `'cpu'`.

        Returns
        -------
        Artifact
            Reconstructed artifact with the state on `device`.

        Raises
        ------
        ValueError
            If `data["version"]` is not `1`.
        """
        version = data.get('version', 1)
        if version != 1:
            raise ValueError(f"Unsupported Artifact serialization version: {version!r}")

        return cls(
            state=Network.deserialize(data['state'], device=device),
            time=float(data['time']),
            step=int(data['step']),
        )


# ---------------------------------------------------------------------------
# Checkpoint / artifact helpers
# ---------------------------------------------------------------------------

def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Atomically write `payload` via a sibling `*_lock` file then rename."""
    import torch

    lock_path = path.with_name(path.stem + '_lock' + path.suffix)
    torch.save(payload, lock_path)
    # Atomic rename: on POSIX this is guaranteed to be atomic; on Windows it
    # is best-effort (Path.replace uses MoveFileExW which is not atomic but
    # still avoids leaving a half-written target on disk).
    lock_path.replace(path)


def _save_evolution(summary: Summary, ckpt_dir: Path) -> None:
    """Write `evolution.ckpt` atomically in `ckpt_dir`."""
    _atomic_torch_save(summary.serialize(), ckpt_dir / 'evolution.ckpt')


def _save_checkpoint(artifact: Artifact, ckpt_dir: Path) -> None:
    """Write `tdvp.ckpt` atomically in `ckpt_dir`."""
    _atomic_torch_save(artifact.serialize(), ckpt_dir / 'tdvp.ckpt')


def _archive_artifact(artifact: Artifact, opts: Options, artifacts_dir: Path) -> None:
    """Archive `artifact` as `step_XXXXX.ckpt` when it falls on the grid."""
    if opts.save_artifacts and artifact.step % opts.save_artifacts_every == 0:
        # Five digits so a long run's archives still sort lexicographically.
        path = artifacts_dir / f'step_{artifact.step:05d}.ckpt'
        _atomic_torch_save(artifact.serialize(), path)


def _bond_profile(mps: MPS) -> List[int]:
    """Return the bond dimension of every internal bond, left to right."""
    return [mps[i].indices[1].dim for i in range(mps.L - 1)]


def _build_summary(
    times: list[float],
    energies: list[float],
    norms: list[float],
    max_bonds: list[int],
    discarded_weights: list[float],
    krylov_errors: list[float],
    bond_dims: list[int],
    step: int,
    finished: bool,
) -> Summary:
    """Assemble a state-free `Summary` from the current trajectory history."""
    return Summary(
        times=list(times),
        energies=list(energies),
        norms=list(norms),
        max_bonds=list(max_bonds),
        discarded_weights=list(discarded_weights),
        krylov_errors=list(krylov_errors),
        bond_dims=list(bond_dims),
        finished=finished,
        n_steps=step,
    )


def _resume_history(
    state: Artifact,
    ckpt_dir: Path,
    dt: float,
) -> Tuple[list[float], list[float], list[float], list[int], list[float], list[float]]:
    """Recover the trajectory history up to `state.step` from `evolution.ckpt`.

    The history is read from `evolution.ckpt` in `ckpt_dir`, where every prior
    call to `run()` wrote it alongside `tdvp.ckpt`. A history reaching past
    `state.step` is truncated rather than rejected, so a run can be continued
    from any archived `artifacts/step_XXXXX.ckpt` — not only from the step the
    previous run stopped at. The steps beyond `state.step` are then recomputed
    by the caller and overwrite the truncated ones on disk.

    Parameters
    ----------
    state:
        Starting MPS with its `time` and `step`.
    ckpt_dir:
        Resolved checkpoint directory holding `evolution.ckpt`.
    dt:
        Time increment from the run options, checked against the spacing of
        the recovered history.

    Returns
    -------
    list[float]
        Times for steps `0 … state.step`.
    list[float]
        Energies for steps `0 … state.step`.
    list[float]
        Norms for steps `0 … state.step`.
    list[int]
        Maximum bond dimensions for steps `0 … state.step`.
    list[float]
        Discarded weights for the `state.step` steps performed so far.
    list[float]
        Krylov errors for the `state.step` steps performed so far.

    Raises
    ------
    FileNotFoundError
        If no `evolution.ckpt` exists in `ckpt_dir`.
    ValueError
        If the history stops before `state.step`, if its time at `state.step`
        does not match `state.time`, or if its step spacing does not match
        `dt`.
    """
    evo_path = ckpt_dir / 'evolution.ckpt'
    if not evo_path.exists():
        raise FileNotFoundError(
            f"resuming from step {state.step} requires an existing evolution.ckpt "
            f"in {ckpt_dir} to recover the time/energy history, but none was found"
        )
    history = Summary.load(evo_path)

    if len(history.times) <= state.step:
        raise ValueError(
            f"evolution.ckpt at {ckpt_dir} only reaches step {len(history.times) - 1}, "
            f"but the given state is at step {state.step}; the two are inconsistent"
        )
    # Compare the time at the state's own step (not at the end of the
    # history), so a history reaching further can still be truncated back.
    if not math.isclose(history.times[state.step], state.time, abs_tol=1e-12):
        raise ValueError(
            f"evolution.ckpt's time at step {state.step} "
            f"({history.times[state.step]:.6g}) at {ckpt_dir} does not match the "
            f"given state's time ({state.time:.6g}); the two are inconsistent"
        )
    if len(history.times) > 1:
        history_dt = history.times[1] - history.times[0]
        if not math.isclose(history_dt, dt, rel_tol=1e-9):
            raise ValueError(
                f"evolution.ckpt at {ckpt_dir} was built with dt = {history_dt:.6g}, "
                f"but the given options specify dt = {dt:.6g}; the two are inconsistent"
            )

    last = len(history.times) - 1
    if last > state.step:
        logger.warning(
            "evolution.ckpt reaches step %d; continuing from step %d — the %d later "
            "entries will be recomputed and overwritten",
            last, state.step, last - state.step,
        )

    k = state.step
    return (
        list(history.times[:k + 1]),
        list(history.energies[:k + 1]),
        list(history.norms[:k + 1]),
        list(history.max_bonds[:k + 1]),
        # One diagnostic per step performed, so `k` entries cover the steps
        # that produced times[1] … times[k].
        list(history.discarded_weights[:k]),
        list(history.krylov_errors[:k]),
    )


# ---------------------------------------------------------------------------
# State preparation
# ---------------------------------------------------------------------------

def _promote_to_complex(mps: MPS, mpo: MPO) -> None:
    """Promote state and Hamiltonian to `complex128` in place, if needed.

    Real-time evolution multiplies by `exp(-i dt H)`, so a real state cannot
    represent the result. A real MPO contracts against the complex state
    perfectly well, but casting it once here saves re-casting every MPO site
    at every step of the environment sweep.
    """
    promoted = []
    if not all(t.dtype.is_complex for t in mps):
        mps.to(dtype=torch.complex128)
        promoted.append('state')
    if not all(t.dtype.is_complex for t in mpo):
        mpo.to(dtype=torch.complex128)
        promoted.append('Hamiltonian')
    if promoted:
        logger.info("  promoted %s to complex128 for real-time evolution",
                    ' and '.join(promoted))


def _warn_frozen_bonds(mps: MPS) -> None:
    """Warn when a 1-site run starts from a state with dimension-1 bonds.

    The 1-site update cannot change a bond dimension, and across a bond of
    dimension 1 it cannot transport correlation at all. Norm and energy stay
    conserved while this happens, so without a warning the run looks healthy
    and produces frozen dynamics.
    """
    frozen = [i for i, d in enumerate(_bond_profile(mps)) if d == 1]
    if not frozen:
        return

    shown = ', '.join(str(i) for i in frozen[:8])
    more = '' if len(frozen) <= 8 else f", … ({len(frozen)} total)"
    logger.warning(
        "scheme '1s' cannot grow bond dimensions, and bond(s) %s%s have "
        "dimension 1; no correlation can cross them, so the dynamics there "
        "will be frozen while norm and energy stay conserved. Use scheme "
        "'2s', or warm up with '2s' until every bond has opened.",
        shown, more,
    )


def _prepare_state(mps: MPS, opts: Options) -> bool:
    """Canonicalize the state for the first sweep and report its direction.

    Returns `True` when the next half-sweep runs left to right. For `order=2`
    this is always the case. For `order=1` the direction alternates, and a
    resumed state records where it stopped in its own orthogonality center:
    a center at the right edge means a forward half-sweep already ran and the
    next one must come back.
    """
    if opts.order == 1 and mps.center == mps.L - 1:
        return False

    if mps.center != 0:
        mps.canonical(0)
    return True


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(
    state: Artifact,
    mpo: MPO,
    opts: Optional[Options] = None,
) -> Tuple[Summary, Artifact]:
    """Evolve an MPS in real time under a Hamiltonian MPO.

    Advances `state.state` by `opts.dt` per step, starting at `state.step` and
    continuing to `opts.n_steps`. Preparing the initial state and wrapping it
    in an `Artifact` at `step=0` is the caller's responsibility. Resuming an
    interrupted run, or continuing a finished one under new options, is
    calling `run()` again with the `Artifact` loaded from `tdvp.ckpt` or from
    any archived `artifacts/step_XXXXX.ckpt`; the matching time/energy history
    is recovered automatically from `evolution.ckpt` in `opts.checkpoint_dir`.

    When that history reaches past `state.step` — the case when restarting
    from an earlier archived step to redo a segment at, say, a larger
    `max_bond` — the later entries are truncated (with a warning), then
    recomputed by this run and overwritten on disk, together with their
    `step_XXXXX.ckpt` archives.

    Both `state.state` and `mpo` are promoted to `complex128` in place when
    they are not already complex.

    Parameters
    ----------
    state:
        Starting MPS, paired with its `time` and `step`. `step=0` for a fresh
        run; `step > 0` to resume or continue.
    mpo:
        Hamiltonian MPO. Must be time-independent and Hermitian; the Krylov
        solver assumes a Hermitian effective Hamiltonian at every site.
    opts:
        TDVP run options. Defaults to `Options()` if `None`. `n_steps` counts
        steps from the start of the history, so it is the absolute step index
        to stop at — not a number of additional steps to perform. When
        artifacts are being saved and `n_steps` is not a multiple of
        `save_artifacts_every`, the run stops at the largest multiple that is
        not greater than it and warns.

    Returns
    -------
    Summary
        Trajectory history (times, energies, norms, bond dimensions, and
        per-step diagnostics). Written to `evolution.ckpt` under the
        checkpoint directory.
    Artifact
        Final state with its `time` and `step`.

    Raises
    ------
    ValueError
        If `opts.scheme` is not a recognized scheme, if `state.step` is past
        `opts.n_steps`, or if `state.step > 0` and the `evolution.ckpt` found
        in `opts.checkpoint_dir` is inconsistent with `state` — because it
        stops before `state.step`, because its time there disagrees with
        `state.time`, or because it was built with a different `dt`.
    NotImplementedError
        If `opts.scheme` is recognized but not yet implemented.
    FileNotFoundError
        If `state.step > 0` and no `evolution.ckpt` exists in
        `opts.checkpoint_dir` to recover the history from.
    """
    if opts is None:
        opts = Options()

    if opts.scheme not in _IMPLEMENTED_SCHEMES:
        raise NotImplementedError(
            f"TDVP scheme {opts.scheme!r} is recognized but not yet implemented; "
            f"implemented schemes are: {', '.join(sorted(_IMPLEMENTED_SCHEMES))}"
        )

    # `n_steps` counts steps from the start of the history, so a target behind
    # the state would leave the written summary claiming fewer steps than its
    # own history carries. Equality is fine: the already-at-target no-op.
    if state.step > opts.n_steps:
        raise ValueError(
            f"the given state is at step {state.step}, past the target "
            f"opts.n_steps = {opts.n_steps}; note that n_steps counts steps from "
            "the start of the history, not additional steps to perform"
        )

    # Stop on a grid point: an endpoint between two archived steps would end
    # the run with no snapshot of its own final state, since tdvp.ckpt is
    # removed on success. The floor at `state.step` covers a continuation
    # whose start already sits past the clamped target (reachable when the
    # stride changed between runs) — that becomes the already-at-target
    # no-op, rather than a summary claiming fewer steps than its own history.
    n_target = opts.n_steps
    if opts.save_artifacts:
        n_target = max(
            (opts.n_steps // opts.save_artifacts_every) * opts.save_artifacts_every,
            state.step,
        )
        if n_target != opts.n_steps:
            logger.warning(
                "n_steps = %d is not a multiple of save_artifacts_every = %d; "
                "stopping at step %d instead so the final state lands on the "
                "artifact grid",
                opts.n_steps, opts.save_artifacts_every, n_target,
            )

    mps = state.state
    L = mps.L

    # Resolve the optional disk-cache directory. A unique subdirectory (first
    # 8 hex digits of a UUID4) is created inside the user-supplied path so
    # that concurrent runs sharing the same config do not collide. The same
    # subdirectory is reused by every time step of this run.
    _cache: Optional[Path] = None
    if opts.env_cache_dir:
        _cache = Path(opts.env_cache_dir) / uuid.uuid4().hex[:8]
        _cache.mkdir(parents=True, exist_ok=True)

    # Resolve the checkpoint directory; default to Path.cwd() when unset,
    # mirroring the convention used by configure_logging / DMRG.
    _ckpt: Path = Path(opts.checkpoint_dir) if opts.checkpoint_dir is not None else Path.cwd()
    _ckpt.mkdir(parents=True, exist_ok=True)
    # artifacts_dir defaults to artifacts/ under the checkpoint directory, but
    # may be pointed elsewhere (e.g. faster or larger storage) independent of
    # where checkpoints themselves live.
    _artifacts: Path = (
        Path(opts.artifacts_dir) if opts.artifacts_dir is not None else _ckpt / 'artifacts'
    )
    if opts.save_artifacts:
        _artifacts.mkdir(parents=True, exist_ok=True)

    max_bond_str = str(opts.max_bond) if opts.max_bond is not None else 'unlimited'
    t_final = state.time + (n_target - state.step) * opts.dt
    logger.info("─" * 60)
    logger.info("Commencing: TDVP Algorithm".center(60))
    logger.info("─" * 60)
    logger.info("")
    logger.info("  scheme            : %s", opts.scheme)
    logger.info("  integrator order  : %d", opts.order)
    logger.info("  chain length      : %d", L)
    logger.info("  time step         : %.6g", opts.dt)
    logger.info("  steps             : %d", n_target)
    if state.step > 0:
        logger.info("  resuming at step  : %d  (%d step(s) remaining)",
                    state.step, n_target - state.step)
    logger.info("  final time        : %.6g", t_final)
    logger.info("  max bond dim      : %s", max_bond_str)
    logger.info("  trunc thresh      : %.2e", opts.trunc_thresh)
    logger.info("  krylov dim        : %d", opts.krylov_dim)
    logger.info("  krylov tol        : %.2e", opts.krylov_tol)
    logger.info("  renormalize       : %s", opts.renormalize)
    if opts.checkpoint_dir is not None:
        logger.info("  checkpoint dir    : %s", _ckpt)
    if _cache is not None:
        logger.info("  env cache dir     : %s", _cache)
    if opts.save_artifacts:
        logger.info("  save artifacts    : True (every %d step(s))",
                    opts.save_artifacts_every)
        logger.info("  artifacts dir     : %s", _artifacts)
    logger.info("")

    _promote_to_complex(mps, mpo)
    if opts.scheme == '1s':
        _warn_frozen_bonds(mps)

    if state.step == 0:
        times: list[float] = [state.time]
        energies: list[float] = [observe(mps, mpo)]
        norms: list[float] = [mps.norm()]
        max_bonds: list[int] = [max(_bond_profile(mps))]
        discarded_weights: list[float] = []
        krylov_errors: list[float] = []
        logger.info("  t = %.6g,  E = %+.12g,  ‖ψ‖ = %.12g",
                    times[-1], energies[-1], norms[-1])
    else:
        # Resuming: the history up to `state.step` lives in evolution.ckpt,
        # alongside tdvp.ckpt (which is where `state` itself typically came
        # from). Both files are written together by every prior call to run().
        (times, energies, norms, max_bonds,
         discarded_weights, krylov_errors) = _resume_history(state, _ckpt, opts.dt)
        logger.info("  resuming from step %d / %d  (t = %.6g,  E = %+.12g)",
                    state.step, n_target, times[-1], energies[-1])

    artifact = Artifact(state=mps, time=times[-1], step=state.step)
    _save_checkpoint(artifact, _ckpt)
    _archive_artifact(artifact, opts, _artifacts)

    forward = _prepare_state(mps, opts)
    _2s = (opts.scheme == '2s')
    env_left = Environment(
        L,
        _cache / 'left' if _cache is not None else None,
        async_io=opts.env_async_io,
        window=opts.env_window,
        fetch_lo=0,
        fetch_hi=L - 2 if _2s else L - 1,
    )
    env_right = Environment(
        L,
        _cache / 'right' if _cache is not None else None,
        async_io=opts.env_async_io,
        window=opts.env_window,
        fetch_lo=1 if _2s else 0,
        fetch_hi=L - 1,
    )

    # Seed the environments at the end the first half-sweep starts from: a
    # sweep consumes the blocks that face away from its direction of travel
    # and rebuilds the others as it goes.
    if forward:
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)
    else:
        env_right[L - 1] = right_env_boundary(mps, mpo)
        build_left_envs(mps, mpo, env_left)

    w = len(str(n_target))
    try:
        for step in range(state.step, n_target):
            err, dw = time_step(mps, mpo, env_left, env_right, opts, forward)
            if opts.order == 1:
                forward = not forward
            if opts.renormalize:
                mps.normalize()

            k = step + 1
            times.append(times[-1] + opts.dt)
            energies.append(observe(mps, mpo))
            norms.append(mps.norm())
            max_bonds.append(max(_bond_profile(mps)))
            discarded_weights.append(dw)
            krylov_errors.append(err)

            logger.info(
                "step %*d / %d: t = %.6g,  E = %+.12g,  ‖ψ‖ = %.12g,  "
                "χ = %d,  dw = %.4e",
                w, k, n_target, times[-1], energies[-1], norms[-1],
                max_bonds[-1], dw,
            )
            logger.debug("  krylov err = %.4e", err)

            mid_summary = _build_summary(
                times, energies, norms, max_bonds, discarded_weights,
                krylov_errors, _bond_profile(mps), step=k, finished=False,
            )
            _save_evolution(mid_summary, _ckpt)

            artifact = Artifact(state=mps, time=times[-1], step=k)
            _save_checkpoint(artifact, _ckpt)
            _archive_artifact(artifact, opts, _artifacts)
    finally:
        env_left.shutdown()
        env_right.shutdown()
        # Remove the run-specific cache subdirectory; ignore errors so that a
        # partially-written or already-deleted directory does not mask the
        # real exception (if any) from the time loop.
        if _cache is not None:
            shutil.rmtree(_cache, ignore_errors=True)

    logger.info("")

    summary = _build_summary(
        times, energies, norms, max_bonds, discarded_weights, krylov_errors,
        _bond_profile(mps), step=n_target, finished=True,
    )
    _save_evolution(summary, _ckpt)

    # Success path only: drop the mid-run checkpoint so a finished job does
    # not leave a stale tdvp.ckpt behind. A crash earlier leaves it on disk.
    (_ckpt / 'tdvp.ckpt').unlink(missing_ok=True)
    (_ckpt / 'tdvp_lock.ckpt').unlink(missing_ok=True)

    return summary, artifact
