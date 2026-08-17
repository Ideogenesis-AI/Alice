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


"""eXponential Tensor Renormalization Group (XTRG) driver.

XTRG exponentially cools the thermal density matrix by repeated squaring:

    ρ(τ₀) → ρ(2τ₀) → ρ(4τ₀) → … → ρ(β_max)

Each squaring step ρ_{n+1} ≈ compress(ρ_n ⊗ ρ_n) is performed via the
variational MPO-MPO compression defined in `environ.py`, `scheme_1s.py`,
`scheme_2s.py`, and `sweep.py`.

Initialization uses `thermal_mpo()` to compute ρ(τ₀) via a Taylor expansion
of e^{-τ₀ H}, which is accurate for sufficiently small τ₀.

Thermodynamic observables (log Z, f, u, c_V, S) are extracted at each
cooling step using log-β finite differences, which give uniform accuracy
across the exponentially spaced temperature grid.
"""

from __future__ import annotations

import logging
import math
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from nicole import Index
from nicole import deserialize as _deserialize_tensor

from alice.network import MPO
from alice.network.thermal import NormalMPO, thermal_mpo

from ..interface import AlgorithmOptions, AlgorithmSummary
from .environ import Environment, build_right_envs, left_env_boundary
from .sweep import backward_sweep, forward_sweep

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

_IMPLEMENTED_SCHEMES = {'1s', '2s', '1sp'}


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
            f"unknown XTRG scheme {alias!r}; recognized values are: {known}"
        )
    return canonical


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class Options(AlgorithmOptions):
    """XTRG run options.

    All fields have sensible defaults so `Options()` is a valid minimal
    configuration. Use `Options.from_toml` to load from a TOML section, or
    `Options.load_toml` to read directly from a file.

    Parameters
    ----------
    scheme:
        Update scheme per squaring step. Canonical values and aliases:
        - `'1s'` / `'1-site'` / `'one-site'`: 1-site direct contraction.
        - `'2s'` / `'2-site'` / `'two-site'`: 2-site SVD with truncation.
        - `'1sp'` / `'1-site-plus'` / `'one-site-plus'`: 1-site-plus / CBE.
    tau_0:
        Initial inverse temperature ≈ 2⁻¹² ≈ 2.44 × 10⁻⁴. Should be small
        enough that the Taylor expansion converges, and is doubled at each
        step: β_n = 2^n × τ₀.
    n_steps:
        Number of doubling steps. The final inverse temperature is
        β_max = 2^n_steps × τ₀.
    taylor_order:
        Truncation order of the Taylor series for ρ(τ₀) = e^{-τ₀ H}.
        Higher order increases accuracy and bond dimension of the initial ρ.
    max_bond:
        Maximum bond dimension of the compressed ρ. `None` means unlimited
        (meaningful for 2-site and 1-site-plus, where SVD truncation
        controls growth; a no-op for 1-site beyond the initial compaction,
        since its local update never changes bond dimension).
    trunc_thresh:
        Singular value truncation threshold (relative to the largest singular
        value per charge sector).
    n_sweeps:
        Maximum number of full variational sweeps (forward + backward) per
        squaring step. A sweep may stop early once `z_tol` is satisfied.
    z_tol:
        Convergence tolerance for the inner variational fit: sweeping stops
        early once `|‖C‖ − ‖C_prev‖| < z_tol`, where `‖C‖` is the Frobenius
        norm of the compressed density matrix (proportional to the
        partition function Z) measured at the orthogonality center after
        each full sweep. Plays the same structural role as DMRG's `e_tol`,
        but tracks fit convergence via ‖C‖ rather than energy, since the
        local update here is an exact least-squares projection (no
        eigenproblem): at that optimum `⟨C, A·B⟩ = ‖C‖²`, so
        `‖C − A·B‖²_F = ‖A·B‖² − ‖C‖²` and `‖A·B‖` is fixed across sweeps,
        making `‖C‖` convergence equivalent to residual convergence.
    env_cache_dir:
        Root directory for environment disk caching. When set, `run()`
        creates a unique subdirectory inside it (first 8 hex characters of a
        UUID4, e.g. `{env_cache_dir}/a1b2c3d4/`) so that concurrent runs
        sharing the same config do not overwrite each other's blocks. Inside
        that subdirectory, `xtrg_left/{i:05d}.pt` and `xtrg_right/{i:05d}.pt`
        files are written, and are reused across all squaring steps of the
        run. The unique subdirectory is removed automatically when `run()`
        returns (or raises). `None` (default) keeps all blocks in memory.
        Useful for large chains where environments do not fit in RAM.
        Stored as `str` for TOML compatibility.
    env_async_io:
        If `True` (default), disk writes are submitted asynchronously so
        they overlap with computation. Has no effect when `env_cache_dir`
        is `None`.
    env_window:
        Number of environment blocks to keep in memory at once when disk
        caching is enabled.
    expand_k:
        Maximum number of complement vectors added to each bond end per CBE
        step. Only used when `scheme = '1sp'`. Larger values give a richer
        expanded space at higher cost; `4` is a typical starting point.
    expand_alpha:
        Internal connector-bond dimension used by the cheap per-operand SVD
        compression inside the CBE expansion (Eq. 13 of arXiv:2510.25022).
        Only used when `scheme = '1sp'`. `None` skips compression (uses the
        exact factor-MPO tensors, at the cost of a full 2-site-scale join).
        The paper recommends `expand_alpha ≈ expand_k ≈ round(sqrt(max_bond))`.
    checkpoint_dir:
        Directory for checkpoint and artifact files. After every cooling
        step a `thermal.ckpt` file (PyTorch format, loadable via
        `xtrg.Summary.load`) is written using an atomic write
        (`thermal_lock.ckpt` → rename). Mid-run progress is stored as
        `progress.ckpt` (an `Artifact`) and removed when `run()` finishes
        successfully. When `save_artifacts` is `True`, per-step density
        matrices are also archived under `artifacts/step_XX.ckpt`.
        `None` (default) resolves to `Path.cwd()` at the time `run()` is
        called, mirroring `.logging`. Pass an explicit path string to write
        elsewhere. Stored as `str` for TOML compatibility.
    save_artifacts:
        If `True` (default), write per-step `Artifact` files under
        `artifacts/` in the checkpoint directory for every step with index
        `>= save_artifacts_since`. Step `0` is after Taylor init (`ρ(τ₀)`);
        step `k` (`1 … n_steps`) is after the `k`-th squaring.
    save_artifacts_since:
        First step index (inclusive) at which `artifacts/step_XX.ckpt` files
        are written when `save_artifacts` is `True`. Must be `>= 0`.
    """

    scheme: str = '2s'
    tau_0: float = 2 ** -12
    n_steps: int = 20
    taylor_order: int = 10
    max_bond: Optional[int] = None
    trunc_thresh: float = 1e-15
    n_sweeps: int = 4
    z_tol: float = 1e-10
    env_cache_dir: Optional[str] = None
    env_async_io: bool = True
    env_window: int = 2
    expand_k: int = 4
    expand_alpha: Optional[int] = None
    checkpoint_dir: Optional[str] = None
    save_artifacts: bool = True
    save_artifacts_since: int = 0

    def __post_init__(self) -> None:
        self.scheme = _resolve_scheme(self.scheme)
        if self.save_artifacts_since < 0:
            raise ValueError(
                f"save_artifacts_since must be >= 0, got {self.save_artifacts_since}"
            )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class Summary(AlgorithmSummary):
    """XTRG thermodynamic summary (no density matrix).

    Density matrices are returned separately as `Artifact` (and optionally
    archived under `artifacts/` when `Options.save_artifacts` is enabled).

    Parameters
    ----------
    betas:
        List of β values at each cooling step: [τ₀, 2τ₀, …, 2^n_steps τ₀].
    log_z:
        `log Z(β_n)` at each step, computed via `NormalMPO.log_trace()`
        (rather than `log(NormalMPO.trace())`) so that it remains finite
        even when `Z(β_n)` itself is far outside float64 range.
    free_energies:
        Free energy per site: f(β_n) = −log Z(β_n) / (β_n L).
    energies:
        Internal energy per site: u(β_n) estimated via log-β finite
        differences (uniform accuracy on the exponential grid).
    specific_heats:
        Specific heat per site: c_V(β_n) estimated via log-β finite
        differences.
    entropies:
        Entropy per site: S(β_n) = β_n (u(β_n) − f(β_n)).
    discarded_weights:
        Per-step discarded weight from the variational compression (2-site
        only; `0.0` entries for 1-site). Length equals the number of
        squaring steps (`n_steps`), not the length of `betas`.
    finished:
        `True` only for the summary returned by a completed `run()` call.
        `False` for mid-run `thermal.ckpt` snapshots written after an
        intermediate squaring step (e.g. if the process is interrupted).
    n_steps:
        Number of cooling (squaring) steps reflected in this summary.
    """

    betas: list[float] = field(default_factory=list)
    log_z: list[float] = field(default_factory=list)
    free_energies: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    specific_heats: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    discarded_weights: list[float] = field(default_factory=list)
    finished: bool = True
    n_steps: int = 0

    def serialize(self) -> dict:
        """Serialize the summary to a plain dict compatible with `torch.save`.

        Returns
        -------
        dict
            Serialized summary (version 2; no density matrix).
        """
        return {
            'version': 2,
            'betas': self.betas,
            'log_z': self.log_z,
            'free_energies': self.free_energies,
            'energies': self.energies,
            'specific_heats': self.specific_heats,
            'entropies': self.entropies,
            'discarded_weights': self.discarded_weights,
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
            Unused for version-2 summaries (no tensors). Accepted for API
            compatibility with `AlgorithmSummary.load`.

        Returns
        -------
        Summary
            Reconstructed thermodynamic summary.

        Raises
        ------
        ValueError
            If `data["version"]` is not `2`.
        """
        del device  # no tensors in version 2
        version = data.get('version', 1)
        if version != 2:
            raise ValueError(f"Unsupported Summary serialization version: {version!r}")

        return cls(
            betas=list(data['betas']),
            log_z=list(data['log_z']),
            free_energies=list(data['free_energies']),
            energies=list(data['energies']),
            specific_heats=list(data['specific_heats']),
            entropies=list(data['entropies']),
            discarded_weights=list(data.get('discarded_weights', [])),
            finished=bool(data.get('finished', True)),
            n_steps=int(data.get('n_steps', 0)),
        )


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

@dataclass
class Artifact(AlgorithmSummary):
    """Density-matrix snapshot at one XTRG cooling step.

    Parameters
    ----------
    rho:
        Thermal density matrix `ρ(β)` as a `NormalMPO`.
    beta:
        Inverse temperature of this snapshot.
    step:
        Step index: `0` after Taylor init; `k` after the `k`-th squaring.
    """

    rho: NormalMPO
    beta: float
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
            'beta': self.beta,
            'rho': self.rho.serialize(),
            'rho_log_scale': self.rho.log_scale,
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
            Reconstructed artifact with the density matrix on `device`.

        Raises
        ------
        ValueError
            If `data["version"]` is not `1`.
        """
        version = data.get('version', 1)
        if version != 1:
            raise ValueError(f"Unsupported Artifact serialization version: {version!r}")

        rho_data = data['rho']
        tensors = [_deserialize_tensor(t) for t in rho_data['tensors']]
        rho = NormalMPO(
            tensors,
            log_scale=float(data['rho_log_scale']),
            bc=rho_data['bc'],
            center=rho_data['center'],
        )
        del device  # tensors currently stay on the device used at save time
        return cls(
            rho=rho,
            beta=float(data['beta']),
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


def _build_summary(
    betas: list[float],
    log_z: list[float],
    discarded_weights: list[float],
    L: int,
    step: int,
    finished: bool,
) -> Summary:
    """Build a rho-free `Summary` from the current thermodynamic history."""
    free_energies, energies, specific_heats, entropies = _compute_observables(
        betas, log_z, L,
    )
    return Summary(
        betas=list(betas),
        log_z=list(log_z),
        free_energies=free_energies,
        energies=energies,
        specific_heats=specific_heats,
        entropies=entropies,
        discarded_weights=list(discarded_weights),
        finished=finished,
        n_steps=step,
    )


def _save_thermal(summary: Summary, ckpt_dir: Path) -> None:
    """Write `thermal.ckpt` atomically in `ckpt_dir`."""
    _atomic_torch_save(summary.serialize(), ckpt_dir / 'thermal.ckpt')


def _save_progress(artifact: Artifact, ckpt_dir: Path) -> None:
    """Write `progress.ckpt` atomically in `ckpt_dir`."""
    _atomic_torch_save(artifact.serialize(), ckpt_dir / 'progress.ckpt')


def _save_artifact_file(artifact: Artifact, artifacts_dir: Path) -> None:
    """Write `artifacts/step_XX.ckpt` atomically for `artifact.step`."""
    path = artifacts_dir / f'step_{artifact.step:02d}.ckpt'
    _atomic_torch_save(artifact.serialize(), path)


def _archive_artifact(
    artifact: Artifact,
    opts: Options,
    artifacts_dir: Path,
) -> None:
    """Archive `artifact` under `artifacts/` when options request it."""
    if opts.save_artifacts and artifact.step >= opts.save_artifacts_since:
        _save_artifact_file(artifact, artifacts_dir)


# ---------------------------------------------------------------------------
# Inner variational compression
# ---------------------------------------------------------------------------

def _fit_mpo(
    mpo_a: NormalMPO,
    mpo_b: NormalMPO,
    opts: Options,
    cache_dir: Optional[Path] = None,
) -> Tuple[NormalMPO, float]:
    """Compress mpo_a @ mpo_b variationaly into a lower-bond-dim NormalMPO.

    Runs up to `opts.n_sweeps` full variational sweeps (each = forward +
    backward half-sweep) to find C ≈ mpo_a · mpo_b by minimizing
    ‖C − A·B‖²_F, stopping early once `opts.z_tol` is satisfied (see
    `Options.z_tol`).

    Parameters
    ----------
    mpo_a:
        Left factor MPO.
    mpo_b:
        Right factor MPO.
    opts:
        XTRG options (scheme, max_bond, trunc_thresh, n_sweeps, z_tol,
        env_*).
    cache_dir:
        Run-specific directory for environment disk caching, already made
        unique by the caller. `None` (default) keeps all blocks in memory,
        regardless of `opts.env_cache_dir`; the resolution of that option
        into a collision-free path is the caller's responsibility.

    Returns
    -------
    NormalMPO
        Compressed and normalized result.
    float
        Discarded weight at the center bond from the last backward sweep
        (0.0 for 1-site scheme).
    """
    L = mpo_a.L
    trunc: Optional[dict] = {'thresh': opts.trunc_thresh}
    if opts.max_bond is not None:
        trunc['nkeep'] = opts.max_bond

    # Initialize mpo_c as a copy of mpo_a with right-canonical form (center=0).
    mpo_c = NormalMPO.from_mpo(mpo_a)
    # Apply initial truncation if a bond limit is set.
    if opts.max_bond is not None:
        mpo_c.compact(trunc)
    # Ensure center=0 for build_right_envs.
    if mpo_c.center != 0:
        mpo_c.canonical(0)

    _2s = (opts.scheme == '2s')
    if cache_dir is not None:
        (cache_dir / 'xtrg_left').mkdir(parents=True, exist_ok=True)
        (cache_dir / 'xtrg_right').mkdir(parents=True, exist_ok=True)

    env_left = Environment(
        L,
        cache_dir / 'xtrg_left' if cache_dir is not None else None,
        async_io=opts.env_async_io,
        window=opts.env_window,
        fetch_lo=0,
        fetch_hi=L - 2 if _2s else L - 1,
    )
    env_right = Environment(
        L,
        cache_dir / 'xtrg_right' if cache_dir is not None else None,
        async_io=opts.env_async_io,
        window=opts.env_window,
        fetch_lo=1 if _2s else 0,
        fetch_hi=L - 1,
    )

    # Build initial left boundary and all right environment blocks.
    env_left[0] = left_env_boundary(mpo_a, mpo_b, mpo_c)
    build_right_envs(mpo_a, mpo_b, mpo_c, env_right)

    dw = 0.0
    prev_norm: Optional[float] = None
    try:
        for sweep_idx in range(opts.n_sweeps):
            logger.debug("  compression sweep %d / %d", sweep_idx + 1, opts.n_sweeps)
            forward_sweep(mpo_a, mpo_b, mpo_c, env_left, env_right, opts)
            dw = backward_sweep(mpo_a, mpo_b, mpo_c, env_left, env_right, opts)

            # ‖C‖ at the orthogonality center (mpo_c.center == 0 after a
            # backward sweep) is a cheap, exact proxy for fit convergence:
            # since each local update is the exact least-squares projection,
            # ⟨C, A·B⟩ = ‖C‖² at the optimum, so ‖C − A·B‖²_F = ‖A·B‖² − ‖C‖²
            # and ‖A·B‖ is fixed across sweeps. No extra contraction needed —
            # `norm()` just reads the already-isometric center tensor.
            norm = mpo_c.norm()
            if prev_norm is not None and abs(norm - prev_norm) < opts.z_tol:
                logger.debug(
                    "  fit converged after %d sweep(s) (Δ‖C‖=%.3e)",
                    sweep_idx + 1, abs(norm - prev_norm),
                )
                break
            prev_norm = norm
    finally:
        env_left.shutdown()
        env_right.shutdown()

    # Normalize: compact() extracts the Frobenius norm of the compressed
    # internal tensors into log_scale and leaves the internal tensors at
    # unit norm.
    mpo_c.compact(trunc)

    # The sweeps operate entirely on the unit-norm internal tensors of mpo_a
    # and mpo_b, so the physical scale magnitudes of mpo_a and mpo_b are
    # absent from mpo_c's scale after compact(). Fold them in now (via
    # scale_by, which combines log-scales additively) so that mpo_c
    # represents the correct physical product A_phys @ B_phys. This is done
    # in log-space rather than by materializing mpo_a.scale * mpo_b.scale,
    # since the physical scale can be far outside float64 range deep into
    # an XTRG run. The sweeps fit mpo_c directly to mpo_a's and mpo_b's
    # tensor data, so any sign the two operands carry propagates through
    # to mpo_c automatically.
    mpo_c.scale_by(mpo_a.log_scale + mpo_b.log_scale)

    return mpo_c, dw


# ---------------------------------------------------------------------------
# Observable extraction
# ---------------------------------------------------------------------------

def _compute_observables(
    betas: list[float],
    log_z: list[float],
    L: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Derive thermodynamic observables from the log Z grid.

    Uses log-β finite differences to compute internal energy and specific heat,
    which give uniform O((ln 2)²) discretization error across the exponentially
    spaced β grid.

    Parameters
    ----------
    betas:
        List of inverse temperatures [τ₀, 2τ₀, …].
    log_z:
        `log Z(β_n)` at each point, same length as `betas`.
    L:
        Chain length (for per-site normalization).

    Returns
    -------
    list[float]
        Free energies f(β) per site.
    list[float]
        Internal energies u(β) per site.
    list[float]
        Specific heats c_V(β) per site.
    list[float]
        Entropies S(β) per site.
    """
    N = len(betas)
    ln2 = math.log(2)

    free_energies = [-lz / (beta * L) for beta, lz in zip(betas, log_z)]

    # Internal energy: u[n] = −∂ ln Z/∂β ≈ −Δ(ln Z) / Δ(ln β) / β_n
    # Δ(ln β) = ln(β_{n+1}/β_n) = ln 2 (constant on XTRG grid).
    # Use forward differences except at the last point (use backward there).
    energies = []
    for n in range(N):
        if n < N - 1:
            u = -(log_z[n + 1] - log_z[n]) / (ln2 * betas[n])
        else:
            # Backward difference for the last point.
            u = -(log_z[n] - log_z[n - 1]) / (ln2 * betas[n - 1])
        energies.append(u / L)

    # Specific heat: c_V = ∂u/∂T = −β² ∂u/∂β = −β ∂u/∂(ln β).
    # On the XTRG grid Δ(ln β) = ln 2, so c_V[n] ≈ −β_n Δu / ln 2.
    specific_heats = []
    for n in range(N):
        if n < N - 1:
            cv = -betas[n] * (energies[n + 1] - energies[n]) / ln2
        else:
            cv = -betas[n] * (energies[n] - energies[n - 1]) / ln2
        specific_heats.append(cv)

    # Entropy: S = β (u − f)  [per site]
    entropies = [
        beta * (u - f)
        for beta, u, f in zip(betas, energies, free_energies)
    ]

    return free_energies, energies, specific_heats, entropies


# ---------------------------------------------------------------------------
# Trace sign guard
# ---------------------------------------------------------------------------

def _ensure_positive_trace(sign: float, beta: float) -> None:
    """Raise if `Tr[ρ(β)]`'s sign is not strictly positive.

    A physical thermal density matrix ρ = e^{-βH} always has a strictly
    positive trace (it is a sum of positive Boltzmann weights). This is a
    defensive check, not expected control flow: a non-positive sign here
    means the compressed density matrix has drifted into an unphysical
    regime (e.g. from accumulated truncation error), which should be
    surfaced immediately rather than silently propagated into a
    nonsensical `log_z` entry.

    Parameters
    ----------
    sign:
        Sign returned by `NormalMPO.log_trace()`, one of `+1.0`, `-1.0`,
        or `0.0`.
    beta:
        Inverse temperature at which the trace was computed (for the error
        message).

    Raises
    ------
    RuntimeError
        If `sign` is not `+1.0`.
    """
    if sign != 1.0:
        raise RuntimeError(
            f"Tr[ρ(β={beta:.6g})] is not positive (sign={sign:+.0f}); "
            "the density matrix has become numerically unphysical"
        )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(
    H: MPO,
    spc: Index,
    opts: Optional[Options] = None,
) -> Tuple[Summary, Artifact]:
    """Run XTRG to compute finite-temperature properties of a Hamiltonian MPO.

    Initializes the thermal density matrix via a Taylor expansion
    ρ(τ₀) ≈ Σ_n (-τ₀)^n/n! H^n, then repeatedly squares it using
    variational MPO-MPO compression to reach β_max = 2^n_steps × τ₀.

    Thermodynamic observables (f, u, c_V, S) are computed from log Z at each
    step using log-β finite differences for uniform accuracy across the
    exponential temperature grid.

    Parameters
    ----------
    H:
        Hamiltonian MPO. Should be in standard MPO form; not modified.
    spc:
        Physical space index, used to build the identity MPO (H⁰ = I)
        inside `thermal_mpo`.
    opts:
        XTRG run options. Defaults to `Options()` if `None`.

    Returns
    -------
    Summary
        Thermodynamic history (β grid, log Z, derived observables). Written
        to `thermal.ckpt` under the checkpoint directory.
    Artifact
        Final density matrix `ρ(β_max)` with its `beta` and `step`.

    Raises
    ------
    ValueError
        If `opts.scheme` is not a recognized scheme.
    NotImplementedError
        If `opts.scheme` is recognized but not yet implemented.
    """
    if opts is None:
        opts = Options()

    if opts.scheme not in _IMPLEMENTED_SCHEMES:
        raise NotImplementedError(
            f"XTRG scheme {opts.scheme!r} is recognized but not yet implemented; "
            f"implemented schemes are: {', '.join(sorted(_IMPLEMENTED_SCHEMES))}"
        )

    L = H.L
    # Resolve the optional disk-cache directory. A unique subdirectory
    # (first 8 hex digits of a UUID4) is created inside the user-supplied
    # path so that concurrent runs sharing the same config do not collide.
    # The same subdirectory is reused by every squaring step of this run.
    _cache: Optional[Path] = None
    if opts.env_cache_dir:
        _cache = Path(opts.env_cache_dir) / uuid.uuid4().hex[:8]
        _cache.mkdir(parents=True, exist_ok=True)

    # Resolve the checkpoint directory; default to Path.cwd() when unset,
    # mirroring the convention used by configure_logging / DMRG.
    _ckpt: Path = Path(opts.checkpoint_dir) if opts.checkpoint_dir is not None else Path.cwd()
    _ckpt.mkdir(parents=True, exist_ok=True)
    _artifacts: Path = _ckpt / 'artifacts'
    if opts.save_artifacts:
        _artifacts.mkdir(parents=True, exist_ok=True)

    max_bond_str = str(opts.max_bond) if opts.max_bond is not None else 'unlimited'
    logger.info("─" * 60)
    logger.info("Commencing: XTRG Algorithm".center(60))
    logger.info("─" * 60)
    logger.info("")
    logger.info("  scheme            : %s", opts.scheme)
    logger.info("  chain length      : %d", L)
    logger.info("  initial tau_0     : %.6g", opts.tau_0)
    logger.info("  cooling steps     : %d", opts.n_steps)
    logger.info("  beta_max          : %.6g", opts.tau_0 * 2 ** opts.n_steps)
    logger.info("  max bond dim      : %s", max_bond_str)
    logger.info("  trunc thresh      : %.2e", opts.trunc_thresh)
    logger.info("  sweeps / step     : %d", opts.n_sweeps)
    logger.info("  fit conv thresh   : %.2e", opts.z_tol)
    logger.info("  taylor order      : %d", opts.taylor_order)
    if opts.scheme == '1sp':
        alpha_str = str(opts.expand_alpha) if opts.expand_alpha is not None else 'no compression'
        logger.info("  expand k          : %d", opts.expand_k)
        logger.info("  expand alpha      : %s", alpha_str)
    if opts.checkpoint_dir is not None:
        logger.info("  checkpoint dir    : %s", _ckpt)
    if _cache is not None:
        logger.info("  env cache dir     : %s", _cache)
    if opts.save_artifacts:
        logger.info("  save artifacts    : True (since step %d)", opts.save_artifacts_since)
    logger.info("")

    # Step 0: initialize ρ(τ₀) via Taylor expansion.
    logger.info("Initializing ρ(τ₀=%.6g) via Taylor expansion (order %d)…",
                opts.tau_0, opts.taylor_order)
    rho = thermal_mpo(H, opts.tau_0, opts.taylor_order, spc)

    betas: list[float] = [opts.tau_0]
    # log_trace() (rather than log(rho.trace())) keeps log_z finite even
    # when Z(beta) itself would overflow float64 deep into the cooling run.
    log_abs_z, sign_z = rho.log_trace()
    _ensure_positive_trace(sign_z, betas[-1])
    log_z: list[float] = [log_abs_z]
    discarded_weights: list[float] = []

    logger.info("  β = %.6g,  log Z = %+.8g", betas[-1], log_z[-1])

    artifact = Artifact(rho=rho, beta=betas[-1], step=0)
    _save_progress(artifact, _ckpt)
    _archive_artifact(artifact, opts, _artifacts)

    w = len(str(opts.n_steps))
    try:
        for step in range(opts.n_steps):
            logger.info("step %*d / %d: squaring ρ(β=%.6g) → ρ(β=%.6g)",
                        w, step + 1, opts.n_steps, betas[-1], betas[-1] * 2)

            rho, dw = _fit_mpo(rho, rho, opts, _cache)
            discarded_weights.append(dw)
            betas.append(betas[-1] * 2)
            lz, sign_z = rho.log_trace()
            _ensure_positive_trace(sign_z, betas[-1])
            log_z.append(lz)

            logger.info("  β = %.6g,  log Z = %+.8g,  dw = %.4e",
                        betas[-1], lz, dw)

            k = step + 1
            mid_summary = _build_summary(
                betas, log_z, discarded_weights, L, step=k, finished=False,
            )
            _save_thermal(mid_summary, _ckpt)

            artifact = Artifact(rho=rho, beta=betas[-1], step=k)
            _save_progress(artifact, _ckpt)
            _archive_artifact(artifact, opts, _artifacts)
    finally:
        # Remove the run-specific cache subdirectory; ignore errors so that a
        # partially-written or already-deleted directory does not mask the real
        # exception (if any) from the cooling loop.
        if _cache is not None:
            shutil.rmtree(_cache, ignore_errors=True)

    logger.info("")

    summary = _build_summary(
        betas, log_z, discarded_weights, L, step=opts.n_steps, finished=True,
    )
    _save_thermal(summary, _ckpt)

    # Success path only: drop mid-run progress so a finished job does not
    # leave a stale progress.ckpt behind. A crash earlier leaves it on disk.
    (_ckpt / 'progress.ckpt').unlink(missing_ok=True)
    (_ckpt / 'progress_lock.ckpt').unlink(missing_ok=True)

    return summary, artifact
