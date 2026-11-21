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


"""Tangent-space Tensor Renormalization Group (TgTRG) driver.

TgTRG cools the thermal density matrix by integrating the exact flow

    dρ/dβ = -H ρ,    ρ(0) = 1

projected onto the tangent space of the MPO manifold — imaginary-time TDVP
on the density matrix read as a super-MPS over the doubled physical space
(arXiv:2212.11973, PRL 130, 226502). Each step is one sweep of local
propagators `exp(-tau H_eff)`, so the cost per step is `O(D³)`, the same as
a DMRG sweep and a factor of `D` below XTRG's squaring.

`run()` operates purely on state: it takes a starting density matrix
(bundled with its `beta` and `step` in an `Artifact`) and cools it along the
grid the options describe. Building ρ(τ₀) is the caller's responsibility,
via a Taylor expansion of `e^{-τ₀ H}` (`thermal_mpo()`, accurate for
sufficiently small τ₀):

    import alice.algorithm.tgtrg as tgtrg
    from alice.network.thermal import thermal_mpo

    opts = tgtrg.Options(tau_0=2 ** -12, n_steps=20)
    rho0 = thermal_mpo(H, opts.tau_0, opts.taylor_order, spc)
    summary, artifact = tgtrg.run(
        tgtrg.Artifact(rho=rho0, beta=opts.tau_0, step=0), H, opts)

Unlike XTRG, the Hamiltonian is needed at every step, so `run()` takes it
explicitly.

Resuming an interrupted run is calling `run()` again with the `Artifact`
written to `tgtrg.ckpt`; the matching β/log Z history is recovered
automatically from `thermal.ckpt` in `opts.checkpoint_dir`:

    resumed = tgtrg.Artifact.load(ckpt_dir / 'tgtrg.ckpt')
    summary, artifact = tgtrg.run(resumed, H, opts)

Two cooling schedules
---------------------
`Options.grid` selects one, never both:

- `'exp'`: `β_k = 2^k τ₀`, each step evolving by `τ = β_k`. This is the
  stage the paper starts with, and it reaches a given β in logarithmically
  many steps.
- `'lin'`: a constant step `τ = Options.tau` from wherever the handed-in
  `Artifact` sits.

Stacking them is a continuation, not a third mode: cool with `'exp'` to the
crossover, then call `run()` again with the final `Artifact` and `'lin'`
options. `n_steps` is the absolute stop index in both modes, so the step
counter and the β/log Z history run continuously across the phases, and
`Summary.taus` records the increment each entry was reached with.

How large a step is too large
-----------------------------
Stacking is not a matter of taste. The projector-splitting integrator is
exact at full rank for *any* step size in exact arithmetic, and on a small
chain it demonstrably is: a single step of `tau = 8` at `beta = 32`
reproduces the analytic log Z to 1e-14. But the backward update runs the
flow the other way, `exp(+tau H⁽⁰⁾)`, and so amplifies the high-energy end
of the local spectrum by `exp(tau ΔE)`. Once that factor passes the inverse
of the float64 resolution, the directions `ρ(β)` has legitimately suppressed
are indistinguishable from roundoff, and amplifying them pollutes the state.
The same chain, stepped once with `tau = 32` instead of four times with
`tau = 8`, loses eight digits of log Z — and reports a Krylov error of
1e-13 while doing it, because nothing is wrong with the local expansions.

Since `order=2` propagates in half-steps, the exponent of one local
propagation is `(tau / 2) ΔE`, and measurements across three Hamiltonian
scales put the breakdown where that exponent passes `log(1 / eps) ≈ 36`:
safe at 18 and 36, ruined at 72, whether 72 is reached by a large `tau` or
by a wide spectrum. The limit is therefore on the exponent, not on `tau`
and not on β — a longer chain or a stronger coupling reaches it at a
proportionally smaller step.

The exponential grid walks into it on its own: its step size *is* β. The
grid is good for one thing, bringing a run from the Taylor state to `β` of
order one in logarithmically many steps; past that, hand over to a linear
phase. The default `n_steps` stops where that handover
belongs, and `run()` warns when a step exceeds the soft limit below.
`docs/algorithms/tgtrg/lessons.md` carries the measurements.
"""

from __future__ import annotations

import logging
import math
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from nicole import deserialize as _deserialize_tensor

from alice.network import MPO, observe
from alice.network.thermal import NormalMPO

from ..interface import AlgorithmOptions, AlgorithmSummary
from .environ import Environment, build_right_envs, left_env_boundary, \
    right_env_boundary
from .sweep import cool_step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheme and grid alias resolution
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

# Soft limit on a single step. The criterion the measurements support is on
# the exponent of one local propagation, `(tau / 2) ΔE < log(1 / eps) ≈ 36`
# (see the module docstring), which this constant expresses for a spectral
# range of order a few — the free-fermion and Hubbard chains the tests and
# examples run. It is therefore conservative there and *silent* where ΔE is
# larger, on a long chain or at strong coupling, since the driver is not
# given the spectral range. Hence a warning and not a refusal.
_TAU_SOFT_LIMIT = 8.0

_GRID_ALIASES: dict[str, str] = {
    'exp': 'exp',
    'exponential': 'exp',
    'doubling': 'exp',
    'lin': 'lin',
    'linear': 'lin',
    'uniform': 'lin',
}


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
            f"unknown TgTRG scheme {alias!r}; recognized values are: {known}"
        )

    return canonical


def _resolve_grid(alias: str) -> str:
    """Normalize a β-grid alias to its canonical name.

    Parameters
    ----------
    alias:
        User-provided grid string.

    Returns
    -------
    str
        Canonical grid name (`'exp'` or `'lin'`).

    Raises
    ------
    ValueError
        If `alias` is not a recognized grid name.
    """
    canonical = _GRID_ALIASES.get(alias.lower())
    if canonical is None:
        known = ', '.join(sorted(_GRID_ALIASES))
        raise ValueError(
            f"unknown TgTRG β grid {alias!r}; recognized values are: {known}"
        )

    return canonical


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class Options(AlgorithmOptions):
    """TgTRG run options.

    All fields have sensible defaults so `Options()` is a valid minimal
    configuration. Use `Options.from_toml` to load from a TOML section, or
    `Options.load_toml` to read directly from a file.

    Parameters
    ----------
    scheme:
        Update scheme per cooling step. Canonical values and aliases:

        - `'1s'` / `'1-site'` / `'one-site'`: fixed-rank 1-site update with
          the zero-site back-evolution on each bond.
        - `'2s'` / `'2-site'` / `'two-site'`: 2-site update with a
          truncating SVD split, so the bond dimension adapts as the run
          cools. The default, and what a cooling run almost always wants:
          `ρ(τ₀)` starts at the Hamiltonian's own bond dimension, far below
          what a cold density matrix needs.
    grid:
        β-grid mode, `'exp'` / `'exponential'` or `'lin'` / `'linear'`. One
        per run, never both; stacking the two is a continuation (see the
        module docstring).
    tau_0:
        Inverse temperature of the initial Taylor state, ≈ 2⁻¹². Small
        enough that the Taylor expansion converges. On the exponential grid
        it also fixes the whole grid, `β_k = 2^k τ₀`.
    tau:
        Step size for `grid='lin'`, ignored on the exponential grid, where
        the step is `β_k` itself.
    n_steps:
        Absolute step index to stop at, counted from the Taylor state — not
        a number of additional steps to perform. On the exponential grid the
        final inverse temperature is `β_max = 2^n_steps τ₀`, and the default
        stops at `β = 2`, where the handover to a linear phase belongs (see
        the module docstring on step size).
    taylor_order:
        Truncation order of the Taylor series for `ρ(τ₀) = e^{-τ₀ H}`. Used
        only by the caller that builds the initial state; recorded here so a
        run is reproducible from its options alone.
    order:
        Integrator order: `1` for a single half-sweep per step
        (Lie-Trotter), `2` for the symmetric forward-and-backward pair
        (Lubich). Defaults to `2`, which puts second-order accuracy at every
        recorded β — and for a cooling run the trajectory is the output.
    max_bond:
        Maximum bond dimension of `ρ`. `None` means unlimited. The cap is
        applied to the handed-in state before the first step as well as at
        every 2-site split, so it holds over the whole run. For `'1s'`,
        whose tangent space has fixed rank by construction, that initial
        compaction is the *only* way the cap takes effect, and it sets the
        rank the run then keeps.
    trunc_thresh:
        Singular-value truncation threshold: a value is discarded when it
        falls below `trunc_thresh × ‖Θ‖`, and the cooling sweeps hold `‖Θ‖`
        at `1`. Being relative, the same setting carries over between chain
        lengths and temperatures. `0.0` disables threshold truncation,
        leaving `max_bond` in charge.
    krylov_dim:
        Maximum Krylov subspace dimension per local propagator.
    krylov_tol:
        Relative convergence threshold for the Krylov expansion, on the
        weight the propagator puts on the newest direction.
    obs_exact:
        If `True`, measure `u = Tr[ρH] / (Tr[ρ] L)` directly through
        `observe` after every step, recorded as `Summary.exact_energies`.
        Off by default so a cooling run pays for nothing but the cooling;
        on, it is the natural cross-check against the finite-difference `u`,
        which carries the discretization error of the grid.
    env_cache_dir:
        Root directory for environment disk caching. When set, `run()`
        creates a unique subdirectory inside it (first 8 hex characters of a
        UUID4) so that concurrent runs sharing the same config do not
        overwrite each other's blocks. The subdirectory is removed when
        `run()` returns (or raises). `None` (default) keeps all blocks in
        memory. Stored as `str` for TOML compatibility.
    env_async_io:
        If `True` (default), disk writes are submitted asynchronously so
        they overlap with computation. No effect when `env_cache_dir` is
        `None`.
    env_window:
        Number of environment blocks to keep in memory at once when disk
        caching is enabled.
    checkpoint_dir:
        Directory for checkpoint files. After every cooling step a
        `thermal.ckpt` file (loadable via `tgtrg.Summary.load`) is written
        using an atomic write (`thermal_lock.ckpt` → rename). Mid-run
        progress is stored as `tgtrg.ckpt` (an `Artifact`) and removed when
        `run()` finishes successfully. That mid-run name carries the
        algorithm, so a TgTRG and an XTRG run can share a directory.
        `None` (default) resolves to `Path.cwd()` at the time `run()` is
        called. Stored as `str` for TOML compatibility.
    artifacts_dir:
        Directory for per-step `Artifact` files (`step_XX.ckpt`), written
        when `save_artifacts` is `True`. `None` (default) resolves to
        `artifacts/` under `checkpoint_dir`. Stored as `str` for TOML
        compatibility.
    save_artifacts:
        If `True` (default), write per-step `Artifact` files under
        `artifacts_dir`.
    save_artifacts_every:
        Stride on the step index for archiving artifacts, independent of the
        cooling grid. The final step of a run is always archived whatever the
        stride, so a finished run leaves a resumable endpoint for the next
        phase of a stacked schedule to pick up.
    """

    scheme: str = '2s'
    grid: str = 'exp'
    tau_0: float = 2 ** -12
    tau: float = 0.1
    n_steps: int = 13
    taylor_order: int = 10
    order: int = 2
    max_bond: Optional[int] = None
    trunc_thresh: float = 1e-12
    krylov_dim: int = 16
    krylov_tol: float = 1e-12
    obs_exact: bool = False
    env_cache_dir: Optional[str] = None
    env_async_io: bool = True
    env_window: int = 2
    checkpoint_dir: Optional[str] = None
    artifacts_dir: Optional[str] = None
    save_artifacts: bool = True
    save_artifacts_every: int = 1

    def __post_init__(self) -> None:
        self.scheme = _resolve_scheme(self.scheme)
        self.grid = _resolve_grid(self.grid)
        if self.order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {self.order}")
        if self.tau_0 <= 0.0:
            raise ValueError(f"tau_0 must be positive, got {self.tau_0}")
        if self.grid == 'lin' and self.tau <= 0.0:
            raise ValueError(
                f"tau must be positive on a linear grid, got {self.tau}"
            )
        if self.n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {self.n_steps}")
        if self.taylor_order < 1:
            raise ValueError(f"taylor_order must be >= 1, got {self.taylor_order}")
        if self.krylov_dim < 1:
            raise ValueError(f"krylov_dim must be >= 1, got {self.krylov_dim}")
        if self.max_bond is not None and self.max_bond < 1:
            raise ValueError(f"max_bond must be >= 1 or None, got {self.max_bond}")
        if self.save_artifacts_every < 1:
            raise ValueError(
                f"save_artifacts_every must be >= 1, got {self.save_artifacts_every}"
            )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class Summary(AlgorithmSummary):
    """TgTRG thermodynamic summary (no density matrix).

    Density matrices are returned separately as `Artifact` (and optionally
    archived under `artifacts/` when `Options.save_artifacts` is enabled).

    Parameters
    ----------
    betas:
        Inverse temperature at each cooling step, starting from the Taylor
        state's own β.
    taus:
        Increment each entry was reached with, aligned with `betas`. The
        first entry is `0.0`: the Taylor state was not reached by a step.
        Recorded because the finite differences need the spacing, and
        because it tells the phases of a stacked schedule apart after the
        fact.
    log_z:
        `log Z(β)` at each step, from `NormalMPO.log_trace()` (never
        `log(trace())`), so it stays finite when `Z(β)` itself is far
        outside float64 range.
    free_energies:
        Free energy per site, `f(β) = -log Z(β) / (β L)`.
    energies:
        Internal energy per site, `u(β)`, by finite differences of `log Z`
        on the β grid.
    specific_heats:
        Specific heat per site, `c_V(β) = -β² du/dβ`, by finite differences
        of `energies`.
    entropies:
        Entropy per site, `S(β) = β (u(β) - f(β))`.
    exact_energies:
        Internal energy per site measured directly as
        `Tr[ρH] / (Tr[ρ] L)`, recorded only when `Options.obs_exact` is
        enabled, and empty otherwise. Free of the grid's discretization
        error, so comparing it against `energies` measures that error.
    discarded_weights:
        Per-step discarded weight at the mid-chain bond (`'2s'`; `0.0`
        entries for `'1s'`, which never truncates). One entry per step, so
        one fewer than `betas`.
    krylov_errors:
        Per-step largest Krylov error estimate over all local propagators.
        One entry per step.
    max_bonds:
        Largest bond dimension of `ρ` at each step, aligned with `betas`.
    finished:
        `True` only for the summary returned by a completed `run()` call.
        `False` for mid-run `thermal.ckpt` snapshots.
    n_steps:
        Number of cooling steps reflected in this summary.
    """

    betas: List[float] = field(default_factory=list)
    taus: List[float] = field(default_factory=list)
    log_z: List[float] = field(default_factory=list)
    free_energies: List[float] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    specific_heats: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    exact_energies: List[float] = field(default_factory=list)
    discarded_weights: List[float] = field(default_factory=list)
    krylov_errors: List[float] = field(default_factory=list)
    max_bonds: List[int] = field(default_factory=list)
    finished: bool = True
    n_steps: int = 0

    def serialize(self) -> dict:
        """Serialize the summary to a plain dict compatible with `torch.save`.

        Returns
        -------
        dict
            Serialized summary (version 1; no density matrix).
        """
        return {
            'version': 1,
            'betas': self.betas,
            'taus': self.taus,
            'log_z': self.log_z,
            'free_energies': self.free_energies,
            'energies': self.energies,
            'specific_heats': self.specific_heats,
            'entropies': self.entropies,
            'exact_energies': self.exact_energies,
            'discarded_weights': self.discarded_weights,
            'krylov_errors': self.krylov_errors,
            'max_bonds': self.max_bonds,
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
            Unused; the summary holds no tensors. Accepted for API
            compatibility with `AlgorithmSummary.load`.

        Returns
        -------
        Summary
            Reconstructed thermodynamic summary.

        Raises
        ------
        ValueError
            If `data['version']` is not `1`.
        """
        del device  # no tensors in a TgTRG summary
        version = data.get('version', 1)
        if version != 1:
            raise ValueError(f"Unsupported Summary serialization version: {version!r}")

        return cls(
            betas=list(data['betas']),
            taus=list(data['taus']),
            log_z=list(data['log_z']),
            free_energies=list(data['free_energies']),
            energies=list(data['energies']),
            specific_heats=list(data['specific_heats']),
            entropies=list(data['entropies']),
            exact_energies=list(data.get('exact_energies', [])),
            discarded_weights=list(data.get('discarded_weights', [])),
            krylov_errors=list(data.get('krylov_errors', [])),
            max_bonds=[int(b) for b in data.get('max_bonds', [])],
            finished=bool(data.get('finished', True)),
            n_steps=int(data.get('n_steps', 0)),
        )


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

@dataclass
class Artifact(AlgorithmSummary):
    """Density-matrix snapshot at one TgTRG cooling step.

    Parameters
    ----------
    rho:
        Thermal density matrix `ρ(β)` as a `NormalMPO`.
    beta:
        Inverse temperature of this snapshot.
    step:
        Step index: `0` for the Taylor state, `k` after the `k`-th cooling
        step.
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
            Reconstructed artifact.

        Raises
        ------
        ValueError
            If `data['version']` is not `1`.
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

        return cls(rho=rho, beta=float(data['beta']), step=int(data['step']))


# ---------------------------------------------------------------------------
# The β grid
# ---------------------------------------------------------------------------

def _build_grid(state: Artifact, opts: Options) -> List[Tuple[float, float]]:
    """Build the `(beta, tau)` pairs for the steps this call will perform.

    Parameters
    ----------
    state:
        Starting artifact, whose `beta` and `step` anchor the grid.
    opts:
        Run options supplying the grid mode and its parameters.

    Returns
    -------
    list of tuple
        One `(beta_after, tau)` pair per step from `state.step` to
        `opts.n_steps`, in order. Empty when the state is already at the
        target.

    Raises
    ------
    ValueError
        If `opts.grid == 'exp'` and `state.beta` is not `2^state.step τ₀`,
        which would make the doubling grid inconsistent with the state it
        starts from.
    """
    if opts.grid == 'exp':
        expected = opts.tau_0 * 2 ** state.step
        if not math.isclose(state.beta, expected, rel_tol=1e-9):
            raise ValueError(
                f"on the exponential grid the state at step {state.step} must "
                f"sit at β = 2^{state.step} τ₀ = {expected:.6g}, but the given "
                f"state is at β = {state.beta:.6g}; either τ₀ is wrong or the "
                "state belongs to a linear phase (use grid='lin' to continue it)"
            )

    grid: List[Tuple[float, float]] = []
    beta = state.beta
    for _ in range(state.step, opts.n_steps):
        # On the exponential grid the step size is β itself, turning one
        # sweep into a doubling of the inverse temperature.
        tau = beta if opts.grid == 'exp' else opts.tau
        beta = beta + tau
        grid.append((beta, tau))

    return grid


# ---------------------------------------------------------------------------
# Observable extraction
# ---------------------------------------------------------------------------

def _derivative(xs: List[float], ys: List[float]) -> List[float]:
    """Differentiate `ys` with respect to `xs` on a possibly non-uniform grid.

    Interior points use the three-point formula for unequal spacing, which
    is second order in the local spacing; the two endpoints fall back to the
    two-point one-sided difference, first order. A single point carries no
    derivative information at all and yields `0.0`.

    The general form is the one thing the two grid modes share. On the
    exponential grid the spacing grows geometrically, so a fixed-spacing
    formula would be wrong there, while XTRG's uniform `log β` shortcut would
    be wrong on the linear grid; this is correct on both, and on anything a
    stacked schedule produces across its phase boundary.

    Parameters
    ----------
    xs:
        Strictly increasing abscissae.
    ys:
        Values at `xs`, the same length.

    Returns
    -------
    list[float]
        `dy/dx` at each point of `xs`.
    """
    n = len(xs)
    if n < 2:
        return [0.0] * n

    out: List[float] = []
    for i in range(n):
        if i == 0:
            out.append((ys[1] - ys[0]) / (xs[1] - xs[0]))
        elif i == n - 1:
            out.append((ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]))
        else:
            h1 = xs[i] - xs[i - 1]
            h2 = xs[i + 1] - xs[i]
            out.append(
                -h2 / (h1 * (h1 + h2)) * ys[i - 1]
                + (h2 - h1) / (h1 * h2) * ys[i]
                + h1 / (h2 * (h1 + h2)) * ys[i + 1]
            )

    return out


def _compute_observables(
    betas: List[float],
    log_z: List[float],
    L: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Derive thermodynamic observables from the log Z grid.

    Parameters
    ----------
    betas:
        Inverse temperatures, strictly increasing.
    log_z:
        `log Z(β)` at each point, the same length as `betas`.
    L:
        Chain length, for per-site normalization.

    Returns
    -------
    list[float]
        Free energies `f(β)` per site.
    list[float]
        Internal energies `u(β)` per site.
    list[float]
        Specific heats `c_V(β)` per site.
    list[float]
        Entropies `S(β)` per site.
    """
    free_energies = [-lz / (beta * L) for beta, lz in zip(betas, log_z)]

    # u = -∂ log Z/∂β, per site.
    d_log_z = _derivative(betas, log_z)
    energies = [-d / L for d in d_log_z]

    # c_V = ∂u/∂T = -β² ∂u/∂β, per site (u already is).
    du = _derivative(betas, energies)
    specific_heats = [-beta * beta * d for beta, d in zip(betas, du)]

    entropies = [beta * (u - f) for beta, u, f in zip(betas, energies, free_energies)]

    return free_energies, energies, specific_heats, entropies


# ---------------------------------------------------------------------------
# Trace sign guard
# ---------------------------------------------------------------------------

def _ensure_positive_trace(sign: float, beta: float) -> None:
    """Raise if `Tr[ρ(β)]`'s sign is not strictly positive.

    A physical thermal density matrix has a strictly positive trace, being a
    sum of positive Boltzmann weights. This is a defensive check, not
    expected control flow: a non-positive sign means the cooled density
    matrix has drifted into an unphysical regime, which should surface
    immediately instead of propagating into a nonsensical `log_z` entry.

    Parameters
    ----------
    sign:
        Sign returned by `NormalMPO.log_trace()`.
    beta:
        Inverse temperature at which the trace was computed, for the message.

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
# Checkpoint / artifact helpers
# ---------------------------------------------------------------------------

def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Atomically write `payload` via a sibling `*_lock` file then rename."""
    import torch

    lock_path = path.with_name(path.stem + '_lock' + path.suffix)
    torch.save(payload, lock_path)
    # Atomic rename: guaranteed on POSIX, best-effort on Windows, where it
    # still avoids leaving a half-written target on disk.
    lock_path.replace(path)


class _History:
    """The per-step records a run accumulates, kept together for brevity.

    `run` threads these six lists through every step, checkpoint write and
    summary build; bundling them keeps the signatures of those helpers from
    turning into parameter lists nobody can read.
    """

    def __init__(self) -> None:
        self.betas: List[float] = []
        self.taus: List[float] = []
        self.log_z: List[float] = []
        self.exact_energies: List[float] = []
        self.discarded_weights: List[float] = []
        self.krylov_errors: List[float] = []
        self.max_bonds: List[int] = []

    def summary(self, L: int, step: int, finished: bool) -> Summary:
        """Build a density-matrix-free `Summary` from the history so far."""
        free_energies, energies, specific_heats, entropies = _compute_observables(
            self.betas, self.log_z, L,
        )

        return Summary(
            betas=list(self.betas),
            taus=list(self.taus),
            log_z=list(self.log_z),
            free_energies=free_energies,
            energies=energies,
            specific_heats=specific_heats,
            entropies=entropies,
            exact_energies=list(self.exact_energies),
            discarded_weights=list(self.discarded_weights),
            krylov_errors=list(self.krylov_errors),
            max_bonds=list(self.max_bonds),
            finished=finished,
            n_steps=step,
        )


def _save_thermal(summary: Summary, ckpt_dir: Path) -> None:
    """Write `thermal.ckpt` atomically in `ckpt_dir`."""
    _atomic_torch_save(summary.serialize(), ckpt_dir / 'thermal.ckpt')


def _save_checkpoint(artifact: Artifact, ckpt_dir: Path) -> None:
    """Write `tgtrg.ckpt` atomically in `ckpt_dir`."""
    _atomic_torch_save(artifact.serialize(), ckpt_dir / 'tgtrg.ckpt')


def _archive_artifact(
    artifact: Artifact,
    opts: Options,
    artifacts_dir: Path,
    final: bool = False,
) -> None:
    """Archive `artifact` under `artifacts_dir` when the options ask for it.

    The stride keys on the absolute step index, so a resumed run archives
    the same grid as an uninterrupted one. The final step is archived
    whatever the stride, leaving an endpoint the next phase of a stacked
    schedule can resume from.
    """
    if not opts.save_artifacts:
        return
    if final or artifact.step % opts.save_artifacts_every == 0:
        _atomic_torch_save(
            artifact.serialize(), artifacts_dir / f'step_{artifact.step:05d}.ckpt'
        )


def _resume_history(state: Artifact, ckpt_dir: Path, opts: Options) -> _History:
    """Recover the per-step history up to `state.step` from `thermal.ckpt`.

    A history reaching past `state.step` is truncated, not rejected, so a
    run can be continued from any archived `artifacts/step_XX.ckpt` — not
    only from the step the previous run stopped at. The steps beyond are
    then recomputed and overwrite the truncated ones on disk.

    The consistency check is β-at-step only. A change of spacing between the
    last entries is a phase boundary of a stacked schedule, not an error.

    Parameters
    ----------
    state:
        Starting density matrix with its `beta` and `step`.
    ckpt_dir:
        Resolved checkpoint directory holding `thermal.ckpt`.
    opts:
        Run options, for `obs_exact`.

    Returns
    -------
    _History
        History covering steps `0 … state.step`.

    Raises
    ------
    FileNotFoundError
        If no `thermal.ckpt` exists in `ckpt_dir`.
    ValueError
        If the history stops before `state.step`, or if its β at
        `state.step` does not match `state.beta`.
    """
    path = ckpt_dir / 'thermal.ckpt'
    if not path.exists():
        raise FileNotFoundError(
            f"resuming from step {state.step} requires an existing thermal.ckpt "
            f"in {ckpt_dir} to recover the β/log Z history, but none was found"
        )
    past = Summary.load(path)

    if len(past.betas) <= state.step:
        raise ValueError(
            f"thermal.ckpt at {ckpt_dir} only reaches step {len(past.betas) - 1}, "
            f"but the given state is at step {state.step}; the two are inconsistent"
        )
    if not math.isclose(past.betas[state.step], state.beta, rel_tol=1e-9):
        raise ValueError(
            f"thermal.ckpt's β at step {state.step} "
            f"({past.betas[state.step]:.6g}) in {ckpt_dir} does not match the "
            f"given state's β ({state.beta:.6g}); the two are inconsistent"
        )

    last = len(past.betas) - 1
    if last > state.step:
        logger.warning(
            "thermal.ckpt reaches step %d; continuing from step %d — the %d "
            "later entries will be recomputed and overwritten",
            last, state.step, last - state.step,
        )

    history = _History()
    k = state.step
    history.betas = list(past.betas[:k + 1])
    history.taus = list(past.taus[:k + 1])
    history.log_z = list(past.log_z[:k + 1])
    history.max_bonds = [int(b) for b in past.max_bonds[:k + 1]]
    # One diagnostic per step, so `k` entries cover the steps that produced
    # β_1 … β_k.
    history.discarded_weights = list(past.discarded_weights[:k])
    history.krylov_errors = list(past.krylov_errors[:k])

    if opts.obs_exact:
        recovered = list(past.exact_energies[:k + 1])
        if len(recovered) != k + 1:
            # The earlier phase ran without `obs_exact`, so there is nothing
            # to recover for those steps. Pad, not refuse: the column is a
            # diagnostic, and the run that wants it should still get it from
            # here on.
            logger.warning(
                "thermal.ckpt holds %d exact energies for %d steps; the earlier "
                "entries were not measured and are recorded as NaN",
                len(recovered), k + 1,
            )
            recovered = [math.nan] * (k + 1 - len(recovered)) + recovered
        history.exact_energies = recovered

    return history


# ---------------------------------------------------------------------------
# run() helpers
# ---------------------------------------------------------------------------

def _warn_on_large_steps(grid: List[Tuple[float, float]]) -> None:
    """Warn once if the schedule contains a step past the soft limit.

    The failure a large step causes is silent — the local Krylov expansions
    converge perfectly well while the state they are applied to drifts — so
    it is worth saying out loud. It is a warning and not a refusal because
    the real criterion involves the spectral range, which the driver is not
    given: a model normalized differently may be fine well past the constant,
    and one with a wider range in trouble before it.
    """
    largest = max((tau for _, tau in grid), default=0.0)
    if largest > _TAU_SOFT_LIMIT:
        logger.warning(
            "the schedule contains a step of τ = %.6g, past the soft limit of "
            "%.6g: the backward bond update amplifies by exp(τ ΔE/2), and once "
            "that exceeds the float64 resolution it amplifies roundoff along "
            "with it. Consider handing over to grid='lin' with a smaller τ; "
            "the Krylov diagnostics will not flag this. The limit is on "
            "(τ/2) ΔE < 36, so a wide spectrum needs a smaller τ than this.",
            largest, _TAU_SOFT_LIMIT,
        )


def _build_envs(
    rho: NormalMPO,
    mpo: MPO,
    opts: Options,
    cache: Optional[Path],
) -> Tuple[Environment, Environment]:
    """Create and seed the two environment stores for a run.

    The right environments are built in full from the right-canonical `ρ`,
    and `env_left[0]` is the trivial boundary. Every later block is produced
    by the sweeps themselves, which leave both stores valid for the step
    after — so the environments are built once per run, not once per step.

    Parameters
    ----------
    rho:
        Density matrix with `center == 0`.
    mpo:
        Hamiltonian MPO.
    opts:
        Run options, for the scheme and the environment caching fields.
    cache:
        Run-specific cache directory, or `None` to keep blocks in memory.

    Returns
    -------
    Environment
        Left environment store, seeded at site `0`.
    Environment
        Right environment store, fully populated.
    """
    L = rho.L
    _2s = opts.scheme == '2s'
    if cache is not None:
        (cache / 'tgtrg_left').mkdir(parents=True, exist_ok=True)
        (cache / 'tgtrg_right').mkdir(parents=True, exist_ok=True)

    env_left = Environment(
        L,
        cache / 'tgtrg_left' if cache is not None else None,
        async_io=opts.env_async_io,
        window=opts.env_window,
        fetch_lo=0,
        fetch_hi=L - 2 if _2s else L - 1,
    )
    env_right = Environment(
        L,
        cache / 'tgtrg_right' if cache is not None else None,
        async_io=opts.env_async_io,
        window=opts.env_window,
        fetch_lo=1 if _2s else 0,
        fetch_hi=L - 1,
    )

    env_left[0] = left_env_boundary(rho, mpo)
    # The 2-site sweeps never fetch `env_right[0]`, but they do need the
    # boundary at `L-1`, which `build_right_envs` writes itself.
    build_right_envs(rho, mpo, env_right)
    if _2s and L > 1:
        # Keep the boundary available even when a window eviction dropped it;
        # it costs one dim-1 tensor and spares a reload.
        env_right[L - 1] = right_env_boundary(rho, mpo)

    return env_left, env_right


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(
    state: Artifact,
    mpo: MPO,
    opts: Optional[Options] = None,
) -> Tuple[Summary, Artifact]:
    """Cool a thermal density matrix along the β grid the options describe.

    Starts at `state.step` and continues to `opts.n_steps`, cooling by one
    tangent-space sweep per step. Building `ρ(τ₀)` (e.g. via `thermal_mpo`)
    and wrapping it in an `Artifact` at `step=0` is the caller's
    responsibility. Resuming an interrupted run, or continuing a finished
    one under new options — the second phase of a stacked exponential-then-
    linear schedule, say — is calling `run()` again with the `Artifact`
    loaded from `tgtrg.ckpt` or from any archived
    `artifacts/step_XX.ckpt`; the matching history is recovered from
    `thermal.ckpt` in `opts.checkpoint_dir`.

    Parameters
    ----------
    state:
        Starting density matrix, paired with its `beta` and `step`.
        `step=0` for a fresh run; `step > 0` to resume or continue. Cooled
        **in place**, so that one phase of a stacked schedule can hand its
        artifact to the next; an `Artifact` loaded from a checkpoint is
        therefore consumed by the first `run()` it is given to, and
        branching two continuations off one checkpoint needs the state
        reloaded between them.
    mpo:
        Hamiltonian MPO. Needed at every step, unlike XTRG, where the
        squaring never refers back to `H`.
    opts:
        Run options. Defaults to `Options()` if `None`. `n_steps` is the
        absolute step index to stop at, not a number of additional steps.

    Returns
    -------
    Summary
        Thermodynamic history (β grid, log Z, derived observables). Written
        to `thermal.ckpt` under the checkpoint directory.
    Artifact
        Final density matrix with its `beta` and `step`.

    Raises
    ------
    ValueError
        If `state.step` is past `opts.n_steps`, if the exponential grid
        disagrees with `state.beta`, or if a recovered `thermal.ckpt` is
        inconsistent with `state`.
    NotImplementedError
        If `opts.scheme` is recognized but not yet implemented.
    FileNotFoundError
        If `state.step > 0` and no `thermal.ckpt` exists in
        `opts.checkpoint_dir`.
    """
    if opts is None:
        opts = Options()

    if opts.scheme not in _IMPLEMENTED_SCHEMES:
        raise NotImplementedError(
            f"TgTRG scheme {opts.scheme!r} is recognized but not yet "
            f"implemented; implemented schemes are: "
            f"{', '.join(sorted(_IMPLEMENTED_SCHEMES))}"
        )
    # `n_steps` is an absolute index, so a target behind the state would
    # leave the written summary claiming fewer steps than its own history
    # carries. Equality is fine: it is the already-at-target no-op.
    if state.step > opts.n_steps:
        raise ValueError(
            f"the given state is at step {state.step}, past the target "
            f"opts.n_steps = {opts.n_steps}; note that n_steps counts cooling "
            "steps from the Taylor state, not additional steps to perform"
        )

    grid = _build_grid(state, opts)
    rho = state.rho
    L = rho.L

    # Resolve the optional disk-cache directory. A unique subdirectory
    # keeps concurrent runs sharing one config from colliding.
    _cache: Optional[Path] = None
    if opts.env_cache_dir:
        _cache = Path(opts.env_cache_dir) / uuid.uuid4().hex[:8]
        _cache.mkdir(parents=True, exist_ok=True)

    _ckpt: Path = (
        Path(opts.checkpoint_dir) if opts.checkpoint_dir is not None else Path.cwd()
    )
    _ckpt.mkdir(parents=True, exist_ok=True)
    _artifacts: Path = (
        Path(opts.artifacts_dir) if opts.artifacts_dir is not None
        else _ckpt / 'artifacts'
    )
    if opts.save_artifacts:
        _artifacts.mkdir(parents=True, exist_ok=True)

    max_bond_str = str(opts.max_bond) if opts.max_bond is not None else 'unlimited'
    logger.info("─" * 60)
    logger.info("Commencing: TgTRG Algorithm".center(60))
    logger.info("─" * 60)
    logger.info("")
    logger.info("  scheme            : %s", opts.scheme)
    logger.info("  integrator order  : %d", opts.order)
    logger.info("  chain length      : %d", L)
    logger.info("  beta grid         : %s", opts.grid)
    logger.info("  initial tau_0     : %.6g", opts.tau_0)
    if opts.grid == 'lin':
        logger.info("  linear tau        : %.6g", opts.tau)
    logger.info("  cooling steps     : %d", opts.n_steps)
    if state.step > 0:
        logger.info("  resuming at step  : %d  (%d step(s) remaining)",
                    state.step, opts.n_steps - state.step)
    # The grid already accounts for a resumed start, so its last entry is the
    # β this run ends at whether or not it begins at step 0.
    logger.info("  beta_max          : %.6g", grid[-1][0] if grid else state.beta)
    logger.info("  max bond dim      : %s", max_bond_str)
    logger.info("  trunc thresh      : %.2e  (relative)", opts.trunc_thresh)
    logger.info("  krylov dim / tol  : %d / %.2e", opts.krylov_dim, opts.krylov_tol)
    logger.info("  taylor order      : %d", opts.taylor_order)
    logger.info("  exact observables : %s", opts.obs_exact)
    if opts.checkpoint_dir is not None:
        logger.info("  checkpoint dir    : %s", _ckpt)
    if _cache is not None:
        logger.info("  env cache dir     : %s", _cache)
    if opts.save_artifacts:
        logger.info("  save artifacts    : True (every %d step(s))",
                    opts.save_artifacts_every)
        logger.info("  artifacts dir     : %s", _artifacts)
    logger.info("")

    _warn_on_large_steps(grid)

    # Bring the handed-in state under the bond limit before anything is
    # measured on it. The 2-site splits would get there on their own, but the
    # 1-site scheme cannot change a bond dimension at all, so for `'1s'` this
    # is where `max_bond` takes effect — and either way the first recorded
    # `max_bonds` entry should describe a state that respects the options.
    if opts.max_bond is not None and max(rho.bond_dims) > opts.max_bond:
        logger.info("  compacting initial ρ from D = %d to D = %d",
                    max(rho.bond_dims), opts.max_bond)
        rho.compact({'nkeep': opts.max_bond, 'thresh': opts.trunc_thresh})

    # Every sweep starts from the left end, and `build_right_envs` needs the
    # same gauge.
    if rho.center != 0:
        rho.canonical(0)

    if state.step == 0:
        history = _History()
        history.betas.append(state.beta)
        # The Taylor state was not reached by a step of this run.
        history.taus.append(0.0)
        lz, sign = rho.log_trace()
        _ensure_positive_trace(sign, state.beta)
        history.log_z.append(lz)
        history.max_bonds.append(max(rho.bond_dims))
        if opts.obs_exact:
            history.exact_energies.append(observe(rho, mpo) / L)
        logger.info("  β = %.6g,  log Z = %+.8g", state.beta, lz)
    else:
        history = _resume_history(state, _ckpt, opts)
        logger.info("  resuming from step %d / %d  (β = %.6g,  log Z = %+.8g)",
                    state.step, opts.n_steps, history.betas[-1], history.log_z[-1])

    artifact = Artifact(rho=rho, beta=history.betas[-1], step=state.step)
    _save_checkpoint(artifact, _ckpt)
    _archive_artifact(artifact, opts, _artifacts, final=not grid)

    env_left, env_right = _build_envs(rho, mpo, opts, _cache)

    w = len(str(opts.n_steps))
    try:
        for k, (beta, tau) in enumerate(grid, start=state.step + 1):
            logger.info("step %*d / %d: cooling ρ(β=%.6g) → ρ(β=%.6g)  (τ=%.6g)",
                        w, k, opts.n_steps, history.betas[-1], beta, tau)

            # For `order=1` a step is a single half-sweep, so the direction
            # alternates to keep consecutive pairs symmetric. `order=2`
            # ignores the flag and always returns the center to site 0.
            forward = (k - state.step) % 2 == 1
            err, dw = cool_step(rho, mpo, env_left, env_right, opts, tau, forward)

            history.betas.append(beta)
            history.taus.append(tau)
            lz, sign = rho.log_trace()
            _ensure_positive_trace(sign, beta)
            history.log_z.append(lz)
            history.max_bonds.append(max(rho.bond_dims))
            history.discarded_weights.append(dw)
            history.krylov_errors.append(err)
            if opts.obs_exact:
                history.exact_energies.append(observe(rho, mpo) / L)

            logger.info("  β = %.6g,  log Z = %+.8g,  D = %d,  "
                        "krylov err = %.4e,  dw = %.4e",
                        beta, lz, history.max_bonds[-1], err, dw)

            _save_thermal(history.summary(L, step=k, finished=False), _ckpt)
            artifact = Artifact(rho=rho, beta=beta, step=k)
            _save_checkpoint(artifact, _ckpt)
            _archive_artifact(artifact, opts, _artifacts, final=k == opts.n_steps)
    finally:
        env_left.shutdown()
        env_right.shutdown()
        # Ignore errors so a partially-written or already-removed cache
        # directory cannot mask a real exception from the cooling loop.
        if _cache is not None:
            shutil.rmtree(_cache, ignore_errors=True)

    logger.info("")

    summary = history.summary(L, step=opts.n_steps, finished=True)
    _save_thermal(summary, _ckpt)

    # Success path only: drop the mid-run checkpoint so a finished job does
    # not leave a stale tgtrg.ckpt behind. A crash earlier leaves it on disk.
    (_ckpt / 'tgtrg.ckpt').unlink(missing_ok=True)
    (_ckpt / 'tgtrg_lock.ckpt').unlink(missing_ok=True)

    return summary, artifact
