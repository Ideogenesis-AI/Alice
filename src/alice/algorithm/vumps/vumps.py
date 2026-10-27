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


"""Top-level VUMPS driver: options, summary, and entry-point function.

Typical usage:

    import tomllib
    from alice import build_interaction, build_ihamiltonian, iMPS
    from alice.algorithm import vumps

    with open("config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    interactions, spc, geo = build_interaction(cfg["heisenberg_u1"])
    impo = build_ihamiltonian(interactions, geo.L, spc)
    imps = iMPS(...)         # provide an initial unit cell
    opts = vumps.Options.from_toml(cfg["heisenberg_u1"]["algorithm"])
    summary = vumps.run(imps, impo, opts)
    print(summary.energy)    # energy per site

How a run proceeds
------------------
The unit of work is an iteration, counted by `n_iters`: every position of
the unit cell is solved against one set of environments, the environments
are re-solved against the updated state, and the cycle repeats. The
positions within an iteration are independent, so their order is
immaterial.

Which positions an iteration solves depends on the scheme. The two-site
scheme, the default, solves a pair of neighboring sites at a time and
splits the answer, and that split is what lets the bond dimension grow
from one iteration to the next, up to `max_bond`. The one-site scheme
solves single sites and bonds, and takes polar decompositions rather
than splits, so it never discards anything and never gains width: a run
under `'1s'` stays at the bond dimension its initial state carried.

Convergence is measured on the gradient. The Davidson eigenvalues belong
to regularized local problems and carry no physical scale, since the
energy of an infinite chain is infinite; the reported energy density comes
from the environment fixed point instead. The convergence measure is
`|AC - AL C|`, the norm of the energy gradient on the uniform MPS
manifold, which vanishes at a stationary point and keeps moving long after
the energy between iterations has flattened out.
"""

from __future__ import annotations

import logging
import math
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from alice.network import iMPO, iMPS
from alice.network.infinite import iNetwork

from ..interface import AlgorithmOptions, AlgorithmSummary
from .environ import Environment
from .sweep import cell_environments, cell_update

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheme alias resolution
# ---------------------------------------------------------------------------

# Maps every accepted TOML alias to the canonical scheme name.
_SCHEME_ALIASES: Dict[str, str] = {
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
        If `alias` is not a recognised scheme name.
    """
    canonical = _SCHEME_ALIASES.get(alias.lower())
    if canonical is None:
        known = ', '.join(sorted(_SCHEME_ALIASES))
        raise ValueError(
            f"unknown VUMPS scheme {alias!r}; recognised values are: {known}"
        )
    return canonical


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class Options(AlgorithmOptions):
    """VUMPS run options.

    All fields have sensible defaults so `Options()` is a valid minimal
    configuration. Use `Options.from_toml` to load from an `[algorithm]`
    TOML section, or `Options.load_toml` to read directly from a file.

    Parameters
    ----------
    scheme:
        Update scheme. Canonical values and their aliases:

        - `'2s'` / `'2-site'` / `'two-site'`: two-site VUMPS, the default.
          The only scheme that can change the bond dimension.
        - `'1s'` / `'1-site'` / `'one-site'`: one-site VUMPS. Cheaper per
          iteration, but confined to the bond dimension of the initial
          state.
        - `'1sp'` / `'1-site-plus'` / `'one-site-plus'`: recognised but not
          implemented, so a config written for it fails with a clear
          message rather than a typo error.
    n_iters:
        Maximum number of VUMPS iterations. Each one solves every position
        of the cell against one set of environments.
    max_bond:
        Largest bond dimension a split may keep. `None` means no limit, in
        which case every split multiplies the bond dimension by the
        physical dimension and the state grows until `trunc_thresh` stops
        it. Ignored by `'1s'`, which never splits anything.
    trunc_thresh:
        Smallest singular value a split may keep, applied before
        `max_bond`. Ignored by `'1s'`.
    davidson_tol:
        Residual norm tolerance for the Davidson eigensolver. Keep it well
        below `grad_tol`. The gradient measures how far `AC` and `C`
        disagree, so neither can be trusted beyond the accuracy it was
        solved to and the gradient cannot fall below this value. Loosening
        it to save work backfires: the run then needs more iterations than
        the cheaper solves save, and stalls at a gradient around
        `davidson_tol` without ever converging.
    davidson_max_iter:
        Maximum Davidson iterations per local problem.
    davidson_max_subspace:
        Maximum Krylov subspace size before a thick restart.
    grad_tol:
        Convergence criterion on the VUMPS gradient norm `|AC - AL C|`,
        maximized over the cell. Cannot be met unless `davidson_tol` is
        smaller. Under `'2s'` against a bond cap that bites, it cannot be
        met at all: the two-site solve keeps asking for a wider bond than
        it is allowed, and the gradient settles at a floor set by the
        discarded weight rather than falling. A run that has grown to
        `max_bond` and stopped improving is converged in every sense that
        matters, and polishing it with `'1s'` will drive the gradient the
        rest of the way down.
    gauge_tol:
        Convergence tolerance for the unit-cell gauge iteration. Used only
        to canonicalize the initial state: every iteration after that
        installs its own gauge through `iMPS.set_canonical`.
    gauge_maxiter:
        Maximum number of gauge sweeps around the unit cell, for that same
        initial canonicalization.
    env_tol:
        Convergence tolerance for the environment fixed point. The single
        most important setting for accuracy, and the most expensive: the
        environments are re-solved from scratch every iteration and account
        for roughly half the running time.
    env_maxiter:
        Maximum number of unit cells absorbed by the environment solver.
        Running out logs a warning and continues with the best estimate
        rather than abandoning the run, because a critical model may not
        reach `env_tol` at any budget. A run that warns on most iterations
        is reporting an energy limited by `env_maxiter`, not by the bond
        dimension.
    env_cache_dir:
        Root directory for environment block cache files. When set,
        `run()` creates a unique subdirectory inside it (first 8 hex
        characters of a UUID4, e.g. `{env_cache_dir}/a1b2c3d4/`) so that
        concurrent runs sharing the same config do not overwrite each
        other's blocks. Inside it, `right/{k:05d}.pt` files are written
        and dropped from memory once they leave the sliding window,
        keeping the footprint proportional to `env_window` rather than to
        the length of the cell. Only the right side appears there: the
        left side is carried one block at a time and never reaches disk.
        The subdirectory is removed when `run()` returns or raises.
        `None` (default) keeps every block in memory. Stored as `str` for
        TOML compatibility.
    env_async_io:
        If `True` (default), environment block I/O is submitted to a
        background thread so it overlaps with the Davidson solves. Only
        relevant when `env_cache_dir` is set.
    env_window:
        Number of right environment blocks kept in memory at once, the
        one being read plus those prefetched ahead. Defaults to `2`, which
        is the smallest value both schemes can run at. Only relevant when
        `env_cache_dir` is set.
    checkpoint_dir:
        Directory to write checkpoint files into. An `vumps.ckpt` file
        (PyTorch format, loadable via `vumps.Summary.load`) is written after
        every iteration using an atomic write: the data is first serialized
        to `vumps_lock.ckpt` in the same directory, then renamed to
        `vumps.ckpt` on success, so a failed write cannot corrupt the
        previous checkpoint. Once `run()` finishes successfully, the final
        `Summary` is archived under `artifacts_dir` (see below) and
        `vumps.ckpt` is removed, since it is redundant with the archived
        artifact. `None` (default) resolves to `Path.cwd()` at the time
        `run()` is called, mirroring `.logging`. Pass an explicit path
        string to write elsewhere. Stored as `str` for TOML compatibility.
    artifacts_dir:
        Directory for the final `Summary` artifact, written as `state.ckpt`
        when `run()` finishes successfully. `None` (default) resolves to
        `artifacts/` under `checkpoint_dir`. Pass an explicit path string to
        archive the final state elsewhere, independent of `checkpoint_dir`.
        Stored as `str` for TOML compatibility.

    Notes
    -----
    An environment block scales as `chi^2 * w`, and a pass over the cell
    needs one on every bond from each side. On a chain that is what it
    sounds like, `2N` small tensors, and the defaults keep them all in
    memory. On a quasi-1D cylinder the unit cell is one column, so `N`
    runs to dozens while `chi` and `w` both grow with the circumference,
    and that is the case `env_cache_dir` exists for.

    Half of those blocks were never needed. A pass produces its left
    blocks in the order it consumes them, so one running tensor does the
    work of `N`, and that is now how both schemes run whether or not
    anything is cached. Only the right side is stored, and only it can be
    evicted. See `alice.algorithm.vumps.sweep` for the argument.
    """

    scheme: str = '2s'
    n_iters: int = 100
    max_bond: Optional[int] = None
    trunc_thresh: float = 1e-15
    davidson_tol: float = 1e-10
    davidson_max_iter: int = 100
    davidson_max_subspace: int = 20
    grad_tol: float = 1e-10
    gauge_tol: float = 1e-12
    gauge_maxiter: int = 200
    env_tol: float = 1e-12
    env_maxiter: int = 200
    env_cache_dir: Optional[str] = None
    env_async_io: bool = True
    env_window: int = 2
    checkpoint_dir: Optional[str] = None
    artifacts_dir: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalize the scheme alias to the canonical name immediately.
        self.scheme = _resolve_scheme(self.scheme)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class Summary(AlgorithmSummary):
    """VUMPS output.

    Attributes
    ----------
    energy:
        Final ground-state energy *per site*, from the environment fixed
        point after the last iteration.
    state:
        Optimized iMPS, gauge-fixed.
    energies:
        Energy per site recorded at the end of each iteration.
    converged:
        `True` if the gradient norm fell below `opts.grad_tol` before
        `n_iters` was reached.
    n_iters:
        Actual number of iterations performed.
    bond_dims:
        Bond dimensions of `state` (length `N`, the last entry being the
        bond onto the next copy of the cell).
    gradients:
        VUMPS gradient norm `|AC - AL C|` recorded at each iteration.
    discarded_weights:
        Largest weight discarded by a split, recorded at each iteration.
        All zeros under `'1s'`, which never splits anything.
    """

    energy: float
    state: iMPS
    energies: List[float] = field(default_factory=list)
    converged: bool = False
    n_iters: int = 0
    bond_dims: List[int] = field(default_factory=list)
    gradients: List[float] = field(default_factory=list)
    discarded_weights: List[float] = field(default_factory=list)

    def serialize(self) -> Dict:
        """Serialize the summary to a plain dict compatible with `torch.save`.

        The iMPS state is serialized via `iNetwork.serialize`, which produces
        a `weights_only`-safe dict of tensors and primitives.

        Returns
        -------
        Dict
            Serialized summary with keys `"version"`, `"energy"`,
            `"energies"`, `"converged"`, `"n_iters"`, `"bond_dims"`,
            `"gradients"`, `"discarded_weights"`, and `"state"`.
        """
        return {
            'version': 1,
            'energy': self.energy,
            'energies': self.energies,
            'converged': self.converged,
            'n_iters': self.n_iters,
            'bond_dims': self.bond_dims,
            'gradients': self.gradients,
            'discarded_weights': self.discarded_weights,
            'state': self.state.serialize(),
        }

    @classmethod
    def deserialize(cls, data: Dict, device: str = 'cpu') -> Summary:
        """Reconstruct a `Summary` from a dict produced by `serialize`.

        Parameters
        ----------
        data:
            Dict previously returned by `serialize`.
        device:
            Device to place all iMPS tensor blocks on. Defaults to `'cpu'`.

        Returns
        -------
        Summary
            Reconstructed summary with the iMPS state placed on `device`.

        Raises
        ------
        ValueError
            If `data["version"]` is not `1`.
        """
        version = data.get('version', 1)
        if version != 1:
            raise ValueError(f"Unsupported Summary serialization version: {version!r}")
        return cls(
            energy=data['energy'],
            energies=data['energies'],
            converged=data['converged'],
            n_iters=data['n_iters'],
            bond_dims=data['bond_dims'],
            gradients=data.get('gradients', []),
            discarded_weights=data.get('discarded_weights', []),
            state=iNetwork.deserialize(data['state'], device=device),
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    imps: iMPS,
    energies: List[float],
    gradients: List[float],
    discarded_weights: List[float],
    iter_count: int,
    ckpt_dir: Path,
) -> None:
    """Write an atomic checkpoint of the current VUMPS state.

    Serializes the current iMPS and energy history to `vumps_lock.ckpt` in
    `ckpt_dir`, then renames it to `vumps.ckpt`. The rename is atomic on
    POSIX systems, so a crash during serialization cannot corrupt the
    previously written checkpoint.

    Parameters
    ----------
    imps:
        Current iMPS (modified in-place by the iteration).
    energies:
        Energy-per-site history up to and including the current iteration.
    gradients:
        Gradient-norm history up to and including the current iteration.
    discarded_weights:
        Discarded-weight history up to and including the current iteration.
    iter_count:
        Number of iterations completed so far.
    ckpt_dir:
        Directory in which `vumps.ckpt` and `vumps_lock.ckpt` are written.
    """
    import torch

    summary = Summary(
        energy=energies[-1],
        state=imps,
        energies=list(energies),
        converged=False,
        n_iters=iter_count,
        bond_dims=list(imps.bond_dims),
        gradients=list(gradients),
        discarded_weights=list(discarded_weights),
    )

    lock_path = ckpt_dir / 'vumps_lock.ckpt'
    ckpt_path = ckpt_dir / 'vumps.ckpt'

    torch.save(summary.serialize(), lock_path)
    lock_path.replace(ckpt_path)


def _save_final_artifact(summary: Summary, artifacts_dir: Path) -> None:
    """Write the final `Summary` atomically to `state.ckpt` in `artifacts_dir`.

    Unlike the per-iteration `vumps.ckpt` checkpoint, `summary` here carries
    the correct `converged` flag, since it is built after the loop exits
    rather than while it is still in progress.

    Parameters
    ----------
    summary:
        Final VUMPS summary, including the optimized iMPS.
    artifacts_dir:
        Directory in which `state.ckpt` and `state_lock.ckpt` are written.
    """
    import torch

    lock_path = artifacts_dir / 'state_lock.ckpt'
    path = artifacts_dir / 'state.ckpt'
    torch.save(summary.serialize(), lock_path)
    lock_path.replace(path)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(imps: iMPS, impo: iMPO, opts: Optional[Options] = None) -> Summary:
    """Run VUMPS to find the ground state of a translation-invariant Hamiltonian.

    Each iteration gauge-fixes the unit cell, solves the environment fixed
    points, updates every center tensor and bond matrix of the cell against
    them, and reports the energy density from a fresh environment solve.
    Convergence is measured on the gradient norm.

    Parameters
    ----------
    imps:
        Initial infinite MPS. Gauge-fixed in place before the first
        iteration and after every one, so it is left in canonical form.
        Under `'2s'` it may come out wider than it went in; under `'1s'`
        its bond dimension is kept throughout.
    impo:
        Hamiltonian iMPO sharing the unit cell of `imps`.
    opts:
        Run options. Defaults to `Options()` if `None`.

    Returns
    -------
    Summary
        Ground-state energy per site, optimized iMPS, energy and gradient
        history, and convergence information. Also archived as `state.ckpt`
        under `opts.artifacts_dir`.

    Raises
    ------
    NotImplementedError
        If `opts.scheme` is recognised but not implemented.
    TypeError
        If `imps` or `impo` is a finite network; use `dmrg.run` instead.
    ValueError
        If `imps` and `impo` have different unit cells, or if `'2s'` is
        asked for a one-site cell.
    """
    if opts is None:
        opts = Options()

    if opts.scheme not in _IMPLEMENTED_SCHEMES:
        raise NotImplementedError(
            f"VUMPS scheme {opts.scheme!r} is recognised but not yet implemented; "
            f"implemented schemes are: {', '.join(sorted(_IMPLEMENTED_SCHEMES))}"
        )

    # A finite state shares no ancestry with iMPS, so this rejects it rather
    # than letting the cell machinery quietly read an open end as the bond
    # onto the next cell.
    if not isinstance(imps, iMPS) or not isinstance(impo, iMPO):
        raise TypeError(
            f"vumps.run operates on an infinite MPS and MPO, got "
            f"{type(imps).__name__} and {type(impo).__name__}; "
            "use dmrg.run for finite networks"
        )

    if imps.N != impo.N:
        raise ValueError(
            f"imps and impo must have the same unit cell, got {imps.N} and {impo.N}"
        )

    N = imps.N

    # A unique subdirectory per run, so that two runs sharing a config do
    # not write over each other. Resolved here to be logged with the rest
    # of the settings, but not created until the try that removes it again
    # is about to open, so that nothing can fail in between.
    _cache: Optional[Path] = None
    if opts.env_cache_dir:
        _cache = Path(opts.env_cache_dir) / uuid.uuid4().hex[:8]

    _ckpt: Path = Path(opts.checkpoint_dir) if opts.checkpoint_dir is not None else Path.cwd()
    _ckpt.mkdir(parents=True, exist_ok=True)
    _artifacts: Path = (
        Path(opts.artifacts_dir) if opts.artifacts_dir is not None else _ckpt / 'artifacts'
    )
    _artifacts.mkdir(parents=True, exist_ok=True)

    energies: List[float] = []
    gradients: List[float] = []
    discarded_weights: List[float] = []
    converged = False
    iter_count = 0
    prev_energy = math.inf

    logger.info("─" * 60)
    logger.info("Commencing: VUMPS Algorithm".center(60))
    logger.info("─" * 60)
    logger.info("")
    logger.info("  scheme            : %s", opts.scheme)
    logger.info("  unit cell         : %d", N)
    logger.info("  bond dim          : %d", max(imps.bond_dims))
    if opts.scheme != '1s':
        max_bond_str = str(opts.max_bond) if opts.max_bond is not None else 'unlimited'
        logger.info("  max bond dim      : %s", max_bond_str)
        logger.info("  trunc thresh      : %.2e", opts.trunc_thresh)
    logger.info("  max iterations    : %d", opts.n_iters)
    logger.info("  grad thresh       : %.2e", opts.grad_tol)
    logger.info("  davidson tol      : %.2e", opts.davidson_tol)
    logger.info("  davidson max iter : %d", opts.davidson_max_iter)
    logger.info("  davidson max space: %d", opts.davidson_max_subspace)
    logger.info("  gauge tol         : %.2e", opts.gauge_tol)
    logger.info("  env tol           : %.2e", opts.env_tol)
    if _cache is not None:
        logger.info("  env cache dir     : %s", _cache)
        logger.info("  env window        : %d", opts.env_window)
    if opts.checkpoint_dir is not None:
        logger.info("  checkpoint dir    : %s", _ckpt)
    logger.info("  artifacts dir     : %s", _artifacts)
    logger.info("")

    # Gauge fixing and the environment solve are one step: the fixed points
    # are defined against the canonical tensors, so an update that has moved
    # them invalidates both at once.
    imps.canonical(0, tol=opts.gauge_tol, maxiter=opts.gauge_maxiter)

    # Only the right side is stored, so only `right/` is made.
    if _cache is not None:
        (_cache / 'right').mkdir(parents=True, exist_ok=True)

    # One store for the whole run, so that the cache directory and the
    # I/O thread are set up once rather than per iteration. The blocks in
    # it are overwritten each time the fixed points are re-solved.
    env_right = Environment(
        N,
        _cache / 'right' if _cache is not None else None,
        async_io=opts.env_async_io,
        window=opts.env_window,
    )

    try:
        env_left, env_right, cell_energy = cell_environments(
            imps, impo, tol=opts.env_tol, maxiter=opts.env_maxiter,
            env_right=env_right,
        )

        w = len(str(opts.n_iters))
        for iter_idx in range(opts.n_iters):
            if iter_idx > 0:
                logger.debug("")
            logger.debug("iteration %*d / %d initiated", w, iter_idx + 1, opts.n_iters)

            gradient, discarded = cell_update(imps, impo, env_left, env_right, opts)

            # The update installs its own gauge, so only the fixed points
            # have to be re-solved. They give the energy density and the
            # environments for the next iteration at once.
            env_left, env_right, cell_energy = cell_environments(
                imps, impo, tol=opts.env_tol, maxiter=opts.env_maxiter,
                env_right=env_right,
            )
            energy = cell_energy / N

            delta_e = abs(energy - prev_energy)
            energies.append(energy)
            gradients.append(gradient)
            discarded_weights.append(discarded)
            iter_count += 1

            _save_checkpoint(
                imps, energies, gradients, discarded_weights, iter_count, _ckpt,
            )

            if opts.scheme == '1s':
                logger.info(
                    "iteration %*d / %d: e = %+.12g, |Δe| = %.4e, |grad| = %.4e",
                    w, iter_idx + 1, opts.n_iters, energy, delta_e, gradient,
                )
            else:
                logger.info(
                    "iteration %*d / %d: e = %+.12g, |Δe| = %.4e, |grad| = %.4e, "
                    "D = %d, dw = %.4e",
                    w, iter_idx + 1, opts.n_iters, energy, delta_e, gradient,
                    max(imps.bond_dims), discarded,
                )

            if gradient < opts.grad_tol:
                converged = True
                logger.info("converged after %d iteration(s)", iter_count)
                break
            prev_energy = energy
    finally:
        # Flush pending writes and release the I/O thread, then drop the
        # run's cache subdirectory. Errors there are ignored so that a
        # half-written directory cannot mask a real failure in the loop.
        env_right.shutdown()
        if _cache is not None:
            shutil.rmtree(_cache, ignore_errors=True)

    logger.info("")

    summary = Summary(
        energy=energies[-1] if energies else math.nan,
        state=imps,
        energies=energies,
        converged=converged,
        n_iters=iter_count,
        bond_dims=list(imps.bond_dims),
        gradients=gradients,
        discarded_weights=discarded_weights,
    )

    _save_final_artifact(summary, _artifacts)
    (_ckpt / 'vumps.ckpt').unlink(missing_ok=True)
    (_ckpt / 'vumps_lock.ckpt').unlink(missing_ok=True)

    return summary
