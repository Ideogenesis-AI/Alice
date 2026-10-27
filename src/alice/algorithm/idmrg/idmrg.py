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


"""Top-level iDMRG driver: options, summary, and entry-point function.

Typical usage:

    import tomllib
    from alice import build_interaction, build_ihamiltonian, iMPS
    from alice.algorithm import idmrg

    with open("config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    interactions, spc, geo = build_interaction(cfg["heisenberg_u1"])
    impo = build_ihamiltonian(interactions, geo.L, spc)
    imps = iMPS(...)         # provide an initial unit cell
    opts = idmrg.Options.from_toml(cfg["heisenberg_u1"]["algorithm"])
    summary = idmrg.run(imps, impo, opts)
    print(summary.energy)    # energy per site

How a run proceeds
------------------
The engine is VUMPS. The unit of work is an iteration, counted by
`n_iters`: every position of the unit cell is solved against one set of
environments, the environments are re-solved against the updated state,
and the cycle repeats. The positions within an iteration are independent,
so their order is immaterial.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from alice.network import iMPO, iMPS
from alice.network.infinite import iNetwork

from ..interface import AlgorithmOptions, AlgorithmSummary
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
            f"unknown iDMRG scheme {alias!r}; recognised values are: {known}"
        )
    return canonical


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class Options(AlgorithmOptions):
    """iDMRG run options.

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
        smaller.
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
    checkpoint_dir:
        Directory to write checkpoint files into. An `idmrg.ckpt` file
        (PyTorch format, loadable via `idmrg.Summary.load`) is written after
        every iteration using an atomic write: the data is first serialized
        to `idmrg_lock.ckpt` in the same directory, then renamed to
        `idmrg.ckpt` on success, so a failed write cannot corrupt the
        previous checkpoint. Once `run()` finishes successfully, the final
        `Summary` is archived under `artifacts_dir` (see below) and
        `idmrg.ckpt` is removed, since it is redundant with the archived
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
    Environments are not cached to disk. A unit cell holds `N` environment
    blocks and all of them are live at once, so there is nothing to evict.
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
    """iDMRG output.

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
    """Write an atomic checkpoint of the current iDMRG state.

    Serializes the current iMPS and energy history to `idmrg_lock.ckpt` in
    `ckpt_dir`, then renames it to `idmrg.ckpt`. The rename is atomic on
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
        Directory in which `idmrg.ckpt` and `idmrg_lock.ckpt` are written.
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

    lock_path = ckpt_dir / 'idmrg_lock.ckpt'
    ckpt_path = ckpt_dir / 'idmrg.ckpt'

    torch.save(summary.serialize(), lock_path)
    lock_path.replace(ckpt_path)


def _save_final_artifact(summary: Summary, artifacts_dir: Path) -> None:
    """Write the final `Summary` atomically to `state.ckpt` in `artifacts_dir`.

    Unlike the per-iteration `idmrg.ckpt` checkpoint, `summary` here carries
    the correct `converged` flag, since it is built after the loop exits
    rather than while it is still in progress.

    Parameters
    ----------
    summary:
        Final iDMRG summary, including the optimized iMPS.
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
            f"iDMRG scheme {opts.scheme!r} is recognised but not yet implemented; "
            f"implemented schemes are: {', '.join(sorted(_IMPLEMENTED_SCHEMES))}"
        )

    # A finite state shares no ancestry with iMPS, so this rejects it rather
    # than letting the cell machinery quietly read an open end as the bond
    # onto the next cell.
    if not isinstance(imps, iMPS) or not isinstance(impo, iMPO):
        raise TypeError(
            f"idmrg.run operates on an infinite MPS and MPO, got "
            f"{type(imps).__name__} and {type(impo).__name__}; "
            "use dmrg.run for finite networks"
        )

    if imps.N != impo.N:
        raise ValueError(
            f"imps and impo must have the same unit cell, got {imps.N} and {impo.N}"
        )

    N = imps.N

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
    logger.info("Commencing: iDMRG Algorithm".center(60))
    logger.info("─" * 60)
    logger.info("")
    logger.info("  scheme            : %s (VUMPS)", opts.scheme)
    logger.info("  unit cell         : %d", N)
    logger.info("  bond dim          : %d", max(imps.bond_dims))
    logger.info("  max iterations    : %d", opts.n_iters)
    logger.info("  grad thresh       : %.2e", opts.grad_tol)
    logger.info("  davidson tol      : %.2e", opts.davidson_tol)
    logger.info("  davidson max iter : %d", opts.davidson_max_iter)
    logger.info("  davidson max space: %d", opts.davidson_max_subspace)
    logger.info("  gauge tol         : %.2e", opts.gauge_tol)
    logger.info("  env tol           : %.2e", opts.env_tol)
    if opts.checkpoint_dir is not None:
        logger.info("  checkpoint dir    : %s", _ckpt)
    logger.info("  artifacts dir     : %s", _artifacts)
    logger.info("")

    # Gauge fixing and the environment solve are one step: the fixed points
    # are defined against the canonical tensors, so an update that has moved
    # them invalidates both at once.
    imps.canonical(0, tol=opts.gauge_tol, maxiter=opts.gauge_maxiter)
    env_left, env_right, cell_energy = cell_environments(
        imps, impo, tol=opts.env_tol, maxiter=opts.env_maxiter,
    )

    w = len(str(opts.n_iters))
    for iter_idx in range(opts.n_iters):
        if iter_idx > 0:
            logger.debug("")
        logger.debug("iteration %*d / %d initiated", w, iter_idx + 1, opts.n_iters)

        gradient, discarded = cell_update(imps, impo, env_left, env_right, opts)

        # The update installs its own gauge, so only the fixed points have
        # to be re-solved. They give the energy density and the environments
        # for the next iteration at once.
        env_left, env_right, cell_energy = cell_environments(
            imps, impo, tol=opts.env_tol, maxiter=opts.env_maxiter,
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

        logger.info(
            "iteration %*d / %d: e = %+.12g, |Δe| = %.4e, |grad| = %.4e",
            w, iter_idx + 1, opts.n_iters, energy, delta_e, gradient,
        )

        if gradient < opts.grad_tol:
            converged = True
            logger.info("converged after %d iteration(s)", iter_count)
            break
        prev_energy = energy

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
    (_ckpt / 'idmrg.ckpt').unlink(missing_ok=True)
    (_ckpt / 'idmrg_lock.ckpt').unlink(missing_ok=True)

    return summary
