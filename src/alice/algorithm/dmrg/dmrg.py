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


"""Top-level DMRG driver: options, summary, and entry-point function.

Typical usage:

    import tomllib
    from alice import build_interaction, build_hamiltonian, MPS
    from alice import dmrg

    with open("config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    interactions, spc, geo = build_interaction(cfg["heisenberg_u1"])
    mpo = build_hamiltonian(interactions, geo.L, spc)
    mps = MPS(...)           # provide an initial state
    opts = dmrg.Options.from_toml(cfg["heisenberg_u1"]["algorithm"])
    summary = dmrg.run(mps, mpo, opts)
    print(summary.energy)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from alice.network import MPS, MPO
from alice.network.network import Network

from ..interface import AlgorithmOptions, AlgorithmSummary
from .environ import Environment, build_right_envs, left_env_boundary
from .sweep import backward_sweep, forward_sweep

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

_IMPLEMENTED_SCHEMES = {'1s', '2s', '1sp'}


def _resolve_scheme(alias: str) -> str:
    """Normalise a scheme alias to its canonical name.

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
            f"unknown DMRG scheme {alias!r}; recognised values are: {known}"
        )
    return canonical


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class Options(AlgorithmOptions):
    """DMRG run options.

    All fields have sensible defaults so `Options()` is a valid minimal
    configuration. Use `Options.from_toml` to load from an `[algorithm]`
    TOML section, or `Options.load_toml` to read directly from a file.

    Parameters
    ----------
    scheme:
        Per-site update scheme. Canonical values and their aliases:

        - `'1s'` / `'1-site'` / `'one-site'`: 1-site DMRG.
        - `'2s'` / `'2-site'` / `'two-site'`: 2-site DMRG.
        - `'1sp'` / `'1-site-plus'` / `'one-site-plus'`: 1-site-plus / CBE.
    n_sweeps:
        Maximum number of full sweeps (one forward + one backward half-sweep each).
    max_bond:
        Maximum bond dimension kept at each QR step. `None` means no limit.
    trunc_thresh:
        SVD truncation threshold forwarded to `decomp` at each canonical step.
    davidson_tol:
        Residual norm tolerance for the Davidson eigensolver.
    davidson_max_iter:
        Maximum Davidson iterations per site optimisation.
    davidson_max_subspace:
        Maximum Krylov subspace size before a thick restart.
    e_tol:
        Energy convergence criterion: DMRG stops when `|E_new - E_old| < e_tol`.
    env_cache_dir:
        Directory for environment block cache files. When set, each block is
        serialised to `{env_cache_dir}/left/{i:05d}.pt` or
        `{env_cache_dir}/right/{i:05d}.pt` and evicted from memory once it
        exits the sliding window, keeping peak memory proportional to
        `env_window` rather than to chain length. `None` (default) keeps all
        blocks in memory. Stored as `str` for TOML compatibility.
    env_async_io:
        If `True` (default), environment block I/O is submitted to a
        background thread so it overlaps with the Davidson step. Only
        relevant when `env_cache_dir` is set.
    env_window:
        Number of environment blocks kept in memory at once per direction
        (current + prefetched ahead). Defaults to `2`. Only relevant when
        `env_cache_dir` is set.
    expand_k:
        Maximum number of complement vectors added to each bond end per CBE
        step. Only used when `scheme = '1sp'`. Larger values give a richer
        expanded space at higher cost; `4` is a typical starting point.
    expand_alpha:
        Internal bond dimension used when forming the cheap 2-site tensor
        Θ̃ = truncated SVD of M[i] ⊗ M[i+1] inside the CBE expansion.
        Only used when `scheme = '1sp'`. `None` keeps the full bond (no
        additional truncation beyond the existing bond dimension).
    checkpoint_dir:
        Directory to write checkpoint files into. A `dmrg.ckpt` file
        (PyTorch format, loadable via `dmrg.Summary.load`) is written after
        every full sweep using an atomic write: the data is first serialised
        to `dmrg_lock.ckpt` in the same directory, then renamed to
        `dmrg.ckpt` on success, so a failed write cannot corrupt the
        previous checkpoint. `None` (default) resolves to `Path.cwd()` at
        the time `run()` is called, mirroring `.logging`. Pass an explicit
        path string to write elsewhere. Stored as `str` for TOML
        compatibility.
    """

    scheme: str = '1s'
    n_sweeps: int = 10
    max_bond: Optional[int] = None
    trunc_thresh: float = 1e-15
    davidson_tol: float = 1e-10
    davidson_max_iter: int = 100
    davidson_max_subspace: int = 20
    e_tol: float = 1e-8
    env_cache_dir: Optional[str] = None
    env_async_io: bool = True
    env_window: int = 2
    expand_k: int = 4
    expand_alpha: Optional[int] = None
    checkpoint_dir: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalise the scheme alias to the canonical name immediately.
        self.scheme = _resolve_scheme(self.scheme)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class Summary(AlgorithmSummary):
    """DMRG output.

    Attributes
    ----------
    energy:
        Final ground-state energy (lowest Ritz value from the last sweep).
    state:
        Optimised MPS after all sweeps.
    energies:
        Energy recorded at the end of each full sweep (forward + backward half-sweep).
    converged:
        `True` if `|E_new - E_old| < opts.e_tol` before `n_sweeps` was reached.
    n_sweeps:
        Actual number of full sweeps performed.
    bond_dims:
        Bond dimensions of `state` after convergence (length `L - 1`).
    discarded_weights:
        Discarded weight at the center bond measured during each backward
        half-sweep (2-site scheme only; always `0.0` for 1-site).
    """

    energy: float
    state: MPS
    energies: List[float] = field(default_factory=list)
    converged: bool = False
    n_sweeps: int = 0
    bond_dims: List[int] = field(default_factory=list)
    discarded_weights: List[float] = field(default_factory=list)

    def serialize(self) -> Dict:
        """Serialize the summary to a plain dict compatible with `torch.save`.

        The MPS state is serialized via `Network.serialize`, which produces
        a `weights_only`-safe dict of tensors and primitives.

        Returns
        -------
        Dict
            Serialized summary with keys `"version"`, `"energy"`,
            `"energies"`, `"converged"`, `"n_sweeps"`, `"bond_dims"`,
            and `"state"`.
        """
        return {
            'version': 1,
            'energy': self.energy,
            'energies': self.energies,
            'converged': self.converged,
            'n_sweeps': self.n_sweeps,
            'bond_dims': self.bond_dims,
            'discarded_weights': self.discarded_weights,
            # Network.serialize() returns a weights_only-safe dict of tensors
            # and primitives, preserving symmetry structure and index metadata.
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
            Device to place all MPS tensor blocks on. Defaults to `'cpu'`.

        Returns
        -------
        Summary
            Reconstructed summary with the MPS state placed on `device`.

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
            n_sweeps=data['n_sweeps'],
            bond_dims=data['bond_dims'],
            discarded_weights=data.get('discarded_weights', []),
            state=Network.deserialize(data['state'], device=device),
        )


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _save_checkpoint(
    mps: MPS,
    energies: List[float],
    discarded_weights: List[float],
    sweep_count: int,
    ckpt_dir: Path,
) -> None:
    """Write an atomic checkpoint of the current DMRG state.

    Serialises the current MPS and energy history to `dmrg_lock.ckpt` in
    `ckpt_dir`, then renames it to `dmrg.ckpt`. The rename is atomic on
    POSIX systems, so a crash during serialisation cannot corrupt the
    previously written checkpoint.

    Parameters
    ----------
    mps:
        Current MPS (modified in-place by the sweep).
    energies:
        Energy history up to and including the current sweep.
    discarded_weights:
        Discarded-weight history up to and including the current sweep.
    sweep_count:
        Number of full sweeps completed so far.
    ckpt_dir:
        Directory in which `dmrg.ckpt` and `dmrg_lock.ckpt` are written.
    """
    import torch

    summary = Summary(
        energy=energies[-1],
        state=mps,
        energies=list(energies),
        converged=False,
        n_sweeps=sweep_count,
        bond_dims=list(mps.bond_dims),
        discarded_weights=list(discarded_weights),
    )

    lock_path = ckpt_dir / 'dmrg_lock.ckpt'
    ckpt_path = ckpt_dir / 'dmrg.ckpt'

    torch.save(summary.serialize(), lock_path)
    # Atomic rename: on POSIX this is guaranteed to be atomic; on Windows it
    # is best-effort (Path.replace uses MoveFileExW which is not atomic but
    # still avoids leaving a half-written dmrg.ckpt on disk).
    lock_path.replace(ckpt_path)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(mps: MPS, mpo: MPO, opts: Optional[Options] = None) -> Summary:
    """Run DMRG to find the ground state of a Hamiltonian MPO.

    Performs alternating forward and backward half-sweeps, optimising each site
    tensor with the Davidson eigensolver, until the energy converges or the
    maximum number of sweeps is reached.

    Parameters
    ----------
    mps:
        Initial MPS state. Canonicalised in-place to `center = 0` before
        the first sweep.
    mpo:
        Hamiltonian MPO of the same length as `mps`.
    opts:
        Run options. Defaults to `Options()` if `None`.

    Returns
    -------
    Summary
        Ground-state energy, optimised MPS, energy history, and convergence
        information.

    Raises
    ------
    NotImplementedError
        If `opts.scheme` is not `'1s'` (other schemes are planned but not
        yet implemented).
    ValueError
        If `mps` and `mpo` have different lengths.
    """
    if opts is None:
        opts = Options()

    if opts.scheme not in _IMPLEMENTED_SCHEMES:
        raise NotImplementedError(
            f"DMRG scheme {opts.scheme!r} is recognised but not yet implemented; "
            f"implemented schemes are: {', '.join(sorted(_IMPLEMENTED_SCHEMES))}"
        )

    if mps.L != mpo.L:
        raise ValueError(
            f"mps and mpo must have the same length, got {mps.L} and {mpo.L}"
        )

    # Bring the MPS into right-canonical form with center at site 0.
    # This is required by build_right_envs and establishes center for the sweep.
    mps.canonical(0)

    L = mps.L
    _2s = (opts.scheme == '2s')
    _1sp = (opts.scheme == '1sp')

    # Resolve the optional disk-cache directory and create sub-dirs if needed.
    _cache: Optional[Path] = Path(opts.env_cache_dir) if opts.env_cache_dir else None
    if _cache is not None:
        (_cache / 'left').mkdir(parents=True, exist_ok=True)
        (_cache / 'right').mkdir(parents=True, exist_ok=True)

    # Resolve the checkpoint directory; default to Path.cwd() when unset,
    # mirroring the convention used by configure_logging.
    _ckpt: Path = Path(opts.checkpoint_dir) if opts.checkpoint_dir is not None else Path.cwd()
    _ckpt.mkdir(parents=True, exist_ok=True)

    # For 2-site DMRG the effective fetch ranges are narrower than [0, L-1]:
    #   env_left  is never fetched at index L-1 (that tensor is never written).
    #   env_right is never fetched at index 0  (the boundary block is written
    #   by build_right_envs but consumed only via env_left in 1-site).
    # 1-site-plus uses the same fetch ranges as 1-site: both env_left and
    # env_right are fetched at all sites because CBE reads env_right[i+1]
    # and env_left[i-1] for the complement computation at each bond.
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

    # Initialise the left boundary and all right environment blocks.
    # __setitem__ auto-caches each block to disk when _cache is set.
    env_left[0] = left_env_boundary(mps, mpo)
    build_right_envs(mps, mpo, env_right)

    energies: List[float] = []
    discarded_weights: List[float] = []
    converged = False
    sweep_count = 0
    prev_energy = math.inf

    # Log startup header and options before the first sweep.
    max_bond_str = str(opts.max_bond) if opts.max_bond is not None else 'unlimited'
    logger.info("─" * 60)
    logger.info("Commencing: DMRG Algorithm".center(60))
    logger.info("─" * 60)
    logger.info("")
    logger.info("  scheme            : %s", opts.scheme)
    logger.info("  chain length      : %d", L)
    logger.info("  max sweeps        : %d", opts.n_sweeps)
    logger.info("  max bond dim      : %s", max_bond_str)
    logger.info("  conv thresh       : %.2e", opts.e_tol)
    logger.info("  trunc thresh      : %.2e", opts.trunc_thresh)
    logger.info("  davidson tol      : %.2e", opts.davidson_tol)
    logger.info("  davidson max iter : %d", opts.davidson_max_iter)
    logger.info("  davidson max space: %d", opts.davidson_max_subspace)
    if _1sp:
        alpha_str = str(opts.expand_alpha) if opts.expand_alpha is not None else 'full bond'
        logger.info("  expand k          : %d", opts.expand_k)
        logger.info("  expand alpha      : %s", alpha_str)
    if _cache is not None:
        logger.info("  env cache dir     : %s", _cache)
        logger.info("  env window        : %d", opts.env_window)
        logger.info("  env async I/O     : %s", opts.env_async_io)
    if opts.checkpoint_dir is not None:
        logger.info("  checkpoint dir    : %s", _ckpt)
    logger.info("")

    w = len(str(opts.n_sweeps))
    try:
        for sweep_idx in range(opts.n_sweeps):
            # Blank debug line between sweeps for visual separation in the log file.
            if sweep_idx > 0:
                logger.debug("")
            logger.debug("sweep %*d / %d: forward sweep initiated", w, sweep_idx + 1, opts.n_sweeps)

            # Forward half-sweep: center moves from 0 to L-1.
            right_energy = forward_sweep(mps, mpo, env_left, env_right, opts)

            logger.debug("sweep %*d / %d: forward sweep finished", w, sweep_idx + 1, opts.n_sweeps)
            logger.debug("  local E = %+.12g", right_energy)
            logger.debug("")
            logger.debug("sweep %*d / %d: backward sweep initiated", w, sweep_idx + 1, opts.n_sweeps)

            # Backward half-sweep: center moves from L-1 to 0; energy recorded here.
            energy, dw = backward_sweep(mps, mpo, env_left, env_right, opts)

            delta_e = abs(energy - prev_energy)
            energies.append(energy)
            discarded_weights.append(dw)
            sweep_count += 1

            _save_checkpoint(mps, energies, discarded_weights, sweep_count, _ckpt)

            logger.debug("sweep %*d / %d: backward sweep finished", w, sweep_idx + 1, opts.n_sweeps)
            logger.debug("  local E = %+.12g", energy)
            logger.debug("  ΔE = %+.4e", energy - prev_energy)
            logger.info(
                "sweep %*d / %d: E = %+.12g, |ΔE| = %.4e, dw = %.4e",
                w, sweep_idx + 1, opts.n_sweeps, energy, delta_e, dw,
            )

            # Check energy convergence.
            if delta_e < opts.e_tol:
                converged = True
                logger.info("converged after %d sweep(s)", sweep_count)
                break
            prev_energy = energy
    finally:
        # Flush pending async writes and release the I/O thread.
        env_left.shutdown()
        env_right.shutdown()

    logger.info("")

    return Summary(
        energy=energies[-1] if energies else math.nan,
        state=mps,
        energies=energies,
        converged=converged,
        n_sweeps=sweep_count,
        bond_dims=list(mps.bond_dims),
        discarded_weights=discarded_weights,
    )
