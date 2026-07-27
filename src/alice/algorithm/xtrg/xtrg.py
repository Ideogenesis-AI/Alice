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

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nicole import Index
from nicole import deserialize as _deserialize_tensor
from nicole import serialize as _serialize_tensor

from alice.network import MPO
from alice.network.thermal import NormalMPO, thermal_mpo

from ..interface import AlgorithmOptions, AlgorithmSummary
from .environ import Environment, build_right_envs, left_env_boundary
from .sweep import backward_sweep, forward_sweep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheme alias resolution
# ---------------------------------------------------------------------------

_SCHEME_ALIASES: Dict[str, str] = {
    '1s': '1s',
    '1-site': '1s',
    'one-site': '1s',
    '2s': '2s',
    '2-site': '2s',
    'two-site': '2s',
}

_IMPLEMENTED_SCHEMES = {'1s', '2s'}


def _resolve_scheme(alias: str) -> str:
    """Normalise a scheme alias to its canonical name.

    Parameters
    ----------
    alias:
        User-provided scheme string (from TOML or direct construction).

    Returns
    -------
    str
        Canonical scheme name (`'1s'` or `'2s'`).

    Raises
    ------
    ValueError
        If `alias` is not a recognised scheme name.
    """
    canonical = _SCHEME_ALIASES.get(alias.lower())
    if canonical is None:
        known = ', '.join(sorted(_SCHEME_ALIASES))
        raise ValueError(
            f"unknown XTRG scheme {alias!r}; recognised values are: {known}"
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
        (only meaningful for 2-site, where SVD truncation controls growth).
    trunc_thresh:
        Singular value truncation threshold (relative to the largest singular
        value per charge sector).
    n_sweeps:
        Number of full variational sweeps (forward + backward) per squaring
        step. More sweeps improve compression accuracy at the cost of compute.
    env_cache_dir:
        Directory for environment disk caching. `None` keeps all blocks in
        memory (default). Useful for large chains where environments do not
        fit in RAM.
    env_async_io:
        If `True` (default), disk writes are submitted asynchronously so
        they overlap with computation. Has no effect when `env_cache_dir`
        is `None`.
    env_window:
        Number of environment blocks to keep in memory at once when disk
        caching is enabled.
    checkpoint_dir:
        Directory for per-step checkpoints of the density matrix. `None`
        disables checkpointing.
    """

    scheme: str = '2s'
    tau_0: float = 2 ** -12
    n_steps: int = 20
    taylor_order: int = 10
    max_bond: Optional[int] = None
    trunc_thresh: float = 1e-15
    n_sweeps: int = 4
    env_cache_dir: Optional[str] = None
    env_async_io: bool = True
    env_window: int = 2
    checkpoint_dir: Optional[str] = None

    def __post_init__(self) -> None:
        self.scheme = _resolve_scheme(self.scheme)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class Summary(AlgorithmSummary):
    """XTRG output summary.

    Parameters
    ----------
    rho:
        Final density matrix ρ(β_max) as a `NormalMPO`.
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
        only; `0.0` entries for 1-site).
    converged:
        Always `True` for XTRG (fixed number of cooling steps).
    n_steps:
        Number of cooling steps actually performed.
    """

    rho: NormalMPO
    betas: List[float] = field(default_factory=list)
    log_z: List[float] = field(default_factory=list)
    free_energies: List[float] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    specific_heats: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    discarded_weights: List[float] = field(default_factory=list)
    converged: bool = True
    n_steps: int = 0

    def serialize(self) -> Dict:
        """Serialize the summary to a plain dict compatible with `torch.save`.

        The density matrix is serialized via `Network.serialize` (tensors)
        plus the separately stored `log_scale` magnitude.

        Returns
        -------
        Dict
            Serialized summary.
        """
        rho_data = self.rho.serialize()
        return {
            'version': 1,
            'rho': rho_data,
            'rho_log_scale': self.rho.log_scale,
            'betas': self.betas,
            'log_z': self.log_z,
            'free_energies': self.free_energies,
            'energies': self.energies,
            'specific_heats': self.specific_heats,
            'entropies': self.entropies,
            'discarded_weights': self.discarded_weights,
            'converged': self.converged,
            'n_steps': self.n_steps,
        }

    @classmethod
    def deserialize(cls, data: Dict, device: str = 'cpu') -> 'Summary':
        """Reconstruct a `Summary` from a dict produced by `serialize`.

        Parameters
        ----------
        data:
            Dict previously returned by `serialize`.
        device:
            Device to place all tensor blocks on. Defaults to `'cpu'`.

        Returns
        -------
        Summary
            Reconstructed summary with the density matrix on `device`.

        Raises
        ------
        ValueError
            If `data["version"]` is not `1`.
        """
        version = data.get('version', 1)
        if version != 1:
            raise ValueError(f"Unsupported Summary serialization version: {version!r}")

        rho_data = data['rho']
        tensors = [_deserialize_tensor(t) for t in rho_data['tensors']]
        rho = NormalMPO(
            tensors,
            log_scale=float(data['rho_log_scale']),
            bc=rho_data['bc'],
            center=rho_data['center'],
        )
        return cls(
            rho=rho,
            betas=list(data['betas']),
            log_z=list(data['log_z']),
            free_energies=list(data['free_energies']),
            energies=list(data['energies']),
            specific_heats=list(data['specific_heats']),
            entropies=list(data['entropies']),
            discarded_weights=list(data.get('discarded_weights', [])),
            converged=bool(data.get('converged', True)),
            n_steps=int(data.get('n_steps', 0)),
        )


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _save_checkpoint(
    rho: NormalMPO,
    betas: List[float],
    log_z: List[float],
    discarded_weights: List[float],
    step: int,
    ckpt_dir: Path,
) -> None:
    """Write an atomic checkpoint of the current XTRG state.

    Serialises the current density matrix and thermodynamic history to
    `xtrg_lock.ckpt` in `ckpt_dir`, then renames it to `xtrg.ckpt`. The
    rename is atomic on POSIX systems.

    Parameters
    ----------
    rho:
        Current density matrix.
    betas:
        β grid up to and including the current step.
    log_z:
        log Z history up to and including the current step.
    discarded_weights:
        Discarded-weight history up to and including the current step.
    step:
        Number of cooling steps completed so far.
    ckpt_dir:
        Directory in which `xtrg.ckpt` and `xtrg_lock.ckpt` are written.
    """
    import torch

    rho_data = rho.serialize()
    payload = {
        'rho': rho_data,
        'rho_log_scale': rho.log_scale,
        'betas': list(betas),
        'log_z': list(log_z),
        'discarded_weights': list(discarded_weights),
        'step': step,
    }
    lock_path = ckpt_dir / 'xtrg_lock.ckpt'
    ckpt_path = ckpt_dir / 'xtrg.ckpt'
    torch.save(payload, lock_path)
    lock_path.replace(ckpt_path)


# ---------------------------------------------------------------------------
# Inner variational compression
# ---------------------------------------------------------------------------

def _fit_mpo(
    mpo_a: NormalMPO,
    mpo_b: NormalMPO,
    opts: Options,
) -> Tuple[NormalMPO, float]:
    """Compress mpo_a @ mpo_b variationaly into a lower-bond-dim NormalMPO.

    Runs `opts.n_sweeps` full variational sweeps (each = forward + backward
    half-sweep) to find C ≈ mpo_a · mpo_b by minimising ‖C − A·B‖²_F.

    Parameters
    ----------
    mpo_a:
        Left factor MPO.
    mpo_b:
        Right factor MPO.
    opts:
        XTRG options (scheme, max_bond, trunc_thresh, n_sweeps, env_*).

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
    _cache: Optional[Path] = Path(opts.env_cache_dir) if opts.env_cache_dir else None
    if _cache is not None:
        (_cache / 'xtrg_left').mkdir(parents=True, exist_ok=True)
        (_cache / 'xtrg_right').mkdir(parents=True, exist_ok=True)

    env_left = Environment(
        L,
        _cache / 'xtrg_left' if _cache is not None else None,
        async_io=opts.env_async_io,
        window=opts.env_window,
        fetch_lo=0,
        fetch_hi=L - 2 if _2s else L - 1,
    )
    env_right = Environment(
        L,
        _cache / 'xtrg_right' if _cache is not None else None,
        async_io=opts.env_async_io,
        window=opts.env_window,
        fetch_lo=1 if _2s else 0,
        fetch_hi=L - 1,
    )

    # Build initial left boundary and all right environment blocks.
    env_left[0] = left_env_boundary(mpo_a, mpo_b, mpo_c)
    build_right_envs(mpo_a, mpo_b, mpo_c, env_right)

    dw = 0.0
    try:
        for sweep_idx in range(opts.n_sweeps):
            logger.debug("  compression sweep %d / %d", sweep_idx + 1, opts.n_sweeps)
            forward_sweep(mpo_a, mpo_b, mpo_c, env_left, env_right, opts)
            dw = backward_sweep(mpo_a, mpo_b, mpo_c, env_left, env_right, opts)
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
    betas: List[float],
    log_z: List[float],
    L: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Derive thermodynamic observables from the log Z grid.

    Uses log-β finite differences to compute internal energy and specific heat,
    which give uniform O((ln 2)²) discretisation error across the exponentially
    spaced β grid.

    Parameters
    ----------
    betas:
        List of inverse temperatures [τ₀, 2τ₀, …].
    log_z:
        `log Z(β_n)` at each point, same length as `betas`.
    L:
        Chain length (for per-site normalisation).

    Returns
    -------
    List[float]
        Free energies f(β) per site.
    List[float]
        Internal energies u(β) per site.
    List[float]
        Specific heats c_V(β) per site.
    List[float]
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

    # Specific heat: c_V[n] = β_n ∂u/∂(ln β) ≈ β_n Δu / Δ(ln β)
    specific_heats = []
    for n in range(N):
        if n < N - 1:
            cv = betas[n] * (energies[n + 1] - energies[n]) / ln2
        else:
            cv = betas[n] * (energies[n] - energies[n - 1]) / ln2
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
) -> Summary:
    """Run XTRG to compute finite-temperature properties of a Hamiltonian MPO.

    Initialises the thermal density matrix via a Taylor expansion
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
        Final density matrix, β grid, log Z history, and derived
        thermodynamic observables.

    Raises
    ------
    ValueError
        If `opts.scheme` is not a recognised scheme.
    NotImplementedError
        If `opts.scheme` is recognised but not yet implemented.
    """
    if opts is None:
        opts = Options()

    if opts.scheme not in _IMPLEMENTED_SCHEMES:
        raise NotImplementedError(
            f"XTRG scheme {opts.scheme!r} is recognised but not yet implemented; "
            f"implemented schemes are: {', '.join(sorted(_IMPLEMENTED_SCHEMES))}"
        )

    L = H.L
    _ckpt: Optional[Path] = (
        Path(opts.checkpoint_dir) if opts.checkpoint_dir is not None else None
    )
    if _ckpt is not None:
        _ckpt.mkdir(parents=True, exist_ok=True)

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
    logger.info("  taylor order      : %d", opts.taylor_order)
    logger.info("")

    # Step 0: initialise ρ(τ₀) via Taylor expansion.
    logger.info("Initializing ρ(τ₀=%.6g) via Taylor expansion (order %d)…",
                opts.tau_0, opts.taylor_order)
    rho = thermal_mpo(H, opts.tau_0, opts.taylor_order, spc)

    betas: List[float] = [opts.tau_0]
    # log_trace() (rather than log(rho.trace())) keeps log_z finite even
    # when Z(beta) itself would overflow float64 deep into the cooling run.
    log_abs_z, sign_z = rho.log_trace()
    _ensure_positive_trace(sign_z, betas[-1])
    log_z: List[float] = [log_abs_z]
    discarded_weights: List[float] = []

    logger.info("  β = %.6g,  log Z = %+.8g", betas[-1], log_z[-1])

    w = len(str(opts.n_steps))
    for step in range(opts.n_steps):
        logger.info("step %*d / %d: squaring ρ(β=%.6g) → ρ(β=%.6g)",
                    w, step + 1, opts.n_steps, betas[-1], betas[-1] * 2)

        rho, dw = _fit_mpo(rho, rho, opts)
        discarded_weights.append(dw)
        betas.append(betas[-1] * 2)
        lz, sign_z = rho.log_trace()
        _ensure_positive_trace(sign_z, betas[-1])
        log_z.append(lz)

        logger.info("  β = %.6g,  log Z = %+.8g,  dw = %.4e",
                    betas[-1], lz, dw)

        if _ckpt is not None:
            _save_checkpoint(rho, betas, log_z, discarded_weights, step + 1, _ckpt)

    logger.info("")

    free_energies, energies, specific_heats, entropies = _compute_observables(
        betas, log_z, L
    )

    return Summary(
        rho=rho,
        betas=betas,
        log_z=log_z,
        free_energies=free_energies,
        energies=energies,
        specific_heats=specific_heats,
        entropies=entropies,
        discarded_weights=discarded_weights,
        converged=True,
        n_steps=opts.n_steps,
    )
