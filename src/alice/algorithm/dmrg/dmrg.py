# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Alice library.
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

    interactions, spc, L = build_interaction(cfg["heisenberg_u1"])
    mpo = build_hamiltonian(interactions, L, spc)
    mps = MPS(...)           # provide an initial state
    opts = dmrg.Options.from_toml(cfg["heisenberg_u1"]["algorithm"])
    summary = dmrg.run(mps, mpo, opts)
    print(summary.energy)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
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

        - `'1s'` / `'1-site'` / `'one-site'`: 1-site DMRG (implemented).
        - `'2s'` / `'2-site'` / `'two-site'`: 2-site DMRG (implemented).
        - `'1sp'` / `'1-site-plus'` / `'one-site-plus'`: 1-site-plus (not yet implemented).
    n_sweeps:
        Maximum number of full sweeps (one right + one left half-sweep each).
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
    """

    scheme: str = '1s'
    n_sweeps: int = 10
    max_bond: Optional[int] = None
    trunc_thresh: float = 1e-15
    davidson_tol: float = 1e-10
    davidson_max_iter: int = 100
    davidson_max_subspace: int = 20
    e_tol: float = 1e-8

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
        Energy recorded at the end of each full sweep (right + left half-sweep).
    converged:
        `True` if `|E_new - E_old| < opts.e_tol` before `n_sweeps` was reached.
    n_sweeps:
        Actual number of full sweeps performed.
    bond_dims:
        Bond dimensions of `state` after convergence (length `L - 1`).
    """

    energy: float
    state: MPS
    energies: List[float] = field(default_factory=list)
    converged: bool = False
    n_sweeps: int = 0
    bond_dims: List[int] = field(default_factory=list)

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
            state=Network.deserialize(data['state'], device=device),
        )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(mps: MPS, mpo: MPO, opts: Optional[Options] = None) -> Summary:
    """Run DMRG to find the ground state of a Hamiltonian MPO.

    Performs alternating right and left half-sweeps, optimising each site
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
            f"DMRG scheme {opts.scheme!r} is not yet implemented; "
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
    env_left = Environment(L)
    env_right = Environment(L)

    # Initialise the left boundary and all right environment blocks.
    env_left[0] = left_env_boundary(mps, mpo)
    build_right_envs(mps, mpo, env_right)

    energies: List[float] = []
    converged = False
    sweep_count = 0
    prev_energy = math.inf

    # Log startup header and options before the first sweep.
    max_bond_str = str(opts.max_bond) if opts.max_bond is not None else 'unlimited'
    logger.info("=" * 60)
    logger.info("Commencing: DMRG Algorithm".center(60))
    logger.info("=" * 60)
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
    logger.info("")

    w = len(str(opts.n_sweeps))
    for sweep_idx in range(opts.n_sweeps):
        # Blank debug line between sweeps for visual separation in the log file.
        if sweep_idx > 0:
            logger.debug("")
        logger.debug("sweep %*d / %d: forward sweep initiated", w, sweep_idx + 1, opts.n_sweeps)

        # Right half-sweep: center moves from 0 to L-1.
        right_energy = forward_sweep(mps, mpo, env_left, env_right, opts)

        logger.debug("sweep %*d / %d: forward sweep finished", w, sweep_idx + 1, opts.n_sweeps)
        logger.debug("  local E = %+.12g", right_energy)
        logger.debug("")
        logger.debug("sweep %*d / %d: backward sweep initiated", w, sweep_idx + 1, opts.n_sweeps)

        # Left half-sweep: center moves from L-1 to 0; energy recorded here.
        energy = backward_sweep(mps, mpo, env_left, env_right, opts)

        delta_e = abs(energy - prev_energy)
        energies.append(energy)
        sweep_count += 1

        logger.debug("sweep %*d / %d: backward sweep finished", w, sweep_idx + 1, opts.n_sweeps)
        logger.debug("  local E = %+.12g", energy)
        logger.debug("  ΔE = %+.4e", energy - prev_energy)
        logger.info("sweep %*d / %d: E = %+.12g, |ΔE| = %.4e", w, sweep_idx + 1, opts.n_sweeps, energy, delta_e)

        # Check energy convergence.
        if delta_e < opts.e_tol:
            converged = True
            logger.info("converged after %d sweep(s)", sweep_count)
            break
        prev_energy = energy

    logger.info("")

    return Summary(
        energy=energies[-1] if energies else math.nan,
        state=mps,
        energies=energies,
        converged=converged,
        n_sweeps=sweep_count,
        bond_dims=list(mps.bond_dims),
    )
