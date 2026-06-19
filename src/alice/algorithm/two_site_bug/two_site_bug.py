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


"""Top-level two-site BUG driver: options, summary, and entry-point function.

The faithful Basis-Update & Galerkin (BUG) integrator (Ceruti, Kusch & Lubich,
arXiv:2304.05660) evolves an `MPS` under a nearest-neighbour Hamiltonian by
odd/even Trotter sweeps of *local* two-site updates. Each bond update is the
rank-adaptive K/L/S step: it augments the left frame from the evolved K factor,
augments the right frame from the evolved L factor, evolves the small core S in
the augmented bases (Galerkin), and truncates with an SVD. The local substeps
exponentiate the *projected* effective Hamiltonian internally (Krylov `expv`) —
no pre-formed gate is applied — so the step is the faithful KLS update, exact at
full rank. Bond Hamiltonians are reused directly from the AutoMPO interaction
list, so any nearest-neighbour model and symmetry that `build_interaction`
supports works unchanged.

Typical usage:

    from alice import build_interaction, init_mps
    from alice.algorithm import two_site_bug

    interactions, spc, geo = build_interaction(cfg)
    mps = init_mps(geo.L, spc, Op, config=[0, 1] * (geo.L // 2), target_qn=0)
    opts = two_site_bug.Options(dt=0.05, n_steps=20, order='strang', max_bond=64)
    summary = two_site_bug.run(mps, interactions, opts)
    print(summary.bond_dims)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from alice.network import MPS
from alice.network.interaction import Interaction
from alice.network.network import Network

from ..interface import AlgorithmOptions, AlgorithmSummary
from ._kernel import with_expv_backend, with_time_prefactor
from .bond import build_bond_generators, kernel_gate, to_complex
from .scheme import parity_sweep

logger = logging.getLogger(__name__)

# Sentinel bond cap used when `Options.max_bond is None` (keep every singular
# value at the post-S-step SVD, i.e. unlimited growth up to the local capacity).
_UNLIMITED_BOND = 1 << 30


# ---------------------------------------------------------------------------
# Order alias resolution
# ---------------------------------------------------------------------------

_ORDER_ALIASES: Dict[str, str] = {
    'lie': 'lie',
    'first': 'lie',
    '1': 'lie',
    'strang': 'strang',
    'second': 'strang',
    '2': 'strang',
}


def _resolve_order(alias: str) -> str:
    """Normalise a Trotter-order alias to its canonical name.

    Parameters
    ----------
    alias:
        User-provided order string.

    Returns
    -------
    str
        Canonical order name (`'lie'` or `'strang'`).

    Raises
    ------
    ValueError
        If `alias` is not a recognised order name.
    """
    canonical = _ORDER_ALIASES.get(alias.lower())
    if canonical is None:
        known = ', '.join(sorted(_ORDER_ALIASES))
        raise ValueError(f"unknown Trotter order {alias!r}; recognised values are: {known}")
    return canonical


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class Options(AlgorithmOptions):
    """Two-site BUG run options.

    All fields have sensible defaults so `Options()` is a valid minimal
    configuration. Use `Options.from_toml` to load from an `[algorithm]` TOML
    section, or `Options.load_toml` to read directly from a file.

    Parameters
    ----------
    dt:
        Time step. Interpreted as real time (evolution operator `exp(-i dt H)`)
        unless `imaginary_time` is set.
    n_steps:
        Number of time steps to perform.
    order:
        Trotter order. Canonical values and their aliases:

        - `'strang'` / `'second'` / `'2'`: symmetric second-order step
          `U_even(dt/2) · U_odd(dt) · U_even(dt/2)`.
        - `'lie'` / `'first'` / `'1'`: first-order step `U_even(dt) · U_odd(dt)`.
    max_bond:
        Maximum bond dimension kept by the post-S-step SVD truncation. `None`
        means no explicit cap (rank adapts up to the local capacity).
    trunc_thresh:
        Singular-value threshold of the post-S-step SVD. Each bond keeps only the
        directions whose weight exceeds it, so the rank grows only as far as the
        state's entanglement requires — the discarded-weight control of the
        rank adaptation.
    augment:
        If `True` (default), the local KLS update may grow the bond basis from
        the evolved K/L directions. If `False`, the bond dimension is held fixed
        (parallel basis update without rank adaptation).
    aug_krylov_depth:
        Number of K/L Krylov directions stacked before the augmented basis is
        extracted (`1` is the standard rank-adaptive BUG).
    lanczos_tol:
        Termination tolerance of the local Lanczos `expv` solves.
    lanczos_maxiter:
        Maximum Lanczos iterations per local substep.
    imaginary_time:
        If `True`, evolve with `exp(-dt H)` (imaginary time) instead of
        `exp(-i dt H)`. Combined with `normalize`, this cools the state toward
        the ground state.
    normalize:
        If `True` (default), renormalise the state after every step. Required
        for imaginary-time evolution; harmless for real time (it only removes
        the small norm leakage from truncation).
    """

    dt: float = 0.05
    n_steps: int = 10
    order: str = 'strang'
    max_bond: Optional[int] = None
    trunc_thresh: float = 1e-12
    augment: bool = True
    aug_krylov_depth: int = 1
    lanczos_tol: float = 1e-15
    lanczos_maxiter: int = 30
    imaginary_time: bool = False
    normalize: bool = True

    def __post_init__(self) -> None:
        self.order = _resolve_order(self.order)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class Summary(AlgorithmSummary):
    """Two-site BUG output.

    Attributes
    ----------
    state:
        Evolved MPS after all steps (orthogonality center at site 0).
    n_steps:
        Number of steps performed.
    times:
        Cumulative evolution time recorded after each step (length `n_steps`).
    norms:
        State norm measured after each step *before* any renormalisation
        (length `n_steps`). For real time these stay near 1; for imaginary time
        they decay.
    bond_dims:
        Bond dimensions of `state` after the final step (length `L - 1`).
    max_bond_dims:
        Maximum *kept* bond dimension after each step (length `n_steps`).
    aug_dims:
        Maximum *proposed* (pre-truncation) augmented bond dimension over the
        bonds of each step (length `n_steps`). This is the rank the K/L
        augmentation reaches before the truncated S-step split; comparing it
        with `max_bond_dims` shows how much rank growth the truncation discards.
    disc_weights:
        Maximum relative discarded weight over the bonds of each step (length
        `n_steps`) — the fraction of bond weight the `trunc_thresh` S-step SVD
        throws away. Near zero means the kept rank captures the state faithfully.
    """

    state: MPS
    n_steps: int = 0
    times: List[float] = field(default_factory=list)
    norms: List[float] = field(default_factory=list)
    bond_dims: List[int] = field(default_factory=list)
    max_bond_dims: List[int] = field(default_factory=list)
    aug_dims: List[int] = field(default_factory=list)
    disc_weights: List[float] = field(default_factory=list)

    def serialize(self) -> Dict:
        """Serialize the summary to a plain dict compatible with `torch.save`.

        Returns
        -------
        Dict
            Serialized summary with keys `"version"`, `"n_steps"`, `"times"`,
            `"norms"`, `"bond_dims"`, `"max_bond_dims"`, `"aug_dims"`,
            `"disc_weights"`, and `"state"`.
        """
        return {
            'version': 1,
            'n_steps': self.n_steps,
            'times': self.times,
            'norms': self.norms,
            'bond_dims': self.bond_dims,
            'max_bond_dims': self.max_bond_dims,
            'aug_dims': self.aug_dims,
            'disc_weights': self.disc_weights,
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
            state=Network.deserialize(data['state'], device=device),
            n_steps=data['n_steps'],
            times=data['times'],
            norms=data['norms'],
            bond_dims=data['bond_dims'],
            max_bond_dims=data['max_bond_dims'],
            aug_dims=data.get('aug_dims', []),
            disc_weights=data.get('disc_weights', []),
        )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(mps: MPS, interactions: List[Interaction], opts: Optional[Options] = None) -> Summary:
    """Evolve an MPS under a nearest-neighbour Hamiltonian with the two-site BUG integrator.

    Builds the per-bond Hamiltonian terms once from the AutoMPO interaction list,
    then applies `opts.n_steps` odd/even Trotter steps of the faithful K/L/S local
    update. The state is canonicalised to `center = 0` before the first step and
    returned with `center = 0`.

    Parameters
    ----------
    mps:
        Initial MPS state. Promoted to `complex128` and canonicalised in-place to
        `center = 0` first.
    interactions:
        Interaction list from `build_interaction`. Every active term must be a
        nearest-neighbour `Interaction2Site` (see
        :func:`alice.algorithm.two_site_bug.bond.build_bond_generators`).
    opts:
        Run options. Defaults to `Options()` if `None`.

    Returns
    -------
    Summary
        Evolved state, time/norm/bond-dimension history, and step count.

    Raises
    ------
    ValueError
        If `mps` has fewer than two sites.
    """
    if opts is None:
        opts = Options()
    if mps.L < 2:
        raise ValueError(f"two-site BUG evolution requires at least 2 sites, got L={mps.L}")

    maxdim = opts.max_bond if opts.max_bond is not None else _UNLIMITED_BOND
    # Real-time evolution uses exp(-i dt H); imaginary time uses exp(-dt H). The
    # kernel multiplies its local timestep by this prefactor internally.
    prefactor: complex = -1.0 if opts.imaginary_time else -1j

    # Promote the state to complex128 so every local exponential shares the
    # PyTorch backend dtype, then bring the center to site 0.
    for site in range(mps.L):
        mps[site] = to_complex(mps[site])
    mps.canonical(0)

    # Bare per-bond Hamiltonian terms, relabelled into the local-KLS kernel's
    # gate convention against the MPS physical itags. Built once and reused for
    # every sweep (the kernel exponentiates the projected term per substep).
    generators = build_bond_generators(interactions, mps.L)
    gates = [
        None if h is None else kernel_gate(h, mps[b].itags[2], mps[b + 1].itags[2])
        for b, h in enumerate(generators)
    ]

    def sweep(parity: str, tau: float):
        return parity_sweep(
            mps, gates, parity, tau, maxdim,
            opts.augment, opts.aug_krylov_depth, opts.trunc_thresh,
            opts.lanczos_tol, opts.lanczos_maxiter,
        )

    times: List[float] = []
    norms: List[float] = []
    max_bond_dims: List[int] = []
    aug_dims: List[int] = []
    disc_weights: List[float] = []

    n_active = sum(1 for h in generators if h is not None)
    logger.info("─" * 60)
    logger.info("Commencing: Two-Site BUG Time Evolution".center(60))
    logger.info("─" * 60)
    logger.info("")
    logger.info("  order             : %s", opts.order)
    logger.info("  chain length      : %d", mps.L)
    logger.info("  active bonds      : %d / %d", n_active, mps.L - 1)
    logger.info("  time step         : %g", opts.dt)
    logger.info("  steps             : %d", opts.n_steps)
    logger.info("  evolution         : %s", "imaginary" if opts.imaginary_time else "real")
    logger.info("  max bond dim      : %s", opts.max_bond if opts.max_bond is not None else 'unlimited')
    logger.info("  augment           : %s", opts.augment)
    logger.info("")

    w = len(str(opts.n_steps))
    with with_time_prefactor(prefactor), with_expv_backend('native_hermitian_lanczos'):
        for step in range(opts.n_steps):
            if opts.order == 'strang':
                # Symmetric Strang step: U_even(dt/2) · U_odd(dt) · U_even(dt/2).
                results = [
                    sweep('even', 0.5 * opts.dt),
                    sweep('odd', opts.dt),
                    sweep('even', 0.5 * opts.dt),
                ]
            else:
                # First-order Lie step: U_even(dt) · U_odd(dt).
                results = [
                    sweep('even', opts.dt),
                    sweep('odd', opts.dt),
                ]
            augmented = max(aug for aug, _ in results)
            discarded = max(disc for _, disc in results)

            norm = mps.norm()
            if opts.normalize:
                mps.normalize()

            times.append((step + 1) * opts.dt)
            norms.append(norm)
            max_bond_dims.append(max(mps.bond_dims) if mps.bond_dims else 1)
            aug_dims.append(augmented)
            disc_weights.append(discarded)

            logger.info(
                "step %*d / %d: t = %g, norm = %.10f, kept bond = %d, augmented = %d, disc = %.2e",
                w, step + 1, opts.n_steps, times[-1], norm, max_bond_dims[-1], augmented, discarded,
            )

    # Ensure the returned state has the center at site 0 for a well-defined norm.
    if mps.center != 0:
        mps.canonical(0)

    logger.info("")

    return Summary(
        state=mps,
        n_steps=opts.n_steps,
        times=times,
        norms=norms,
        bond_dims=list(mps.bond_dims),
        max_bond_dims=max_bond_dims,
        aug_dims=aug_dims,
        disc_weights=disc_weights,
    )
