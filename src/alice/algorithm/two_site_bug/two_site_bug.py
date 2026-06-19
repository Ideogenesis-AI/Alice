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


"""Top-level BUG driver: options, summary, and entry-point function.

The gate-based BUG (Basis-Update & Galerkin) integrator evolves an `MPS` under a
nearest-neighbour Hamiltonian by applying two-site bond gates in symmetric
(Strang) or first-order (Lie) Trotter half-sweeps, splitting each two-site block
with a truncated SVD that adapts the bond dimension. Bond Hamiltonians are reused
directly from the AutoMPO interaction list, so any nearest-neighbour model and
symmetry that `build_interaction` supports works unchanged.

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
from .gate import build_bond_generators, exp_bond_gate, to_complex
from .scheme import parity_sweep

logger = logging.getLogger(__name__)


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
    """BUG run options.

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
          (forward + backward half-sweep with half-step gates).
        - `'lie'` / `'first'` / `'1'`: first-order step (one half-sweep,
          alternating direction each step).
    max_bond:
        Maximum bond dimension kept at each SVD split. `None` means no limit.
    trunc_thresh:
        SVD truncation threshold forwarded to `decomp` at each split.
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
    imaginary_time: bool = False
    normalize: bool = True

    def __post_init__(self) -> None:
        self.order = _resolve_order(self.order)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class Summary(AlgorithmSummary):
    """BUG output.

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
        bonds of each step (length `n_steps`). This is the basis-augmentation
        size the BUG step works in before the truncated split; comparing it with
        `max_bond_dims` shows how much rank growth the truncation discards.
    """

    state: MPS
    n_steps: int = 0
    times: List[float] = field(default_factory=list)
    norms: List[float] = field(default_factory=list)
    bond_dims: List[int] = field(default_factory=list)
    max_bond_dims: List[int] = field(default_factory=list)
    aug_dims: List[int] = field(default_factory=list)

    def serialize(self) -> Dict:
        """Serialize the summary to a plain dict compatible with `torch.save`.

        Returns
        -------
        Dict
            Serialized summary with keys `"version"`, `"n_steps"`, `"times"`,
            `"norms"`, `"bond_dims"`, `"max_bond_dims"`, `"aug_dims"`, and
            `"state"`.
        """
        return {
            'version': 1,
            'n_steps': self.n_steps,
            'times': self.times,
            'norms': self.norms,
            'bond_dims': self.bond_dims,
            'max_bond_dims': self.max_bond_dims,
            'aug_dims': self.aug_dims,
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
        )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(mps: MPS, interactions: List[Interaction], opts: Optional[Options] = None) -> Summary:
    """Evolve an MPS under a nearest-neighbour Hamiltonian with the BUG integrator.

    Builds the per-bond gates once from the AutoMPO interaction list, then applies
    `opts.n_steps` Trotter steps. The state is canonicalised to `center = 0`
    before the first step and returned with `center = 0`.

    Parameters
    ----------
    mps:
        Initial MPS state. Canonicalised in-place to `center = 0` first.
    interactions:
        Interaction list from `build_interaction`. Every active term must be a
        nearest-neighbour `Interaction2Site` (see
        :func:`alice.algorithm.two_site_bug.gate.build_bond_generators`).
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
        raise ValueError(f"BUG evolution requires at least 2 sites, got L={mps.L}")

    trunc: Optional[dict] = {'thresh': opts.trunc_thresh}
    if opts.max_bond is not None:
        trunc['nkeep'] = opts.max_bond

    # Real-time evolution uses exp(-i dt H); imaginary time uses exp(-dt H).
    step_coeff: complex = -opts.dt if opts.imaginary_time else -1j * opts.dt

    generators = build_bond_generators(interactions, mps.L)
    gates_full = [None if h is None else exp_bond_gate(h, step_coeff) for h in generators]
    gates_half = [None if h is None else exp_bond_gate(h, 0.5 * step_coeff) for h in generators]

    # The gates are complex (matrix exponential); promote the state so every
    # contraction shares the complex128 dtype of the PyTorch backend.
    for site in range(mps.L):
        mps[site] = to_complex(mps[site])

    # Bring the MPS into right-canonical form with the center at site 0.
    mps.canonical(0)

    times: List[float] = []
    norms: List[float] = []
    max_bond_dims: List[int] = []
    aug_dims: List[int] = []

    n_active = sum(1 for h in generators if h is not None)
    logger.info("─" * 60)
    logger.info("Commencing: BUG Time Evolution".center(60))
    logger.info("─" * 60)
    logger.info("")
    logger.info("  order             : %s", opts.order)
    logger.info("  chain length      : %d", mps.L)
    logger.info("  active bonds      : %d / %d", n_active, mps.L - 1)
    logger.info("  time step         : %g", opts.dt)
    logger.info("  steps             : %d", opts.n_steps)
    logger.info("  evolution         : %s", "imaginary" if opts.imaginary_time else "real")
    logger.info("  max bond dim      : %s", opts.max_bond if opts.max_bond is not None else 'unlimited')
    logger.info("  trunc thresh      : %.2e", opts.trunc_thresh)
    logger.info("")

    w = len(str(opts.n_steps))
    for step in range(opts.n_steps):
        if opts.order == 'strang':
            # Symmetric Strang step: U_odd(dt/2) · U_even(dt) · U_odd(dt/2).
            augmented = max(
                parity_sweep(mps, gates_half, 'odd', trunc),
                parity_sweep(mps, gates_full, 'even', trunc),
                parity_sweep(mps, gates_half, 'odd', trunc),
            )
        else:
            # First-order Lie step: U_odd(dt) · U_even(dt).
            augmented = max(
                parity_sweep(mps, gates_full, 'odd', trunc),
                parity_sweep(mps, gates_full, 'even', trunc),
            )

        norm = mps.norm()
        if opts.normalize:
            mps.normalize()

        times.append((step + 1) * opts.dt)
        norms.append(norm)
        max_bond_dims.append(max(mps.bond_dims) if mps.bond_dims else 1)
        aug_dims.append(augmented)

        logger.info(
            "step %*d / %d: t = %g, norm = %.10f, kept bond = %d, augmented = %d",
            w, step + 1, opts.n_steps, times[-1], norm, max_bond_dims[-1], augmented,
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
    )
