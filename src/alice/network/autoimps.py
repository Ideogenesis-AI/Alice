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


"""Universal iMPS initializer for bosonic, fermionic, and conductor systems.

`init_imps` constructs the unit cell of an infinite MPS from a physical space
`(Spc, Op)` returned by `load_space`, in the same two modes as `init_mps`:

- `bond_dim=1` — deterministic product state. Every bond carries a single
  sector, fixed by the site configuration. Best used with `scheme='2s'`,
  which grows the bond dimension from the first iteration.
- `bond_dim>1` — random cell with group-derived bond sectors, ready for
  either scheme.

What the tiling demands
-----------------------
A unit cell describes an infinite chain by repeating, so bond `N` is bond
`0`: the bond charge has to return to where it started after `N` sites.
Two things follow, and they are the whole difference from `init_mps`.

**The cell is charge neutral.** Fusing the physical charges of the cell has
to give back the charge it started from. In Alice's particle-hole symmetric
conventions that means half filling for fermions and zero magnetization for
spins, and it is a property of the cell rather than a choice: a doped or
magnetized uniform state would need the bond charges to come back shifted,
which is not a cell that tiles. A cell length that admits no neutral
configuration at all, an odd one for spin-1/2, is refused rather than built.
There is no `target_qn` here for the same reason; `config` is what chooses
the pattern within the cell.

**Each bond carries only the charges reachable at its own position.** Bond
`k` sits after `k` physical sites, so it can only hold charges an even
number of fusion steps away from the ones its neighbors hold. Giving every
bond the same sectors instead is what produces a transfer operator with two
dominant eigenvalues of opposite sign, which `iMPS.canonical` reports as a
two-cycle: the cell maps the two classes of bond charge into each other
rather than onto themselves.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch
from nicole import Direction, Tensor
from nicole.index import Index, Sector

from .infinite import iMPS

logger = logging.getLogger(__name__)

# How many random cells to draw before giving up, and how far apart their
# seeds sit. The stride only has to exceed any unit cell length.
_DRAWS = 8
_SEED_STRIDE = 1000


# ---------------------------------------------------------------------------
# Charge helpers
# ---------------------------------------------------------------------------

def _next_charge(group, q_bond, q_phys):
    """Advance a bond charge by one physical site.

    Abelian groups fuse to a single charge. For SU(2), and products
    containing it, the minimum branch is taken, which is the dimer rule the
    finite initializer uses.
    """
    if hasattr(group, 'fuse_channels'):
        return min(group.fuse_channels(q_bond, q_phys))
    return group.fuse_unique(q_bond, q_phys)


def _fuse_channels(group, q_bond, q_phys):
    """Every charge one physical site can lead to, as a tuple."""
    if hasattr(group, 'fuse_channels'):
        return tuple(group.fuse_channels(q_bond, q_phys))
    return (group.fuse_unique(q_bond, q_phys),)


def _bond_charges(group, Spc: Index, config: Sequence[int], q_start) -> List:
    """Walk a configuration from `q_start` and report every bond charge.

    Parameters
    ----------
    group:
        Nicole group object.
    Spc:
        Physical index returned by `load_space`.
    config:
        Sector index into `Spc.sectors` for each site of the cell.
    q_start:
        Charge of bond 0.

    Returns
    -------
    list
        Charges of bonds `0` to `N`, so `N + 1` entries. The cell tiles when
        the last equals the first.
    """
    charges = [q_start]
    for k in config:
        charges.append(_next_charge(group, charges[-1], Spc.sectors[k].charge))
    return charges


def _even_neighborhood(group, Spc: Index, q, steps: int = 1) -> set:
    """Charges reachable from `q` in an even number of fusion steps.

    Each level of the walk fuses twice, so every charge it returns sits in
    the same class as `q` and can appear on the same bond. Walking an odd
    number of steps instead would mix the classes the cell alternates
    between, which is what sends gauge fixing into a two-cycle.

    Parameters
    ----------
    group:
        Nicole group object.
    Spc:
        Physical index returned by `load_space`.
    q:
        Charge to walk out from.
    steps:
        Number of double steps to take.

    Returns
    -------
    set
        Charges within `steps` double steps of `q`, `q` included.
    """
    visited = {q}
    frontier = {q}
    for _ in range(steps):
        reached = set()
        for q_here in frontier:
            for first in Spc.sectors:
                for q_mid in _fuse_channels(group, q_here, first.charge):
                    for second in Spc.sectors:
                        reached.update(_fuse_channels(group, q_mid, second.charge))
        frontier = reached - visited
        visited |= reached
    return visited


# ---------------------------------------------------------------------------
# Configuration over the cell
# ---------------------------------------------------------------------------

def _closes(group, Spc: Index, config: Sequence[int], q_start) -> bool:
    """Report whether a configuration returns the bond charge to `q_start`."""
    charges = _bond_charges(group, Spc, config, q_start)
    return charges[-1] == charges[0]


def _auto_config(N: int, Spc: Index, group, q_start) -> List[int]:
    """Choose a neutral configuration for a cell of `N` sites.

    Tried in order: one sector everywhere, which covers a space with a
    neutral sector; then every two-sector alternation, which covers the
    Neel pattern a spin chain and a half-filled fermion chain both want.

    Parameters
    ----------
    N:
        Number of sites in the unit cell.
    Spc:
        Physical index returned by `load_space`.
    group:
        Nicole group object.
    q_start:
        Charge of bond 0, which the configuration has to return to.

    Returns
    -------
    list[int]
        Sector index into `Spc.sectors` for each site.

    Raises
    ------
    ValueError
        If no configuration of either shape is neutral over the cell.
    """
    n = len(Spc.sectors)
    order = sorted(range(n), key=lambda k: Spc.sectors[k].charge, reverse=True)

    for k in order:
        if _closes(group, Spc, [k] * N, q_start):
            return [k] * N

    for ki in order:
        for kj in order:
            config = [ki if i % 2 == 0 else kj for i in range(N)]
            if _closes(group, Spc, config, q_start):
                return config

    raise ValueError(
        f"init_imps: no charge-neutral configuration exists for a unit cell "
        f"of N={N} in this physical space, so no state on it tiles. Double "
        f"the cell, which makes a neutral pattern available whenever an odd "
        f"one is not, or pass an explicit config= that fuses back to the "
        f"charge it starts from."
    )


# ---------------------------------------------------------------------------
# The two modes
# ---------------------------------------------------------------------------

def _product_imps(
    N: int,
    Spc: Index,
    config: Sequence[int],
    charges: Sequence,
    dtype: torch.dtype,
) -> iMPS:
    """Build a cell of bond dimension one from a configuration.

    Parameters
    ----------
    N:
        Number of sites in the unit cell.
    Spc:
        Physical index returned by `load_space`.
    config:
        Sector index into `Spc.sectors` for each site.
    charges:
        Bond charges from `_bond_charges`.
    dtype:
        Block dtype of the site tensors.

    Returns
    -------
    iMPS
        Gauge-fixed product state.
    """
    group = Spc.group
    tensors = []
    for i in range(N):
        left = Index(
            direction=Direction.IN,
            group=group,
            sectors=(Sector(charge=charges[i], dim=1),),
        )
        right = Index(
            direction=Direction.OUT,
            group=group,
            sectors=(Sector(charge=charges[i + 1], dim=1),),
        )
        t = Tensor.zeros(
            [left, right, Spc],
            dtype=dtype,
            itags=[f'A{i % N:02d}', f'A{(i + 1) % N:02d}', f's{i:02d}'],
        )
        # Both bonds carry one sector each, so charge conservation leaves a
        # single admissible block and one component in it.
        _, block = next(iter(t.data.items()))
        block[0, 0, 0] = 1.0
        tensors.append(t)
    imps = iMPS(tensors)
    imps.canonical(0)
    return imps


def _random_imps(
    N: int,
    Spc: Index,
    charges: Sequence,
    bond_dim: int,
    seed: int,
    dtype: torch.dtype,
) -> iMPS:
    """Build a random cell whose bonds carry the charges they can reach.

    Bond `k` is given the charges an even number of fusion steps from the
    one the configuration puts there, so each bond stays in its own class
    and the cell can host a translation-invariant state.

    Parameters
    ----------
    N:
        Number of sites in the unit cell.
    Spc:
        Physical index returned by `load_space`.
    charges:
        Bond charges from `_bond_charges`.
    bond_dim:
        Target width of each bond, spread over its sectors.
    seed:
        Base random seed. Site `i` uses `seed + i`.
    dtype:
        Block dtype of the site tensors.

    Returns
    -------
    iMPS
        Gauge-fixed random state.

    Raises
    ------
    ValueError
        If every draw lands on a cell that cannot be gauge fixed.

    Notes
    -----
    Random amplitudes land often enough on a transfer operator whose two
    largest eigenvalues are equal and opposite, or too close to separate in
    the sweep budget, and neither is a state a gauge can be found for. The
    draw is repeated with a shifted seed when that happens, which is
    deterministic for a given `seed` and far friendlier than handing the
    caller a cell nothing can use.
    """
    group = Spc.group
    bonds = []
    for k in range(N):
        reachable = sorted(_even_neighborhood(group, Spc, charges[k]))
        width = max(1, bond_dim // len(reachable))
        bonds.append(Index(
            direction=Direction.IN,
            group=group,
            sectors=tuple(Sector(charge=q, dim=width) for q in reachable),
        ))

    for attempt in range(_DRAWS):
        offset = seed + attempt * _SEED_STRIDE
        imps = iMPS([
            Tensor.random(
                [bonds[i], bonds[(i + 1) % N].flip(), Spc],
                dtype=dtype,
                seed=offset + i,
                itags=[f'A{i % N:02d}', f'A{(i + 1) % N:02d}', f's{i:02d}'],
            )
            for i in range(N)
        ])
        try:
            imps.canonical(0)
        except ValueError as exc:
            logger.debug("init_imps: draw %d is not gauge fixable (%s)", attempt, exc)
            continue
        if min(imps.bond_dims) == 0:
            logger.debug("init_imps: draw %d left an empty bond", attempt)
            continue
        logger.debug("init_imps: drew a gauge-fixable cell on attempt %d", attempt)
        return imps

    raise ValueError(
        f"init_imps: none of {_DRAWS} random cells of bond_dim={bond_dim} at "
        f"N={N} could be gauge fixed. Try another seed, a different width, "
        "or bond_dim=1 with scheme='2s', which grows the width from a "
        "product state instead."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_imps(
    N: int,
    Spc: Index,
    Op: Dict[str, Tensor],
    bond_dim: int = 1,
    *,
    config: Optional[List[int]] = None,
    seed: int = 42,
    dtype: torch.dtype = torch.float64,
) -> iMPS:
    """Construct the unit cell of an initial iMPS for VUMPS.

    Works for all three particle types without a particle-type argument, as
    the `(Spc, Op)` pair from `load_space` carries the symmetry. The cell is
    returned gauge-fixed and normalized, which is what `vumps.run` expects.

    Parameters
    ----------
    N:
        Number of sites in the unit cell.
    Spc:
        Physical index returned by `load_space`.
    Op:
        Operator dictionary returned by `load_space`, read for the vacuum
        charge the walk over the cell starts from.
    bond_dim:
        Target width of each bond.

        - `bond_dim=1`: product state, one sector per bond. The one-site
          scheme can never leave it, so pair it with `scheme='2s'`.
        - `bond_dim>1`: random cell, each bond carrying the charges
          reachable at its position.
    config:
        Sector index into `Spc.sectors` for each site, or `None` to choose a
        pattern automatically. An explicit one has to be neutral over the
        cell, as the module docstring describes.
    seed:
        Base random seed used when `bond_dim > 1`. Site `i` uses `seed + i`.
    dtype:
        Block dtype of the site tensors, `torch.float64` by default.

    Returns
    -------
    iMPS
        Gauge-fixed unit cell, normalized so that every bond's Schmidt
        weights sum to one.

    Raises
    ------
    ValueError
        If `N` or `bond_dim` is below one; if `config` has the wrong length
        or does not fuse back to the charge it starts from; if no neutral
        configuration exists for this cell length; or if gauge fixing finds
        no state on the cell that was built.

    Examples
    --------
    Product cell for the spin-1/2 Heisenberg chain under U(1), ready for the
    two-site scheme to widen:

    >>> from nicole import load_space
    >>> from alice import init_imps
    >>> Spc, Op = load_space('Spin', 'U1', {'J': 0.5})
    >>> imps = init_imps(2, Spc, Op)

    Random cell of width 16 for spinless fermions at half filling:

    >>> Spc, Op = load_space('Ferm', 'U1')
    >>> imps = init_imps(2, Spc, Op, bond_dim=16)

    A four-site cell with an explicit pattern, two up then two down:

    >>> Spc, Op = load_space('Spin', 'U1', {'J': 0.5})
    >>> imps = init_imps(4, Spc, Op, config=[1, 1, 0, 0])
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if bond_dim < 1:
        raise ValueError(f"bond_dim must be >= 1, got {bond_dim}")
    if config is not None and len(config) != N:
        raise ValueError(f"config has length {len(config)}, expected {N}")

    group = Spc.group
    q_vac = Op['vac'].sectors[0].charge

    if config is None:
        config = _auto_config(N, Spc, group, q_vac)
    elif not _closes(group, Spc, config, q_vac):
        reached = _bond_charges(group, Spc, config, q_vac)[-1]
        raise ValueError(
            f"init_imps: config fuses to {reached!r} over the cell but "
            f"starts from {q_vac!r}, so the cell does not tile. A unit cell "
            "carries no net charge, which is half filling for fermions and "
            "zero magnetization for spins."
        )

    charges = _bond_charges(group, Spc, config, q_vac)
    logger.debug("init_imps: config %s, bond charges %s", list(config), charges)

    if bond_dim == 1:
        return _product_imps(N, Spc, config, charges, dtype)
    return _random_imps(N, Spc, charges, bond_dim, seed, dtype)
