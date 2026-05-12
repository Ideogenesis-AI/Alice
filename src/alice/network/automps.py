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


"""Universal MPS initializer for bosonic, fermionic, and conductor systems.

`init_mps` constructs an initial MPS for DMRG from a physical space `(Spc, Op)`
returned by `load_space`. It supports two modes:

- `bond_dim=1` — deterministic product state with exact charge targeting.
  Every bond carries a single sector determined by the site configuration.
  Best used with CBE (`scheme='1sp'`) or 2-site (`scheme='2s'`) DMRG, which
  grow the bond dimension during the first sweep.

- `bond_dim>1` — random MPS with group-derived bond sectors. Bond sectors are
  chosen via a BFS from the center-bond charge, replacing the hardcoded
  heuristics in the per-example `_random_mps` helpers.

Both modes are particle-type agnostic: the `(Spc, Op)` pair fully encodes
all symmetry information, so no separate `spin=`, `symmetry=`, or
`particle_type=` argument is required.

Charge conventions
------------------
All standard Nicole physical spaces use a particle-hole symmetric convention:
- Spin U1: Sz = ±1/2 mapped to charges ±1.
- Ferm U1: empty = -1, occupied = +1.
- Band (U1⊗U1 or U1⊗SU2): half-filled site has first U1 component = 0.

This means the center-bond charge Q_c for a balanced auto-config is always
in {0, (0,0), (0,...)} — independent of chain length L.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

from nicole import Direction, Tensor
from nicole.index import Index, Sector

from .network import MPS


# ---------------------------------------------------------------------------
# Internal charge helpers
# ---------------------------------------------------------------------------

def _charge_sort_key(q):
    """Sort key for a charge: scalar or tuple, suitable for `sorted()`."""
    return q


def _charge_dist(q1, q2) -> int:
    """L1 distance between two charges (scalar or tuple)."""
    if isinstance(q1, tuple):
        return sum(abs(a - b) for a, b in zip(q1, q2))
    return abs(q1 - q2)


def _next_charge(group, q_bond, q_phys):
    """Advance a bond charge by one physical site.

    For Abelian groups (U1, Z2) uses `fuse_unique` (deterministic). For SU2
    and ProductGroup containing SU2 uses `min(fuse_channels(...))` — the
    minimum-branch (VBS/dimer) rule.
    """
    if hasattr(group, 'fuse_channels'):
        return min(group.fuse_channels(q_bond, q_phys))
    return group.fuse_unique(q_bond, q_phys)


# ---------------------------------------------------------------------------
# Bond charge path
# ---------------------------------------------------------------------------

def _bond_charges(group, Spc: Index, config: list[int], Q_vac) -> list:
    """Compute the bond charge sequence Q_0, Q_1, …, Q_L from a site config.

    Parameters
    ----------
    group:
        Nicole group object (U1Group, SU2Group, ProductGroup, …).
    Spc:
        Physical Index returned by `load_space`.
    config:
        List of sector indices (0-based into `Spc.sectors`), length L.
    Q_vac:
        Vacuum charge (charge of `Op['vac'].sectors[0]`).

    Returns
    -------
    list
        Length-L+1 list of bond charges Q_0…Q_L.
    """
    charges = [Q_vac]
    for k in config:
        q_phys = Spc.sectors[k].charge
        charges.append(_next_charge(group, charges[-1], q_phys))
    return charges


# ---------------------------------------------------------------------------
# Reachable bond sectors for random mode
# ---------------------------------------------------------------------------

def _reachable_charges(group, Spc: Index, Q_c, d: int = 2) -> set:
    """BFS from `Q_c` over all physical sectors up to depth `d`.

    Returns the set of all charges reachable from `Q_c` within `d` fusion
    steps. Used to build the bulk bond Index for the random mode.

    The set size is fixed regardless of chain length L.

    Parameters
    ----------
    group:
        Nicole group object.
    Spc:
        Physical Index returned by `load_space`.
    Q_c:
        Starting center-bond charge.
    d:
        BFS depth (default 2).

    Returns
    -------
    set
        Set of reachable charges.
    """
    frontier: set = {Q_c}
    visited: set = {Q_c}
    for _ in range(d):
        next_frontier: set = set()
        for q in frontier:
            for sector in Spc.sectors:
                q_phys = sector.charge
                if hasattr(group, 'fuse_channels'):
                    channels = group.fuse_channels(q, q_phys)
                else:
                    channels = (group.fuse_unique(q, q_phys),)
                next_frontier.update(channels)
        frontier = next_frontier - visited
        visited |= next_frontier
    return visited


# ---------------------------------------------------------------------------
# Auto-balanced site configuration
# ---------------------------------------------------------------------------

def _auto_config(L: int, Spc: Index, group, Q_vac) -> list[int]:
    """Choose a physically balanced site configuration automatically.

    Produces a configuration (list of physical-sector indices) that:

    1. Gives alternating bond charges (dimer/VBS path for SU2).
    2. Has total charge Q_L = Q_vac (matches the right-boundary vacuum).
    3. Requires no particle-type knowledge — inferred from `Spc.sectors`.

    Strategy
    --------
    - 1 sector: all sites use sector 0 (forced).
    - 2+ sectors: tries single-sector fill first (works for SU2 and Z2 with
      even L), then period-2 alternation (works for U1 and Band spaces).
    - Checks Q_L = Q_vac by construction; tries pairs sorted high-charge first
      so that bond Q_1 is positive, matching the natural half-filling path.
    - Falls back to a greedy charge-neutrality heuristic if no exact pattern
      is found (e.g., some odd-L cases).

    Parameters
    ----------
    L:
        Chain length.
    Spc:
        Physical Index returned by `load_space`.
    group:
        Nicole group object.
    Q_vac:
        Vacuum charge (charge of `Op['vac'].sectors[0]`).

    Returns
    -------
    list[int]
        Site configuration of length L.
    """
    n = len(Spc.sectors)
    if n == 1:
        return [0] * L

    # Sort sector indices descending by charge (high-charge sectors first).
    # For tuples this is lexicographic; for scalars numerical.
    sector_order = sorted(
        range(n),
        key=lambda k: _charge_sort_key(Spc.sectors[k].charge),
        reverse=True,
    )

    # Priority 1: single sector k applied L times returns to Q_vac.
    # Works for Z2 (occupied sector flips parity each time, returning to 0
    # after an even number of sites) and SU2 (even L, dimer path returns to 0).
    for k in sector_order:
        Q = Q_vac
        for _ in range(L):
            Q = _next_charge(group, Q, Spc.sectors[k].charge)
        if Q == Q_vac:
            return [k] * L

    # Priority 2: period-2 alternation (ki, kj) where applying ki then kj
    # returns to Q_vac.  Works for U1 (alternating ±1 charges) and Band
    # spaces (alternating spin-up/spin-down or doublet/doublet).
    #
    # Collect all valid pairs, then prefer those where both sectors have a
    # neutral first-component charge (i.e., U1 = 0 for product groups).  This
    # selects spin-up/spin-down alternation for Band U1⊗U1 over the
    # doubly-occupied/empty alternation, which is closer to half-filling.
    valid_pairs = []
    for ki in sector_order:
        Q_mid = _next_charge(group, Q_vac, Spc.sectors[ki].charge)
        for kj in sector_order:
            if _next_charge(group, Q_mid, Spc.sectors[kj].charge) != Q_vac:
                continue
            cfg = [ki if i % 2 == 0 else kj for i in range(L)]
            Q = Q_vac
            for c in cfg:
                Q = _next_charge(group, Q, Spc.sectors[c].charge)
            if Q == Q_vac:
                valid_pairs.append((ki, kj))

    if valid_pairs:
        # Prefer pairs where both sectors have neutral first charge component
        # (works for both scalar and tuple charges).
        def _first_comp(q):
            return q[0] if isinstance(q, tuple) else q

        neutral_pairs = [
            (ki, kj) for ki, kj in valid_pairs
            if _first_comp(Spc.sectors[ki].charge) == 0
            and _first_comp(Spc.sectors[kj].charge) == 0
        ]
        ki, kj = (neutral_pairs if neutral_pairs else valid_pairs)[0]
        return [ki if i % 2 == 0 else kj for i in range(L)]

    # Fallback: greedy — at each step pick the sector that minimizes the
    # distance of the new bond charge from Q_vac.  This does not guarantee
    # Q_L = Q_vac but is better than an arbitrary choice.
    warnings.warn(
        f"init_mps: could not find an exact auto-config for L={L}. "
        "The product state may not close to the vacuum boundary. "
        "Pass an explicit config for precise charge targeting.",
        UserWarning,
        stacklevel=4,
    )
    cfg = []
    Q = Q_vac
    for _ in range(L):
        best_k = min(
            range(n),
            key=lambda k: _charge_dist(_next_charge(group, Q, Spc.sectors[k].charge), Q_vac),
        )
        cfg.append(best_k)
        Q = _next_charge(group, Q, Spc.sectors[best_k].charge)
    return cfg


# ---------------------------------------------------------------------------
# Product state (bond_dim=1)
# ---------------------------------------------------------------------------

def _product_state_mps(
    L: int,
    Spc: Index,
    Op: Dict[str, Tensor],
    cfg: list[int],
    Q: list,
) -> MPS:
    """Build a bond-dim-1 product-state MPS.

    Each site tensor has exactly one non-zero block, with the (0, 0, 0) entry
    set to 1.0 in the multiplet basis. Left and right bond indices each carry
    a single sector determined by the precomputed charge path.

    Parameters
    ----------
    L:
        Chain length.
    Spc:
        Physical Index.
    Op:
        Operator dictionary from `load_space`.
    cfg:
        Site sector indices (length L).
    Q:
        Bond charge sequence Q[0..L] from `_bond_charges`.

    Returns
    -------
    MPS
        Right-canonical product-state MPS with center 0.
    """
    group = Spc.group
    tensors = []
    for i in range(L):
        l_idx = Index(
            direction=Direction.IN,
            group=group,
            sectors=(Sector(charge=Q[i], dim=1),),
        )
        r_idx = Index(
            direction=Direction.OUT,
            group=group,
            sectors=(Sector(charge=Q[i + 1], dim=1),),
        )
        t = Tensor.zeros(
            [l_idx, r_idx, Spc],
            itags=[f'A{i:02d}', f'A{i + 1:02d}', f's{i:02d}'],
        )
        # There is exactly one admissible block (l_idx and r_idx each carry
        # only one sector, so charge conservation fixes the physical sector
        # uniquely). Set the single multiplet component to 1.
        _, blk = next(iter(t.data.items()))
        blk[0, 0, 0] = 1.0
        tensors.append(t)
    mps = MPS(tensors, center=None)
    mps.canonical(0)
    mps.normalize()
    return mps


# ---------------------------------------------------------------------------
# Random mode (bond_dim > 1)
# ---------------------------------------------------------------------------

def _random_mps(
    L: int,
    Spc: Index,
    Op: Dict[str, Tensor],
    Q: list,
    bond_dim: int,
    seed: int,
) -> MPS:
    """Build a random MPS with group-derived bond sectors.

    Bond sectors are chosen via BFS from the center-bond charge Q_c =
    Q[L//2], using depth d=2. This guarantees:

    - The sector set is L-independent (fixed count regardless of chain length).
    - All sectors are within 2 fusion steps of the physically relevant charge.
    - No phantom sectors (charges that cannot be reached by any physical
      configuration) waste bond dimension.

    Parameters
    ----------
    L:
        Chain length.
    Spc:
        Physical Index.
    Op:
        Operator dictionary from `load_space`.
    Q:
        Bond charge sequence from `_bond_charges` (used to determine Q_c).
    bond_dim:
        Target total bond dimension distributed across bond sectors.
    seed:
        Base random seed; site i uses seed+i.

    Returns
    -------
    MPS
        Right-canonical random MPS with center 0.
    """
    group = Spc.group
    vac = Op['vac']
    Q_c = Q[L // 2]

    charges = _reachable_charges(group, Spc, Q_c, d=2)

    # For product groups the BFS cross-product can grow; cap the number of
    # sectors to keep bond dimension well distributed.
    n_max = max(4, bond_dim // 10)
    if len(charges) > n_max:
        charges = sorted(charges, key=lambda q: _charge_dist(q, Q_c))[:n_max]

    n_sectors = len(charges)
    dim_per_sector = max(1, bond_dim // n_sectors)
    bulk = Index(
        direction=Direction.IN,
        group=group,
        sectors=tuple(
            Sector(charge=q, dim=dim_per_sector)
            for q in sorted(charges, key=_charge_sort_key)
        ),
    )

    tensors = []
    for i in range(L):
        l_idx = vac if i == 0 else bulk
        r_idx = (vac if i == L - 1 else bulk).flip()
        t = Tensor.random(
            [l_idx, r_idx, Spc],
            seed=seed + i,
            itags=[f'A{i:02d}', f'A{i + 1:02d}', f's{i:02d}'],
        )
        tensors.append(t)
    mps = MPS(tensors, center=None)
    # Two-pass canonicalization: sweep right to L-1 (left-canonical) then
    # left back to 0 (right-canonical). This compresses bond dimensions from
    # both ends — analogous to one full DMRG sweep — and avoids leaving the
    # leftmost bonds undercompressed after a single right-to-left pass.
    mps.canonical(L - 1)
    mps.canonical(0)
    return mps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_mps(
    L: int,
    Spc: Index,
    Op: Dict[str, Tensor],
    bond_dim: int = 1,
    *,
    config: Optional[list[int]] = None,
    seed: int = 42,
) -> MPS:
    """Construct an initial MPS for DMRG.

    Works for all three particle types (bosonic, fermionic, conductor) without
    a particle-type argument. The `(Spc, Op)` pair from `load_space` carries
    all required symmetry information.

    Parameters
    ----------
    L:
        Chain length.
    Spc:
        Physical Index returned by `load_space`.
    Op:
        Operator dictionary returned by `load_space`.
    bond_dim:
        Target bond dimension.

        - `bond_dim=1`: deterministic product state, bond dimension 1. Best
          used with CBE (`scheme='1sp'`) or 2-site (`scheme='2s'`) DMRG.
        - `bond_dim>1`: random MPS with group-derived bond sectors.  Bond
          sectors are chosen by BFS from the center-bond charge to depth 2,
          fixing the sector count regardless of L.
    config:
        Optional list of physical-sector indices (0-based into `Spc.sectors`),
        one per site. When `None` (default) an auto-balanced configuration is
        selected: alternating high/low sectors for 2-sector spaces, single
        neutral-charge sector for 3-sector spaces, alternating neutral-pair
        for 4-sector spaces, and the SU2 dimer path for pure-SU2 spaces.
        The auto-config is designed for even L and balanced (half-filled)
        systems; pass an explicit config for odd L or unusual fillings.
    seed:
        Base random seed used when `bond_dim>1`.  Site `i` uses `seed+i`.

    Returns
    -------
    MPS
        Right-canonical MPS with orthogonality center at site 0.

    Raises
    ------
    ValueError
        If `bond_dim < 1` or if an explicit `config` has the wrong length.

    Examples
    --------
    Product state for Heisenberg spin-1/2 (U1), ready for 1sp DMRG:

    >>> from nicole import load_space
    >>> from alice import init_mps
    >>> Spc, Op = load_space('Spin', 'U1', {'J': 0.5})
    >>> mps = init_mps(20, Spc, Op, bond_dim=1)

    Random MPS for spinless fermions (U1), bond dimension 32:

    >>> Spc, Op = load_space('Ferm', 'U1')
    >>> mps = init_mps(20, Spc, Op, bond_dim=32)
    """
    if bond_dim < 1:
        raise ValueError(f"bond_dim must be >= 1, got {bond_dim}")

    if config is not None and len(config) != L:
        raise ValueError(
            f"config has length {len(config)}, expected {L}"
        )

    group = Spc.group
    Q_vac = Op['vac'].sectors[0].charge

    cfg = (
        list(config)
        if config is not None
        else _auto_config(L, Spc, group, Q_vac)
    )

    Q = _bond_charges(group, Spc, cfg, Q_vac)

    if bond_dim == 1:
        return _product_state_mps(L, Spc, Op, cfg, Q)
    return _random_mps(L, Spc, Op, Q, bond_dim, seed)
