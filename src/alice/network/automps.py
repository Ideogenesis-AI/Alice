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
  chosen via a breadth-first search (BFS) from the center-bond charge,
  replacing the hardcoded heuristics in the per-example `_random_mps` helpers.

Both modes are particle-type agnostic: the `(Spc, Op)` pair fully encodes
all symmetry information, so no separate `spin=`, `symmetry=`, or
`particle_type=` argument is required.

Charge conventions
------------------
All standard Nicole physical spaces use a particle-hole symmetric convention:
- Spin U1: Sz = ±1/2 mapped to charges ±1.
- Ferm U1: empty = -1, occupied = +1.
- Band (U1⊗U1 or U1⊗SU2): half-filled site has first U1 component = 0.

For even L with a balanced auto-config the center-bond charge Q_c is in
{0, (0,0), (0,...)}. For odd L the bond charge path cannot return to Q_vac
and Q[L] ≠ Q_vac; `init_mps` logs a WARNING and the `target_qn` parameter
allows the caller to select the desired sector explicitly.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from nicole import Direction, Tensor
from nicole.index import Index, Sector

from .network import MPS


logger = logging.getLogger(__name__)


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

def _auto_config(L: int, Spc: Index, group, Q_vac, target_qn) -> list[int]:
    """Choose a site configuration that targets a given right-boundary charge.

    Returns a list of physical-sector indices (length L) whose accumulated bond
    charge `Q[L]` equals `target_qn` whenever an exact pattern exists. When
    no exact pattern is found, falls back to a greedy heuristic that minimizes
    `_charge_dist(Q_next, target_qn)` at each step; in that case `Q[L]` may
    differ from `target_qn` and `init_mps` (the single warning site) will log
    a WARNING.

    This function is a pure helper with no side effects.

    Strategy (in priority order)
    ----------------------------
    1. **Single-sector fill** — all sites use sector `k`; accepted if the
       accumulated path closes exactly at `target_qn`. Works for Z2/SU2 with
       compatible `(L, target_qn)` combinations.
    2. **Period-2 alternation** — all (ki, kj) pairs are enumerated; the full
       path of length L is checked against `target_qn`. Pairs where both
       sectors have a neutral first-component (U1=0) are preferred, favoring
       half-filling for Band spaces.
    3. **Greedy fallback** — at each step picks the sector whose next bond
       charge is closest (L1) to `target_qn`. Exact for Abelian 2-sector
       spaces when `target_qn` is achievable; best-effort otherwise.

    Parameters
    ----------
    L:
        Chain length.
    Spc:
        Physical Index returned by `load_space`.
    group:
        Nicole group object.
    Q_vac:
        Vacuum charge; used as the left boundary (starting charge).
    target_qn:
        Desired right-boundary charge `Q[L]`.

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

    # Priority 1: single sector k applied L times reaches target_qn.
    for k in sector_order:
        Q = Q_vac
        for _ in range(L):
            Q = _next_charge(group, Q, Spc.sectors[k].charge)
        if Q == target_qn:
            return [k] * L

    # Priority 2: period-2 alternation (ki, kj) — enumerate all pairs and
    # check the full length-L path against target_qn. No per-period shortcut
    # is applied, so this works for any target (not just Q_vac).
    #
    # Among valid pairs prefer those where both sectors have a neutral first
    # charge component (U1=0), which selects spin-up/spin-down alternation for
    # Band U1⊗U1 over doubly-occupied/empty, staying closer to half-filling.
    valid_pairs = []
    for ki in sector_order:
        for kj in sector_order:
            cfg = [ki if i % 2 == 0 else kj for i in range(L)]
            Q = Q_vac
            for c in cfg:
                Q = _next_charge(group, Q, Spc.sectors[c].charge)
            if Q == target_qn:
                valid_pairs.append((ki, kj))

    if valid_pairs:
        def _first_comp(q):
            return q[0] if isinstance(q, tuple) else q

        neutral_pairs = [
            (ki, kj) for ki, kj in valid_pairs
            if _first_comp(Spc.sectors[ki].charge) == 0
            and _first_comp(Spc.sectors[kj].charge) == 0
        ]
        ki, kj = (neutral_pairs if neutral_pairs else valid_pairs)[0]
        return [ki if i % 2 == 0 else kj for i in range(L)]

    # Fallback: greedy — minimize L1 distance to target_qn at each step.
    # For Abelian 2-sector spaces this reaches target_qn exactly when it is
    # achievable (correct parity/magnitude); otherwise returns the closest
    # approximation. init_mps checks Q[L] == target_qn and warns if not.
    cfg = []
    Q = Q_vac
    for _ in range(L):
        best_k = min(
            range(n),
            key=lambda k: _charge_dist(_next_charge(group, Q, Spc.sectors[k].charge), target_qn),
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
    Q: list,
    bond_dim: int,
    seed: int,
    target_qn,
) -> MPS:
    """Build a random MPS with group-derived bond sectors.

    Bond sectors are chosen via BFS from the center-bond charge Q_c =
    Q[L//2], using depth d=2. This guarantees:

    - The sector set is L-independent (fixed count regardless of chain length).
    - All sectors are within 2 fusion steps of the physically relevant charge.
    - No phantom sectors (charges that cannot be reached by any physical
      configuration) waste bond dimension.

    Both boundary indices are constructed as single-sector dummy indices with
    dimension 1: the left boundary carries Q[0] = Q_vac and the right boundary
    carries `target_qn`. Using `target_qn` (instead of always pinning the right
    boundary to Q_vac) is essential for odd L, where Q[L] ≠ Q_vac and a
    hard-coded vacuum right boundary makes every block of the last tensor
    charge-forbidden, producing a zero MPS after canonicalization.

    Parameters
    ----------
    L:
        Chain length.
    Spc:
        Physical Index.
    Q:
        Bond charge sequence from `_bond_charges` (used to determine Q_c and
        the left boundary charge Q[0]).
    bond_dim:
        Target total bond dimension distributed across bond sectors.
    seed:
        Base random seed; site i uses seed+i.
    target_qn:
        Charge for the right boundary dummy index. Should equal Q[L] for a
        physically consistent MPS; may be overridden by the caller via
        `init_mps(target_qn=...)`.

    Returns
    -------
    MPS
        Right-canonical random MPS with center 0.
    """
    group = Spc.group
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

    # Single-sector boundary indices: left carries Q[0] = Q_vac, right
    # carries target_qn (= Q[L] for the default auto-config path).
    l_bnd = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=Q[0], dim=1),),
    )
    r_bnd = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(Sector(charge=target_qn, dim=1),),
    )

    tensors = []
    for i in range(L):
        l_idx = l_bnd if i == 0 else bulk
        r_idx = r_bnd if i == L - 1 else bulk.flip()
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
    target_qn=None,
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
        - `bond_dim>1`: random MPS with group-derived bond sectors. Bond
          sectors are chosen by BFS from the center-bond charge to depth 2,
          fixing the sector count regardless of L.
    config:
        Optional list of physical-sector indices (0-based into `Spc.sectors`),
        one per site. When `None` (default) `_auto_config` selects a balanced
        configuration targeting `target_qn` (see below).
    target_qn:
        Desired total quantum number of the chain, i.e. the right-boundary
        charge `Q[L]`. When `None` (default) the target is `Q_vac`
        (half-filling).

        The parameter affects both modes, but in different ways:

        - `bond_dim=1` (product state): `target_qn` is passed to `_auto_config`
          which tries to find a config whose charge path ends at `target_qn`.
          The right boundary is always `Q[L]` from the resulting path.
        - `bond_dim>1` (random MPS): `_auto_config` targets `target_qn` for a
          physically relevant center-bond charge. The right boundary is
          explicitly pinned to `target_qn`.

        In both modes, if the auto-config's `Q[L]` differs from `target_qn`
        (meaning `target_qn` is unreachable for this `L` and physical space),
        a `ValueError` is raised. When `target_qn` is `None` and `Q[L]` is
        not `Q_vac` (e.g. odd L), a WARNING is logged instead.
    seed:
        Base random seed used when `bond_dim>1`. Site `i` uses `seed+i`.

    Returns
    -------
    MPS
        Right-canonical MPS with orthogonality center at site 0.

    Raises
    ------
    ValueError
        If `bond_dim < 1`, if an explicit `config` has the wrong length, or if
        an explicit `target_qn` is not reachable for the given `L` and physical
        space (i.e. auto-config's charge path ends at a different sector).

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

    Odd-length chain — product state and random MPS with explicit Sz = +½:

    >>> Spc, Op = load_space('Spin', 'U1', {'J': 0.5})
    >>> mps1 = init_mps(7, Spc, Op, bond_dim=1,  target_qn=1)
    >>> mps2 = init_mps(7, Spc, Op, bond_dim=32, target_qn=1)
    """
    if bond_dim < 1:
        raise ValueError(f"bond_dim must be >= 1, got {bond_dim}")

    if config is not None and len(config) != L:
        raise ValueError(
            f"config has length {len(config)}, expected {L}"
        )

    group = Spc.group
    Q_vac = Op['vac'].sectors[0].charge

    # Track whether target_qn was given explicitly so we can emit the right
    # warning message when the auto-config cannot achieve it.
    _target_given = target_qn is not None
    if target_qn is None:
        target_qn = Q_vac

    cfg = (
        list(config)
        if config is not None
        else _auto_config(L, Spc, group, Q_vac, target_qn)
    )

    Q = _bond_charges(group, Spc, cfg, Q_vac)

    # Determine the effective right-boundary charge.
    # - When target_qn was given explicitly, honor it for both modes. For
    #   bond_dim>1 the boundary is pinned directly; for bond_dim=1 the boundary
    #   is Q[L] (rigid), so a mismatch warning is emitted below.
    # - When target_qn was defaulted to Q_vac, use Q[L] as the effective right
    #   boundary for bond_dim>1. This is the crucial fix for odd L: the auto-
    #   config cannot return to Q_vac, so using Q_vac as the right boundary
    #   makes all blocks charge-forbidden; Q[L] is always safe.
    effective_right = target_qn if _target_given else Q[L]

    # When the charge path does not end at the target, either warn (default
    # target, e.g. odd L) or raise (explicit target that is unreachable).
    if Q[L] != target_qn:
        if not _target_given:
            # Default target (Q_vac) was not achieved — typically odd L.
            logger.warning(
                "init_mps: auto-config could not return to Q_vac=%s for L=%d "
                "(got Q[L]=%s). Multiple target sectors may be valid "
                "(e.g. Sz = \u00b1\u00bd for odd-L spin-\u00bd). "
                "Pass target_qn= to select a sector explicitly.",
                Q_vac, L, Q[L],
            )
        else:
            raise ValueError(
                f"init_mps: target_qn={target_qn!r} is not reachable for "
                f"L={L} with the given physical space "
                f"(auto-config ended at Q[L]={Q[L]!r}). "
                "Adjust target_qn or pass an explicit config= whose charge "
                "path reaches the desired sector."
            )

    if bond_dim == 1:
        return _product_state_mps(L, Spc, Op, cfg, Q)
    return _random_mps(L, Spc, Q, bond_dim, seed, effective_right)
