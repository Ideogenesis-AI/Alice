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


"""Infinite AutoMPO: automaton-based Hamiltonian iMPO construction.

Why this is not `build_hamiltonian`
-----------------------------------

The finite builder accumulates one term at a time with `oplus` on the outer
axes and then squeezes the result with `MPO.compact()`. Both steps lean on
the chain having a trivial dimension-1 bond at each end, and neither assumption
survives on an infinite chain:

- The left and right bonds of the unit cell are the *same* bond, and the
  Hamiltonian is recovered not from trivial boundaries but from a unique
  "start" channel and a unique "finish" channel of an upper-triangular

      W = [[I, C, D],
           [0, A, B],
           [0, 0, I]].

  A plain `oplus` of two such operators duplicates both channels. That is not
  a harmless redundancy: the start channel is what propagates "no term has
  begun yet" down the chain, and a second copy of it generates a whole extra
  family of spurious terms.
- `MPO.compact()` is an open-boundary two-sweep canonicalization. A unit cell
  has no open end for it to start from, and it would scramble the
  upper-triangular structure that the VUMPS environment solver depends on.

So the unit cell is assembled directly as a finite-state automaton instead.

The automaton
-------------

Read a Hamiltonian term as a walk that enters at the start channel, fires the
leading operator, carries an operator string across the intervening bonds, and
exits through the finish channel at the terminal operator. The bond index at
position `b` of the unit cell is therefore

    start (+) [one channel per live operator string] (+) finish

and the entries of `W[i]` are exactly the transitions of that walk. A two-site
interaction `k` with leading site `l_k` and span `d_k = t_k - l_k` contributes
the channels `(k, 1) … (k, d_k)`, where channel `(k, j)` sits at bond
`(l_k + j) % N`: the string has advanced `j` bonds since it started. Because
`j` is part of the channel identity, an interaction reaching beyond the unit
cell simply places several channels on the same bond, so the range is not
restricted by `N`.

Both the start and the finish channel carry the vacuum charge, and they are
laid down first and last in the concatenation. Within the vacuum charge sector
of any bond index, start is therefore at offset `0` and finish at offset `-1`;
`boundary_vectors` relies on that and nothing else does.
"""

from __future__ import annotations

import logging
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

from nicole import Direction, Index, Tensor
from nicole import einsum, identity, oplus

from .infinite import iMPO
from .interaction import (
    Interaction,
    Interaction1Site,
    Interaction2Site,
    InteractionNSite,
)
from .network import MPO

logger = logging.getLogger(__name__)

# Row/column labels for the two boundary channels of the automaton. A channel
# of an actual operator string is labelled by the tuple `(k, j)` instead, so a
# plain `isinstance(label, str)` separates the two kinds.
_START = '_start_'
_FINISH = '_finish_'

# Placeholder itags used while the blocks of one site tensor are concatenated.
# The final bond and physical itags are applied once the tensor is complete.
_TMP_ITAGS = ('_wl_', '_wr_', '_ws_', '_ws_')

_Label = Union[str, Tuple[int, int]]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _split_active(
    interactions: List[Interaction],
    N: int,
) -> Tuple[List[Interaction2Site], List[Interaction1Site]]:
    """Separate the non-zero interactions into two-site and on-site groups.

    Also checks everything the automaton assembly later takes for granted:
    site indices inside the unit cell, a positive span, and the tensor slots
    each interaction type needs.

    Parameters
    ----------
    interactions:
        Interaction list from `build_interaction`, with tensors populated.
    N:
        Unit-cell size.

    Returns
    -------
    tuple
        `(two_site, on_site)`, both filtered to `cpl != 0`.

    Raises
    ------
    ValueError
        If a site index falls outside the unit cell, a span is non-positive,
        or a required tensor slot is `None`.
    TypeError
        If an active interaction is not a supported subclass.
    """
    two_site: List[Interaction2Site] = []
    on_site: List[Interaction1Site] = []

    for k, intr in enumerate(interactions):
        # A zero coupling contributes nothing and may legitimately have its
        # tensor slots unset (e.g. NNN bonds with Jp = 0), so skip it before
        # validating anything.
        if intr.cpl == 0.0:
            continue

        if isinstance(intr, Interaction1Site):
            if not (0 <= intr.site < N):
                raise ValueError(
                    f"interactions[{k}] (Interaction1Site) has site={intr.site}, "
                    f"which is outside the unit cell [0, {N})"
                )
            if intr.tnsr is None:
                raise ValueError(
                    f"interactions[{k}] (Interaction1Site at site {intr.site}) "
                    f"has tnsr=None"
                )
            on_site.append(intr)

        elif isinstance(intr, Interaction2Site):
            if not (0 <= intr.leading_site < N):
                raise ValueError(
                    f"interactions[{k}] (Interaction2Site) has leading_site="
                    f"{intr.leading_site}, which is outside the unit cell "
                    f"[0, {N}). Every infinite-chain bond must be named by the "
                    f"cell site that leads it; the terminal site is the one "
                    f"allowed to run past the cell."
                )
            span = intr.terminal_site - intr.leading_site
            if span < 1:
                raise ValueError(
                    f"interactions[{k}] (Interaction2Site {intr.leading_site}"
                    f"\u2192{intr.terminal_site}) has span {span}; "
                    f"terminal_site must exceed leading_site"
                )
            if intr.leading_tnsr is None:
                raise ValueError(
                    f"interactions[{k}] (Interaction2Site leading_site="
                    f"{intr.leading_site}) has leading_tnsr=None"
                )
            if intr.terminal_tnsr is None:
                raise ValueError(
                    f"interactions[{k}] (Interaction2Site leading_site="
                    f"{intr.leading_site}) has terminal_tnsr=None"
                )
            if span > 1 and intr.intermid_tnsr is None:
                raise ValueError(
                    f"interactions[{k}] (Interaction2Site {intr.leading_site}"
                    f"\u2192{intr.terminal_site}) has intermid_tnsr=None but "
                    f"spans {span} bonds"
                )
            two_site.append(intr)

        elif isinstance(intr, InteractionNSite):
            raise NotImplementedError(
                f"interactions[{k}]: build_ihamiltonian does not yet support "
                "InteractionNSite; only 1-site and 2-site terms are implemented"
            )

        else:
            raise TypeError(
                f"interactions[{k}] has unsupported type "
                f"'{type(intr).__name__}'; expected Interaction1Site or "
                f"Interaction2Site"
            )

    return two_site, on_site


# ---------------------------------------------------------------------------
# Block helpers
# ---------------------------------------------------------------------------

def _zero_block(l_idx: Index, r_idx: Index, bra: Index, ket: Index) -> Tensor:
    """Build an all-zero 4-index block with exactly the requested indices.

    `Tensor.zeros` drops any sector that takes part in no charge-conserving
    block, which would silently narrow a bond and make the later `oplus`
    concatenation disagree on the bond width. In every physical space Alice
    ships, the physical index couples each operator-channel sector to
    something admissible and nothing is dropped, so this is a guard rather
    than a workaround — but an exotic space could trip it, and a clear error
    beats a structural mismatch reported several calls later.

    Parameters
    ----------
    l_idx:
        Left (row) channel index.
    r_idx:
        Right (column) channel index.
    bra:
        Physical bra index.
    ket:
        Physical ket index.

    Returns
    -------
    Tensor
        Zero-filled tensor with axes `(left, right, bra, ket)`.

    Raises
    ------
    ValueError
        If constructing the block dropped a sector from either channel index.
    """
    Z = Tensor.zeros([l_idx, r_idx, bra, ket], itags=list(_TMP_ITAGS))
    for axis, want in ((0, l_idx), (1, r_idx)):
        if len(Z.indices[axis].sectors) != len(want.sectors):
            raise ValueError(
                f"padding block lost charge sectors on axis {axis} "
                f"({len(want.sectors)} requested, {len(Z.indices[axis].sectors)} "
                "admissible); the physical space does not couple every operator "
                "channel to a charge-conserving block"
            )
    return Z


def _prep(t: Tensor) -> Tensor:
    """Return a clone of `t` retagged to the placeholder block itags."""
    out = t.clone()
    out.retag([0, 1, 2, 3], list(_TMP_ITAGS))
    return out


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_ihamiltonian(
    interactions: List[Interaction],
    N: int,
    spc: Index,
) -> iMPO:
    """Build a Hamiltonian iMPO for one unit cell of an infinite chain.

    Assembles the unit cell directly as an upper-triangular finite-state
    automaton (see the module docstring). Unlike `build_hamiltonian` there is
    no term-by-term accumulation and no compression pass: the automaton is
    already minimal for finite-range models, and compressing it would destroy
    the channel structure the VUMPS environments rely on.

    Interactions are consumed exactly as `build_hamiltonian` consumes them:
    tensor slots must be pre-filled, and the coupling `cpl` must *not* be
    baked in. It is applied here, scaling the on-site tensor or the terminal
    tensor of a two-site term.

    A two-site interaction names its bond by the unit-cell site that leads it,
    so `leading_site` must lie in `[0, N)`. Its `terminal_site` may run past
    the cell, which is how a bond reaching into the next unit cell is spelled:
    `Interaction2Site(leading_site=N-1, terminal_site=N)` couples the last site
    of one cell to the first site of the next. `intrcmap_1dchain` emits exactly
    that when `bcx='IBC'`.

    Parameters
    ----------
    interactions:
        List of `Interaction1Site` and `Interaction2Site` objects with all
        required tensor fields set.
    N:
        Number of sites in the unit cell.
    spc:
        Physical `Index` (space) shared by all sites.

    Returns
    -------
    iMPO
        Hamiltonian iMPO with bond itags `W{i % N:0M}` and physical itags
        `s{i:02d}`, where *M* is `max(2, len(str(N)))`.

    Raises
    ------
    ValueError
        If `N < 1`, if a site index falls outside the unit cell, if a span is
        non-positive, or if a required tensor slot is `None`.
    TypeError
        If an active interaction is not `Interaction1Site` or
        `Interaction2Site`.
    NotImplementedError
        If an active interaction is an `InteractionNSite`.

    Examples
    --------
    Heisenberg chain with a two-site unit cell:

    >>> cfg = {
    ...     'geometry': {'lattice': 'chain', 'lx': 2, 'bcx': 'IBC'},
    ...     'model': {'category': 'bosonic', 'label': 'Heisenberg',
    ...               'symmetry': 'U1', 'spin': 0.5, 'J': 1.0},
    ... }
    >>> interactions, spc, geo = build_interaction(cfg)
    >>> impo = build_ihamiltonian(interactions, geo.L, spc)
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")

    two_site, on_site = _split_active(interactions, N)

    # Physical identity with trivial auxiliary bonds. Its axes double as the
    # source of the trivial (vacuum) channel indices carried by start/finish
    # and of the physical indices every block shares.
    I4 = identity(spc)
    I4.insert_index(0, direction=Direction.IN, itag='_aux_')
    I4.insert_index(1, direction=Direction.OUT, itag='_aux_')
    triv_in, triv_out = I4.indices[0], I4.indices[1]
    bra, ket = I4.indices[2], I4.indices[3]
    I4 = _prep(I4)

    spans = [intr.terminal_site - intr.leading_site for intr in two_site]

    # Channel (k, j) lives at bond (l_k + j) % N: the string of interaction k
    # has advanced j bonds since leaving its leading site. Appending in order
    # of k and then j leaves each bond's list sorted by (k, j), which is the
    # canonical order every site tensor concatenates in.
    channels_at: List[List[Tuple[int, int]]] = [[] for _ in range(N)]
    for k, intr in enumerate(two_site):
        for j in range(1, spans[k] + 1):
            channels_at[(intr.leading_site + j) % N].append((k, j))

    # On-site terms grouped by site, pre-scaled and summed: the whole group
    # occupies the single (start, finish) entry of its site tensor.
    onsite_at: Dict[int, Tensor] = {}
    for intr in on_site:
        term = intr.tnsr.clone() * intr.cpl
        prev = onsite_at.get(intr.site)
        onsite_at[intr.site] = term if prev is None else prev + term

    def _row_index(label: _Label) -> Index:
        """Left (IN) channel index carried by row `label`."""
        if isinstance(label, str):
            return triv_in
        intr = two_site[label[0]]
        # An interaction of span 1 has no intermediate tensor, so its only
        # incoming channel index comes from the terminal tensor.
        src = intr.intermid_tnsr if intr.intermid_tnsr is not None else intr.terminal_tnsr
        return src.indices[0]

    def _col_index(label: _Label) -> Index:
        """Right (OUT) channel index carried by column `label`."""
        if isinstance(label, str):
            return triv_out
        return two_site[label[0]].leading_tnsr.indices[1]

    def _entry(i: int, row: _Label, col: _Label) -> Optional[Tensor]:
        """Return the automaton transition from `row` to `col` at site `i`.

        `None` means the entry is structurally zero and the caller should
        substitute a padding block.
        """
        # Both identity channels pass straight through; the finish channel
        # has no other outgoing transition.
        if row == _START and col == _START:
            return I4
        if row == _FINISH:
            return I4 if col == _FINISH else None
        if col == _START:
            return None

        if row == _START:
            if col == _FINISH:
                term = onsite_at.get(i)
                return None if term is None else _prep(term)
            # A string starts here only if site i leads it.
            k, j = col
            if j == 1 and two_site[k].leading_site == i:
                return _prep(two_site[k].leading_tnsr)
            return None

        k, j = row
        if col == _FINISH:
            # The string closes on its terminal site, where the coupling is
            # applied — matching how build_hamiltonian scales the term.
            if j == spans[k]:
                return _prep(two_site[k].terminal_tnsr * two_site[k].cpl)
            return None
        # A string only ever advances within its own interaction.
        k2, j2 = col
        if k == k2 and j2 == j + 1:
            return _prep(two_site[k].intermid_tnsr)
        return None

    ndigits = max(2, len(str(N)))
    tensors: List[Tensor] = []

    for i in range(N):
        rows: List[_Label] = [_START, *channels_at[i], _FINISH]
        cols: List[_Label] = [_START, *channels_at[(i + 1) % N], _FINISH]

        row_tensors: List[Tensor] = []
        for row in rows:
            l_idx = _row_index(row)
            blocks: List[Tensor] = []
            for col in cols:
                entry = _entry(i, row, col)
                if entry is None:
                    entry = _zero_block(l_idx, _col_index(col), bra, ket)
                blocks.append(entry)
            # Concatenate the row horizontally; axis 0 is shared by every block.
            row_tensors.append(reduce(lambda a, b: oplus(a, b, axes=1), blocks))

        # Stack the rows vertically; axis 1 is now shared by every row.
        W = reduce(lambda a, b: oplus(a, b, axes=0), row_tensors)
        W.retag([0, 1, 2, 3], [
            f'W{i % N:0{ndigits}d}',
            f'W{(i + 1) % N:0{ndigits}d}',
            f's{i:02d}',
            f's{i:02d}',
        ])
        tensors.append(W)

    impo = iMPO(tensors)
    logger.debug(
        "build_ihamiltonian: N=%d, %d two-site and %d on-site terms, bond dims %s",
        N, len(two_site), len(on_site), impo.bond_dims,
    )
    return impo


# ---------------------------------------------------------------------------
# Boundary vectors and finite windows
# ---------------------------------------------------------------------------

def _cap_index(impo: iMPO) -> Index:
    """Return a dimension-1 IN index carrying the vacuum charge.

    Nicole rejects 1-index tensors, so a channel selector has to be a matrix
    with one trivial axis. `insert_index` is the public way to mint such an
    axis, so a throwaway identity on the physical space supplies it.
    """
    probe = identity(impo[0].indices[2])
    probe.insert_index(0, direction=Direction.IN, itag='_cap_')
    return probe.indices[0]


def _selector(
    cap: Index,
    idx: Index,
    itag: str,
    position: int,
    dtype,
    cap_first: bool,
) -> Tensor:
    """Build a unit vector picking one component of a bond index.

    The bond index passed in is the one the selector must *contract against*,
    so the selector carries its flip. Pairing it with the trivial `cap` axis
    leaves exactly one charge-conserving block — the vacuum sector, which is
    where both automaton boundary channels live — so there is no sector
    ambiguity to resolve.

    Parameters
    ----------
    cap:
        Dimension-1 vacuum index from `_cap_index`, used in the orientation
        the capped MPO axis needs.
    idx:
        Bond index to contract against.
    itag:
        itag for the bond axis, matching the axis it will contract with.
    position:
        Component to set within the vacuum block: `0` for start, `-1` for
        finish.
    dtype:
        Block dtype, taken from the iMPO.
    cap_first:
        Whether the trivial axis comes before the bond axis. The start
        selector caps a left bond and so leads with the cap; the finish
        selector caps a right bond and so trails with it.

    Returns
    -------
    Tensor
        2-index tensor with a single unit entry.
    """
    indices = [cap, idx.flip()] if cap_first else [idx.flip(), cap]
    itags = ['_cap_', itag] if cap_first else [itag, '_cap_']
    v = Tensor.zeros(indices, dtype=dtype, itags=itags)

    _, blk = next(iter(v.data.items()))
    # The bond axis is whichever one is not the trivial cap. Collapsing every
    # other axis — the cap and, for a generic symmetry group, the trailing
    # multiplicity axis — covers the Abelian and generic cases with one line.
    bond_axis = 1 if cap_first else 0
    blk.movedim(bond_axis, 0).reshape(blk.shape[bond_axis], -1)[position, 0] = 1.0
    return v


def boundary_vectors(impo: iMPO, bond: int = 0) -> Tuple[Tensor, Tensor]:
    """Build the start and finish selector vectors for one iMPO bond.

    The automaton closes a Hamiltonian by entering at the start channel and
    leaving at the finish channel. These two 1-index tensors pick out those
    channels: contracting `start` into the left bond of a site tensor keeps
    only the "no term has begun yet" row, and contracting `finish` into a
    right bond keeps only the "the term is complete" column.

    Both channels carry the vacuum charge, and `build_ihamiltonian` lays
    start down first and finish last, so within the vacuum charge sector they
    sit at offsets `0` and `-1`. That convention is the only thing this
    function assumes about the bond layout.

    The two vectors attach to opposite ends, so they come back in opposite
    orientations. `start` has axes `(cap, bond)` and contracts with a left
    (IN) bond; `finish` has axes `(bond, cap)` and contracts with a right
    (OUT) bond. The `cap` axis is a dimension-1 vacuum index, which keeps
    both of them valid Nicole tensors and gives a capped MPO the trivial
    boundary axis it expects.

    Parameters
    ----------
    impo:
        Hamiltonian iMPO produced by `build_ihamiltonian`.
    bond:
        Bond position, taken modulo the unit-cell size. Bond `b` is the one
        between site `b-1` and site `b`. Defaults to `0`.

    Returns
    -------
    tuple
        `(start, finish)`, contractible with a left and a right bond at the
        chosen position respectively.
    """
    N = impo.N
    left_site = bond % N
    right_site = (bond - 1) % N
    cap = _cap_index(impo)

    start = _selector(
        cap, impo[left_site].indices[0], impo[left_site].itags[0],
        0, impo[left_site].dtype, cap_first=True,
    )
    finish = _selector(
        cap.flip(), impo[right_site].indices[1], impo[right_site].itags[1],
        -1, impo[right_site].dtype, cap_first=False,
    )
    return start, finish


def window_mpo(impo: iMPO, L: int) -> MPO:
    """Cut a finite open-boundary MPO out of an infinite Hamiltonian.

    Repeats the unit cell `L` times, caps the left bond with the start vector
    and the right bond with the finish vector, and returns the result as an
    ordinary finite `MPO`. The operator this encodes is the infinite
    Hamiltonian restricted to `L` consecutive sites, dropping every term that
    would reach outside the window — in other words, the open-boundary
    Hamiltonian on those sites.

    That makes it the natural way to check an infinite Hamiltonian: for a
    model both builders can express, `window_mpo` of the iMPO and
    `build_hamiltonian` of the same model must act identically on any state.

    Parameters
    ----------
    impo:
        Hamiltonian iMPO produced by `build_ihamiltonian`.
    L:
        Number of sites in the window. Must be at least 2.

    Returns
    -------
    MPO
        Finite MPO of length `L` with bond itags `W{i:02d}` and physical
        itags `s{i:02d}`.

    Raises
    ------
    ValueError
        If `L < 2`.
    """
    if L < 2:
        raise ValueError(f"window length must be at least 2, got {L}")

    N = impo.N
    # The left cap closes bond 0; the right cap closes the bond leaving site
    # L-1, which is bond L of the infinite chain.
    start, _ = boundary_vectors(impo, 0)
    _, finish = boundary_vectors(impo, L % N)

    tensors: List[Tensor] = []
    for i in range(L):
        W = impo[i % N].clone()
        # Retag before capping: in the finite convention the physical itags
        # follow the absolute site, not the position within the unit cell.
        W.retag([0, 1, 2, 3], ['_wl_', '_wr_', f's{i:02d}', f's{i:02d}'])

        # Each selector carries a trivial cap axis, so contracting it in both
        # closes the channel and leaves the 4-axis MPO layout intact.
        if i == 0:
            v = start.clone()
            v.retag([0, 1], ['_cap_', '_wl_'])
            W = einsum('xa,abrs->xbrs', v, W)
        if i == L - 1:
            v = finish.clone()
            v.retag([0, 1], ['_wr_', '_cap_'])
            W = einsum('abrs,by->ayrs', W, v)

        W.retag([0, 1], [f'W{i:02d}', f'W{i+1:02d}'])
        tensors.append(W)

    return MPO(tensors)
