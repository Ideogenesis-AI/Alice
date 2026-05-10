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


"""AutoMPO: term-by-term Hamiltonian MPO construction."""

from __future__ import annotations

from typing import List, Optional

from nicole import Direction, Index, Tensor
from nicole import identity, oplus, capcup

from .interaction import Interaction, Interaction1Site, Interaction2Site
from .network import MPO


def build_hamiltonian(
    interactions: List[Interaction],
    L: int,
    spc: Index,
    trunc: Optional[dict] = None,
    compact_every: int = 10,
) -> MPO:
    """Build a Hamiltonian MPO from a list of `Interaction` objects.

    Each interaction is accumulated term by term into a running MPO via
    `oplus`, then the result is compressed with two canonical sweeps. This
    approach handles arbitrary symmetries — including non-Abelian SU(2) — by
    delegating sector arithmetic to Nicole's `oplus`.

    All tensor fields (`tnsr`, `leading_tnsr`, `terminal_tnsr`,
    `intermid_tnsr`) must be pre-filled (including any coupling constants)
    before calling this function. `build_hamiltonian` uses them verbatim.

    Parameters
    ----------
    interactions:
        List of `Interaction1Site` or `Interaction2Site` objects with all
        required tensor fields set.
    L:
        Chain length (number of sites).
    spc:
        Physical `Index` (space) shared by all sites. Used to construct the
        site-wise identity tensor.
    trunc:
        Truncation parameters forwarded to `MPO.canonical()` during the
        right-to-left compression sweep. Defaults to `{'thresh': 1e-14}`.
    compact_every:
        Call `MPO.compact` after every this many accumulated terms. Defaults
        to `10`. Set to `0` to disable intermediate compaction.

    Returns
    -------
    MPO
        Compressed Hamiltonian MPO with bond itags `W{i:02d}` / `W{i+1:02d}`
        and physical itags `s{i:02d}` at each site.

    Raises
    ------
    ValueError
        If any required tensor slot is `None`, or if `intermid_tnsr` is `None`
        for a two-site interaction with `terminal_site > leading_site + 1`.
    """
    if trunc is None:
        trunc = {'thresh': 1e-14}

    # Filter out zero-coupling interactions before validation. A zero coupling
    # contributes nothing to the Hamiltonian and may legitimately have tensor
    # fields left unset (e.g. NNN bonds with Jp=0 from the model builder).
    active = [intr for intr in interactions if intr.cpl != 0.0]

    # Validate tensor slots for all active (non-zero coupling) interactions.
    for k, intr in enumerate(active):
        if isinstance(intr, Interaction1Site):
            if intr.tnsr is None:
                raise ValueError(
                    f"interactions[{k}] (Interaction1Site at site {intr.site}) "
                    f"has tnsr=None"
                )
        elif isinstance(intr, Interaction2Site):
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
            if (intr.terminal_site > intr.leading_site + 1
                    and intr.intermid_tnsr is None):
                raise ValueError(
                    f"interactions[{k}] (Interaction2Site {intr.leading_site}"
                    f"→{intr.terminal_site}) has intermid_tnsr=None but "
                    f"terminal_site > leading_site + 1"
                )

    # Build the physical identity with trivial bond axes as a reusable template.
    I = identity(spc)
    I4 = I.clone()
    I4.insert_index(0, direction=Direction.IN,  itag='L')
    I4.insert_index(1, direction=Direction.OUT, itag='R')

    # Step 1 — Initialize: zero tensor with trivial (dim-1) bonds at every site.
    mpo_list: List[Tensor] = []
    for i in range(L):
        z = (I * 0.0).clone()
        z.insert_index(0, direction=Direction.IN,  itag='L')
        z.insert_index(1, direction=Direction.OUT, itag='R')
        z.retag([0, 1, 2, 3], [f'W{i:02d}', f'W{i+1:02d}', f's{i:02d}', f's{i:02d}'])
        mpo_list.append(z)

    # Wrap in an MPO object so compact() can be called on it incrementally.
    # MPO.__init__ copies the list, so mpo_list is not used after this point.
    mpo = MPO(mpo_list, center=None)

    def _identity_term() -> List[Tensor]:
        """Build a term MPO with identity at every site."""
        term: List[Tensor] = []
        for k in range(L):
            t = I4.clone()
            t.retag([0, 1, 2, 3], [f'W{k:02d}', f'W{k+1:02d}', f's{k:02d}', f's{k:02d}'])
            term.append(t)
        return term

    def _merge(term: List[Tensor]) -> None:
        """Accumulate `term` into `mpo` via `oplus`."""
        mpo[0] = oplus(mpo[0], term[0], axes=[1])
        for k in range(1, L - 1):
            mpo[k] = oplus(mpo[k], term[k], axes=[0, 1])
        mpo[L - 1] = oplus(mpo[L - 1], term[L - 1], axes=[0])

    # Step 2 — Accumulate: one `oplus` pass per active interaction term.
    # Coupling is applied here (not baked into tensors by the model builder).
    # Every `compact_every` terms an intermediate compact() is applied to keep
    # bond dimensions in check before they grow too large.
    for n_term, intr in enumerate(active, start=1):
        term = _identity_term()

        if isinstance(intr, Interaction1Site):
            s = intr.site
            # Scale the on-site tensor by the coupling constant.
            t = intr.tnsr.clone() * intr.cpl
            t.retag([0, 1, 2, 3], [f'W{s:02d}', f'W{s+1:02d}', f's{s:02d}', f's{s:02d}'])
            term[s] = t

        elif isinstance(intr, Interaction2Site):
            i_site = intr.leading_site
            j_site = intr.terminal_site

            # Leading tensor carries the operator channel; no coupling here.
            t = intr.leading_tnsr.clone()
            t.retag([0, 1, 2, 3], [
                f'W{i_site:02d}', f'W{i_site+1:02d}',
                f's{i_site:02d}', f's{i_site:02d}',
            ])
            term[i_site] = t

            # Intermediate tensors propagate the operator string; no coupling.
            for k in range(i_site + 1, j_site):
                t = intr.intermid_tnsr.clone()
                t.retag([0, 1, 2, 3], [
                    f'W{k:02d}', f'W{k+1:02d}', f's{k:02d}', f's{k:02d}',
                ])
                term[k] = t

            # Terminal tensor is scaled by the coupling constant.
            t = intr.terminal_tnsr.clone() * intr.cpl
            t.retag([0, 1, 2, 3], [
                f'W{j_site:02d}', f'W{j_site+1:02d}',
                f's{j_site:02d}', f's{j_site:02d}',
            ])
            term[j_site] = t

        _merge(term)

        # oplus invalidates the canonical form; reset before compacting.
        if compact_every > 0 and n_term % compact_every == 0:
            mpo._center = None
            mpo.compact(trunc)
            # compact() reorients bond arrows; restore the original IN/OUT
            # convention so that subsequent oplus calls match the term tensors.
            for k in range(L - 1):
                capcup(mpo[k], 1, mpo[k + 1], 0)

    # Step 3 — Final compress: handles any leftover terms past the last
    # intermediate compact, and ensures the returned MPO is always compressed.
    mpo._center = None
    mpo.compact(trunc)

    return mpo
