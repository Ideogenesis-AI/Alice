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


"""Model builders for MPO Hamiltonian construction.

Each builder takes a list of `Interaction` objects produced by the geometry
stage, populates the tensor fields and sets `cpl` in place, then returns
`(spc, ops)`. Coupling constants are stored in `intr.cpl` but are **not**
baked into the tensors; `build_hamiltonian` applies them when constructing
the MPO.

Supported models
----------------
- `build_heisenberg` — Heisenberg spin model on any spin-`s` site.
- `build_free_fermion` — spinless free-fermion (tight-binding) model.
- `build_hubbard` — Hubbard model (spinful fermions with on-site U).

Each function accepts a `space_fn` keyword argument that overrides the
default operator-set builder (`build_bosonic`, `build_fermionic`, or
`build_conductor` from `system.py`). This enables custom physical spaces
while reusing the standard model structure.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from nicole.index import Index
from nicole import Tensor

from alice.network.interaction import Interaction, Interaction1Site, Interaction2Site
from alice.physics.system import build_bosonic, build_fermionic, build_conductor


def build_heisenberg(
    interactions: List[Interaction],
    L: int = 0,
    *,
    symmetry: str = 'U1',
    spin: float = 0.5,
    J: float = 1.0,
    Jp: float = 0.0,
    space_fn: Optional[Callable] = None,
    **_ignored,
) -> Tuple[Index, Dict[str, Tensor]]:
    """Populate interactions for a Heisenberg spin model.

    Assigns the spin-spin coupling `J S†_i · S_j` to each NN bond and
    `Jp S†_i · S_j` to each NNN bond. Tensors are built from `build_bosonic`
    (or `space_fn` if provided) and stored without baking in the coupling;
    `build_hamiltonian` applies `intr.cpl` when constructing the MPO.

    Parameters
    ----------
    interactions:
        List of `Interaction2Site` objects from the geometry stage. Modified
        in place.
    L:
        Chain length. Unused here (present for uniform model-builder API).
    symmetry:
        Symmetry passed to `build_bosonic` — `'U1'` or `'SU2'`.
    spin:
        Site spin quantum number.
    J:
        Coupling for NN bonds (label `'NN'`).
    Jp:
        Coupling for NNN bonds (label `'NNN'`). Interactions with `Jp == 0`
        are left with `cpl = 0.0` and no tensors; `build_hamiltonian` skips
        them.
    space_fn:
        Optional replacement for `build_bosonic`. Must have the same
        signature: `space_fn(symmetry, spin) -> (spc, ops)`.
    **_ignored:
        Extra TOML keys forwarded from the dispatcher are silently ignored.

    Returns
    -------
    tuple
        `(spc, ops)` from the operator-set builder.
    """
    _space = space_fn if space_fn is not None else build_bosonic
    spc, ops = _space(symmetry, spin)

    S4    = ops['S4']
    S4dag = ops['S4dag']
    I4mid = ops['I4mid']

    for intr in interactions:
        if not isinstance(intr, Interaction2Site):
            continue

        if 'NN' in intr.label:
            intr.cpl           = J
            intr.leading_tnsr  = S4.clone()
            intr.terminal_tnsr = S4dag.clone()
            if intr.terminal_site > intr.leading_site + 1:
                intr.intermid_tnsr = I4mid.clone()

        elif 'NNN' in intr.label:
            intr.cpl = Jp
            # Only populate tensors when the coupling is non-zero; otherwise
            # build_hamiltonian skips the interaction entirely (cpl == 0.0).
            if Jp != 0.0:
                intr.leading_tnsr  = S4.clone()
                intr.terminal_tnsr = S4dag.clone()
                if intr.terminal_site > intr.leading_site + 1:
                    intr.intermid_tnsr = I4mid.clone()

    return spc, ops


def build_free_fermion(
    interactions: List[Interaction],
    L: int = 0,
    *,
    symmetry: str = 'U1',
    t: float = 1.0,
    tp: float = 0.0,
    mu: float = 0.0,
    space_fn: Optional[Callable] = None,
    **_ignored,
) -> Tuple[Index, Dict[str, Tensor]]:
    """Populate interactions for a spinless free-fermion (tight-binding) model.

    Assigns the hopping term `-t (c†_i c_j + h.c.)` to NN bonds and
    `-tp (...)` to NNN bonds. An optional chemical potential `-mu n_i` is
    added as one `Interaction1Site` per site when `mu != 0`.

    The coupling stored in `intr.cpl` for hopping terms is negative (`-t`
    for NN) because `build_hamiltonian` scales `terminal_tnsr` by `cpl`; the
    sign encodes the physics.

    Parameters
    ----------
    interactions:
        List of `Interaction` objects from the geometry stage. Modified in
        place; `Interaction1Site` objects for the chemical potential are
        appended when `mu != 0`.
    L:
        Chain length. Required when `mu != 0` to generate the on-site terms;
        the dispatcher always supplies it.
    symmetry:
        Symmetry passed to `build_fermionic` — `'U1'` or `'Z2'`.
    t:
        NN hopping amplitude. Stored as `cpl = -t`.
    tp:
        NNN hopping amplitude. Stored as `cpl = -tp`.
    mu:
        Chemical potential. Stored as `cpl = -mu` on each on-site
        `Interaction1Site`; zero by default (no on-site term).
    space_fn:
        Optional replacement for `build_fermionic`.
    **_ignored:
        Extra TOML keys silently ignored.

    Returns
    -------
    tuple
        `(spc, ops)` from the operator-set builder.
    """
    _space = space_fn if space_fn is not None else build_fermionic
    spc, ops = _space(symmetry)

    G4    = ops['G4']
    G4dag = ops['G4dag']
    Z4mid = ops['Z4mid']

    for intr in interactions:
        if not isinstance(intr, Interaction2Site):
            continue

        if 'NN' in intr.label:
            intr.cpl           = -t
            intr.leading_tnsr  = G4.clone()
            intr.terminal_tnsr = G4dag.clone()
            if intr.terminal_site > intr.leading_site + 1:
                intr.intermid_tnsr = Z4mid.clone()

        elif 'NNN' in intr.label:
            intr.cpl = -tp
            if tp != 0.0:
                intr.leading_tnsr  = G4.clone()
                intr.terminal_tnsr = G4dag.clone()
                if intr.terminal_site > intr.leading_site + 1:
                    intr.intermid_tnsr = Z4mid.clone()

    # --- Chemical potential: -mu n_i for each site ---
    if mu != 0.0:
        N4 = ops['N4']
        for site in range(L):
            interactions.append(Interaction1Site(
                cpl=-mu,
                label=['mu', 'onsite'],
                site=site,
                tnsr=N4.clone(),
            ))

    return spc, ops


def build_hubbard(
    interactions: List[Interaction],
    L: int,
    *,
    symmetry: str = 'U1,U1',
    t: float = 1.0,
    U: float = 4.0,
    tp: float = 0.0,
    mu: float = 0.0,
    space_fn: Optional[Callable] = None,
    **_ignored,
) -> Tuple[Index, Dict[str, Tensor]]:
    """Populate interactions for a Hubbard model.

    Assigns the hopping term `-t Σ_σ (c†_{i,σ} c_{j,σ} + h.c.)` to NN bonds
    (via JW-dressed operators from `build_conductor`) and appends one
    `Interaction1Site` per site for the on-site Hubbard-U term `U n_{up} n_{dn}`
    and an optional chemical potential term.

    The chemical potential `mu` is defined **relative to half-filling**. The
    half-filling chemical potential for the standard Hubbard model is
    `mu_half = U / 2`, derived from the particle-hole symmetry condition
    `<n> = 1`. The effective on-site energy is therefore:

        -mu_eff * n_i,   mu_eff = mu + U / 2.

    This term is added whenever `mu_eff != 0`, i.e. even at `mu = 0` when
    `U != 0`.

    The coupling is NOT baked into the tensors; `build_hamiltonian` applies
    `intr.cpl` when constructing the MPO.

    Parameters
    ----------
    interactions:
        List of `Interaction` objects from the geometry stage. Modified in
        place; `Interaction1Site` objects for Hubbard-U and chemical potential
        are appended.
    L:
        Chain length. Used to create one `Interaction1Site` per site for each
        on-site term.
    symmetry:
        Symmetry passed to `build_conductor` — e.g. `'U1,U1'` or `'U1,SU2'`.
    t:
        NN hopping amplitude. Stored as `cpl = -t`.
    U:
        On-site Coulomb repulsion. Stored as `cpl = U` on Hubbard-U
        `Interaction1Site` objects.
    tp:
        NNN hopping amplitude. Stored as `cpl = -tp`.
    mu:
        Chemical potential relative to half-filling. The effective chemical
        potential applied is `mu_eff = mu + U / 2`. Stored as
        `cpl = -mu_eff` on chemical-potential `Interaction1Site` objects.
        When `mu_eff == 0` (i.e. `mu == 0` and `U == 0`) no term is added.
    space_fn:
        Optional replacement for `build_conductor`.
    **_ignored:
        Extra TOML keys silently ignored.

    Returns
    -------
    tuple
        `(spc, ops)` from the operator-set builder.
    """
    _space = space_fn if space_fn is not None else build_conductor
    spc, ops = _space(symmetry)

    G4    = ops['G4']
    G4dag = ops['G4dag']
    NN4   = ops['NN4']
    N4    = ops['N4']
    Z4mid = ops['Z4mid']

    # --- 2-site hopping terms ---
    for intr in interactions:
        if not isinstance(intr, Interaction2Site):
            continue

        if 'NN' in intr.label:
            intr.cpl           = -t
            intr.leading_tnsr  = G4.clone()
            intr.terminal_tnsr = G4dag.clone()
            if intr.terminal_site > intr.leading_site + 1:
                intr.intermid_tnsr = Z4mid.clone()

        elif 'NNN' in intr.label:
            intr.cpl = -tp
            if tp != 0.0:
                intr.leading_tnsr  = G4.clone()
                intr.terminal_tnsr = G4dag.clone()
                if intr.terminal_site > intr.leading_site + 1:
                    intr.intermid_tnsr = Z4mid.clone()

    # --- On-site Hubbard-U terms: U n_up n_dn ---
    for site in range(L):
        interactions.append(Interaction1Site(
            cpl=U,
            label=['U', 'onsite'],
            site=site,
            tnsr=NN4.clone(),
        ))

    # --- Chemical potential: -(mu + U/2) n_i, referenced to half-filling ---
    # The half-filling shift U/2 comes from the particle-hole symmetry of the
    # Hubbard model: at mu_eff = U/2 the average occupation is exactly 1.
    mu_eff = mu + U / 2.0
    if mu_eff != 0.0:
        for site in range(L):
            interactions.append(Interaction1Site(
                cpl=-mu_eff,
                label=['mu', 'onsite'],
                site=site,
                tnsr=N4.clone(),
            ))

    return spc, ops
