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


"""Tests for the infinite AutoMPO finite-state-automaton builder.

The central check compares a finite window cut from the infinite MPO against
the finite MPO built term by term for the same open chain. The two encode the
same operator, so every expectation value must agree to machine precision.
"""

from __future__ import annotations

import logging

import pytest
from nicole import load_space

from alice.network import (
    Interaction2Site,
    InteractionNSite,
    boundary_vectors,
    build_hamiltonian,
    build_ihamiltonian,
    build_interaction,
    iMPO,
    init_mps,
    observe,
    window_mpo,
)


@pytest.fixture(autouse=True)
def _quiet(caplog):
    with caplog.at_level(logging.WARNING, logger='alice'):
        yield


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

# Each entry is (model config, load_space arguments) for a model whose infinite
# MPO must reproduce the corresponding finite one.
MODELS = {
    'heisenberg_u1': (
        dict(category='bosonic', label='Heisenberg', symmetry='U1', spin=0.5, J=1.0),
        ('Spin', 'U1', {'J': 0.5}),
        False,
    ),
    'heisenberg_su2': (
        dict(category='bosonic', label='Heisenberg', symmetry='SU2', spin=0.5, J=1.0),
        ('Spin', 'SU2', {'J': 0.5}),
        False,
    ),
    'heisenberg_s1': (
        dict(category='bosonic', label='Heisenberg', symmetry='U1', spin=1.0, J=1.0),
        ('Spin', 'U1', {'J': 1.0}),
        False,
    ),
    'majumdar_ghosh': (
        dict(category='bosonic', label='Heisenberg', symmetry='U1',
             spin=0.5, J=1.0, Jp=0.5),
        ('Spin', 'U1', {'J': 0.5}),
        True,
    ),
    'free_fermion': (
        dict(category='fermionic', label='FreeFermion', symmetry='U1', t=1.0),
        ('Ferm', 'U1'),
        False,
    ),
    'free_fermion_nnn': (
        dict(category='fermionic', label='FreeFermion', symmetry='U1',
             t=1.0, tp=0.3, mu=0.7),
        ('Ferm', 'U1'),
        True,
    ),
    'hubbard': (
        dict(category='conductor', label='Hubbard', symmetry='U1,U1', t=1.0, U=4.0),
        ('Band', 'U1,U1'),
        False,
    ),
}


def _geo_cfg(lx, bcx, n3x):
    return {'lattice': 'chain', 'lx': lx, 'bcx': bcx, 'n2x': True, 'n3x': n3x}


def _build_inf(key, N):
    """Build the infinite MPO for a named model with unit cell size `N`."""
    model, _, n3x = MODELS[key]
    intrs, spc, geo = build_interaction(
        {'geometry': _geo_cfg(N, 'IBC', n3x), 'model': model}
    )
    return build_ihamiltonian(intrs, geo.L, spc)


def _build_fin(key, L):
    """Build the finite open-boundary MPO for a named model on `L` sites."""
    model, _, n3x = MODELS[key]
    intrs, spc, geo = build_interaction(
        {'geometry': _geo_cfg(L, 'OBC', n3x), 'model': model}
    )
    return build_hamiltonian(intrs, geo.L, spc)


# ---------------------------------------------------------------------------
# Equivalence with the finite MPO
# ---------------------------------------------------------------------------

class TestWindowMatchesFinite:
    """A capped window of the infinite MPO is the finite open-chain MPO."""

    @pytest.mark.parametrize("key", list(MODELS))
    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_expectation_values_agree(self, key, N):
        """Window and finite MPO give identical expectation values."""
        L = 6
        model, space, _ = MODELS[key]
        window = window_mpo(_build_inf(key, N), L)
        finite = _build_fin(key, L)

        spc, ops = load_space(*space)
        for seed in (1, 2, 3):
            mps = init_mps(L, spc, ops, bond_dim=10, seed=seed)
            got = observe(mps, window)
            ref = observe(mps, finite)
            assert got == pytest.approx(ref, rel=1e-10, abs=1e-12)

    @pytest.mark.parametrize("L", [2, 3, 4, 7, 9])
    def test_window_length_independence(self, L):
        """Windows of any length reproduce the finite chain of that length."""
        window = window_mpo(_build_inf('heisenberg_u1', 2), L)
        finite = _build_fin('heisenberg_u1', L)

        spc, ops = load_space('Spin', 'U1', {'J': 0.5})
        mps = init_mps(L, spc, ops, bond_dim=10, seed=7)
        assert observe(mps, window) == pytest.approx(observe(mps, finite), abs=1e-12)

    def test_window_shorter_than_unit_cell(self):
        """A window may be shorter than the unit cell it was cut from."""
        window = window_mpo(_build_inf('heisenberg_u1', 4), 2)
        finite = _build_fin('heisenberg_u1', 2)

        spc, ops = load_space('Spin', 'U1', {'J': 0.5})
        mps = init_mps(2, spc, ops, bond_dim=4, seed=3)
        assert observe(mps, window) == pytest.approx(observe(mps, finite), abs=1e-12)

    @pytest.mark.parametrize("N", [1, 2, 3, 4, 6])
    def test_unit_cell_size_independence(self, N):
        """The operator does not depend on how the unit cell was chosen."""
        L = 8
        window = window_mpo(_build_inf('heisenberg_u1', N), L)
        finite = _build_fin('heisenberg_u1', L)

        spc, ops = load_space('Spin', 'U1', {'J': 0.5})
        mps = init_mps(L, spc, ops, bond_dim=10, seed=11)
        assert observe(mps, window) == pytest.approx(observe(mps, finite), abs=1e-12)


# ---------------------------------------------------------------------------
# Cell tiling and structure
# ---------------------------------------------------------------------------

class TestCellStructure:
    """Shape, itags and bond matching across the unit cell."""

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_is_impo_of_right_length(self, N):
        """The builder returns an iMPO with one tensor per unit-cell site."""
        impo = _build_inf('heisenberg_u1', N)
        assert isinstance(impo, iMPO)
        assert impo.N == N
        assert len(impo.bond_dims) == N

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_cell_boundary_bond_matches(self, N):
        """The last tensor's right index is dual to the first tensor's left."""
        impo = _build_inf('heisenberg_u1', N)
        right = impo[N - 1].indices[1]
        left  = impo[0].indices[0]
        assert right.flip() == left
        assert impo[N - 1].itags[1] == impo[0].itags[0]

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_bond_itags_wrap(self, N):
        """Bond itags are numbered modulo the unit-cell size."""
        impo = _build_inf('heisenberg_u1', N)
        ndigits = max(2, len(str(N)))
        for i in range(N):
            assert impo[i].itags[0] == f'W{i:0{ndigits}d}'
            assert impo[i].itags[1] == f'W{(i + 1) % N:0{ndigits}d}'

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_physical_itags(self, N):
        """Both physical axes of a site carry that site's physical itag."""
        impo = _build_inf('heisenberg_u1', N)
        for i in range(N):
            assert impo[i].itags[2] == f's{i:02d}'
            assert impo[i].itags[3] == f's{i:02d}'

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_bond_dim_uniform_for_uniform_model(self, N):
        """A translationally invariant model has one bond dimension."""
        impo = _build_inf('heisenberg_u1', N)
        assert len(set(impo.bond_dims)) == 1

    def test_nearest_neighbor_bond_dim(self):
        """A single NN interaction with 3 operators gives bond dimension 5."""
        # One start channel, three operator channels (S+, Sz, S-), one finish.
        assert _build_inf('heisenberg_u1', 2).bond_dims == [5, 5]

    def test_su2_bond_dim(self):
        """Under SU(2) the three spin channels fuse into one spin-1 channel."""
        assert _build_inf('heisenberg_su2', 2).bond_dims == [3, 3]

    def test_next_nearest_neighbor_bond_dim(self):
        """NNN terms add a second relay stage of operator channels."""
        # start + 3 (NN) + 3 + 3 (NNN, two stages) + finish.
        assert _build_inf('majumdar_ghosh', 2).bond_dims == [11, 11]


# ---------------------------------------------------------------------------
# Charge conservation
# ---------------------------------------------------------------------------

class TestChargeConservation:
    """Symmetry structure of the assembled tensors."""

    @pytest.mark.parametrize("key", list(MODELS))
    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_every_block_is_admissible(self, key, N):
        """Nicole validates charge conservation, so construction must succeed."""
        impo = _build_inf(key, N)
        for tensor in impo:
            assert len(tensor.data) > 0

    @pytest.mark.parametrize("key", ['heisenberg_u1', 'free_fermion', 'hubbard'])
    def test_boundary_channels_are_in_the_vacuum_sector(self, key):
        """Start and finish channels carry no charge, so a window is neutral."""
        impo = _build_inf(key, 2)
        start, finish = boundary_vectors(impo)
        # One admissible block each: the trivial cap forces the vacuum sector.
        assert len(start.data) == 1
        assert len(finish.data) == 1

    def test_bond_sectors_match_across_the_cell(self):
        """Sectors agree on the bond onto the next cell, so the cell tiles."""
        impo = _build_inf('hubbard', 3)
        for i in range(impo.N):
            j = (i + 1) % impo.N
            assert impo[i].indices[1].flip() == impo[j].indices[0]


# ---------------------------------------------------------------------------
# Boundary vectors
# ---------------------------------------------------------------------------

class TestBoundaryVectors:
    """Selector vectors that cap the automaton at a chosen bond."""

    def test_shapes_and_itags(self):
        """Start leads with the cap axis; finish trails with it."""
        impo = _build_inf('heisenberg_u1', 2)
        start, finish = boundary_vectors(impo, bond=0)
        assert start.itags == ('_cap_', impo[0].itags[0])
        assert finish.itags == (impo[1].itags[1], '_cap_')

    @pytest.mark.parametrize("bond", [0, 1, 2, 3, -1])
    def test_bond_taken_modulo_unit_cell(self, bond):
        """Any bond index is valid; it is taken modulo the unit cell."""
        impo = _build_inf('heisenberg_u1', 2)
        start, finish = boundary_vectors(impo, bond=bond)
        assert start.itags[1] == impo[bond % 2].itags[0]

    def test_single_unit_entry(self):
        """Each selector picks exactly one automaton channel."""
        impo = _build_inf('heisenberg_u1', 2)
        for vec in boundary_vectors(impo):
            total = sum(float(blk.abs().sum()) for blk in vec.data.values())
            assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Rejections for inputs the automaton cannot express."""

    def _spc(self):
        _, spc, _ = build_interaction({
            'geometry': _geo_cfg(2, 'IBC', False),
            'model': MODELS['heisenberg_u1'][0],
        })
        return spc

    def _intrs(self, N=2):
        intrs, _, _ = build_interaction({
            'geometry': _geo_cfg(N, 'IBC', False),
            'model': MODELS['heisenberg_u1'][0],
        })
        return intrs

    def test_leading_site_outside_unit_cell(self):
        """A leading site must index into the unit cell."""
        intrs = self._intrs()
        intrs[0].leading_site = 5
        with pytest.raises(ValueError, match='leading_site'):
            build_ihamiltonian(intrs, 2, self._spc())

    def test_non_positive_span(self):
        """An interaction must reach forward by at least one site."""
        intrs = self._intrs()
        intrs[0].terminal_site = intrs[0].leading_site
        with pytest.raises(ValueError, match='span'):
            build_ihamiltonian(intrs, 2, self._spc())

    def test_n_site_interaction_rejected(self):
        """Interactions beyond two sites are not supported yet."""
        intrs = self._intrs()
        intrs.append(InteractionNSite(label=['X'], cpl=1.0, sites=[0, 1, 2, 3]))
        with pytest.raises(NotImplementedError, match='InteractionNSite'):
            build_ihamiltonian(intrs, 2, self._spc())

    def test_missing_tensors_rejected(self):
        """An active interaction must carry its operator tensors."""
        intrs = self._intrs()
        intrs[0].leading_tnsr = None
        with pytest.raises(ValueError, match='leading_tnsr'):
            build_ihamiltonian(intrs, 2, self._spc())

    def test_zero_coupling_interactions_are_dropped(self):
        """A zero coupling contributes no channel, shrinking the bond."""
        intrs = self._intrs()
        for intr in intrs:
            intr.cpl = 0.0
        impo = build_ihamiltonian(intrs, 2, self._spc())
        # Only the start and finish channels survive.
        assert impo.bond_dims == [2, 2]

    def test_identity_only_mpo_measures_norm(self):
        """With no interactions the MPO is the identity on every window."""
        intrs = self._intrs()
        for intr in intrs:
            intr.cpl = 0.0
        impo = build_ihamiltonian(intrs, 2, self._spc())

        spc, ops = load_space('Spin', 'U1', {'J': 0.5})
        mps = init_mps(6, spc, ops, bond_dim=8, seed=5)
        assert observe(mps, window_mpo(impo, 6)) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Long-range interactions
# ---------------------------------------------------------------------------

class TestLongRange:
    """Interactions reaching several unit cells ahead.

    The chain geometry only emits ranges 1 and 2, so these build their own
    interaction lists by stretching a next-nearest-neighbor template, which
    already carries the `intermid_tnsr` a multi-bond string needs.
    """

    # Heisenberg with a J' term, so the geometry emits range-2 bonds.
    _MODEL = MODELS['majumdar_ghosh'][0]

    def _template(self, L):
        """A range-2 interaction with all three operator tensors populated."""
        intrs, spc, _ = build_interaction(
            {'geometry': _geo_cfg(L, 'OBC', True), 'model': self._MODEL}
        )
        nnn = [i for i in intrs if 'NNN' in i.label]
        return nnn[0], spc

    def _stretch(self, template, sites, span):
        """Copy `template` onto each leading site, reaching `span` sites ahead."""
        copies = []
        for site in sites:
            intr = Interaction2Site(
                label=list(template.label),
                leading_site=site,
                terminal_site=site + span,
                cpl=template.cpl,
            )
            intr.leading_tnsr  = template.leading_tnsr
            intr.terminal_tnsr = template.terminal_tnsr
            intr.intermid_tnsr = template.intermid_tnsr
            copies.append(intr)
        return copies

    @pytest.mark.parametrize("span", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_long_range_term_matches_finite(self, span, N):
        """Interactions of any span reproduce their finite counterpart."""
        L = 8
        template, spc = self._template(L)

        impo = build_ihamiltonian(
            self._stretch(template, range(N), span), N, spc
        )
        finite = build_hamiltonian(
            self._stretch(template, range(L - span), span), L, spc
        )

        pspc, ops = load_space(*MODELS['majumdar_ghosh'][1])
        mps = init_mps(L, pspc, ops, bond_dim=10, seed=13)
        assert observe(mps, window_mpo(impo, L)) == pytest.approx(
            observe(mps, finite), abs=1e-11
        )

    @pytest.mark.parametrize("span", [1, 2, 3, 4])
    def test_span_widens_the_bond(self, span):
        """A span of d holds the operator string across d bonds."""
        template, spc = self._template(8)
        impo = build_ihamiltonian(self._stretch(template, [0], span), 1, spc)
        # Start and finish, plus the 3-dimensional spin channel on each of the
        # d bonds the string has to cross.
        assert impo.bond_dims == [2 + 3 * span]
