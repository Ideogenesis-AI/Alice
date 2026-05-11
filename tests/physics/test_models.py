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


"""Tests for the Hamiltonian model builders in `alice.physics.models`."""

from __future__ import annotations

import logging

import pytest

from alice.network.interaction import Interaction1Site, Interaction2Site
from alice.physics.geometry import build_geometry
from alice.physics.square import intrcmap_square
from alice.physics.models import build_heisenberg, build_free_fermion, build_hubbard
from alice.physics.system import build_bosonic, build_fermionic, build_conductor


# Suppress INFO-level geometry logs during tests.
@pytest.fixture(autouse=True)
def _quiet_geometry(caplog):
    with caplog.at_level(logging.WARNING, logger='alice.physics.geometry'):
        yield


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _nn_chain(L: int) -> list:
    """Return L-1 nearest-neighbor Interaction2Site objects for a 1D chain."""
    geo = build_geometry({
        'lattice': 'square',
        'lx': L, 'ly': 1,
        'bcx': 'OBC', 'bcy': 'OBC',
        'n2x': True, 'n2y': False,
    })
    return intrcmap_square(geo)


def _nn_nnn_chain(L: int) -> list:
    """Return NN + NNN interactions for a 1D chain of length L.

    NNN bonds span 2 MPS sites, so terminal_site == leading_site + 2.
    Generated as a 1xL... actually NNN doesn't make sense for ly=1.
    Use a 2-row lattice to get both NN and NNN bonds from the geometry.
    """
    geo = build_geometry({
        'lattice': 'square',
        'lx': L // 2, 'ly': 2,
        'bcx': 'OBC', 'bcy': 'OBC',
        'n2x': True, 'n2y': True,
        'n3d': True, 'n3o': True,
    })
    return intrcmap_square(geo)


def _synthetic(label: str, gap: int) -> Interaction2Site:
    """Return a single synthetic `Interaction2Site` with the given label and gap."""
    return Interaction2Site(label=[label], leading_site=0, terminal_site=gap)


# ---------------------------------------------------------------------------
# build_heisenberg
# ---------------------------------------------------------------------------

class TestBuildHeisenberg:
    """Tests for `build_heisenberg`."""

    def test_returns_spc_and_ops(self):
        """Builder returns a (spc, ops) tuple."""
        interactions = _nn_chain(6)
        spc, ops = build_heisenberg(interactions)
        assert spc is not None
        assert isinstance(ops, dict)

    def test_nn_cpl_set(self):
        """NN interactions get cpl == J."""
        J = 1.5
        interactions = _nn_chain(6)
        build_heisenberg(interactions, J=J)
        for intr in interactions:
            assert intr.cpl == J

    def test_nn_tensors_populated(self):
        """NN interactions get non-None leading and terminal tensors."""
        interactions = _nn_chain(6)
        build_heisenberg(interactions, J=1.0)
        for intr in interactions:
            assert isinstance(intr, Interaction2Site)
            assert intr.leading_tnsr  is not None
            assert intr.terminal_tnsr is not None

    def test_tensors_not_scaled_by_cpl(self):
        """Terminal tensor norm is independent of J (coupling not baked in)."""
        interactions_a = _nn_chain(4)
        interactions_b = _nn_chain(4)
        build_heisenberg(interactions_a, J=1.0)
        build_heisenberg(interactions_b, J=5.0)
        # Tensor norms should be the same regardless of coupling.
        for a, b in zip(interactions_a, interactions_b):
            assert abs(a.terminal_tnsr.norm() - b.terminal_tnsr.norm()) < 1e-12, (
                "terminal_tnsr norm differs between J=1.0 and J=5.0; "
                "coupling appears to be baked into the tensor."
            )

    def test_su2_symmetry(self):
        """Builder works with SU2 symmetry."""
        interactions = _nn_chain(4)
        spc, ops = build_heisenberg(interactions, symmetry='SU2', spin=0.5)
        assert spc is not None
        for intr in interactions:
            assert intr.leading_tnsr is not None

    @pytest.mark.parametrize("spin", [0.5, 1.0])
    def test_spin_parameter(self, spin):
        """Builder accepts different spin quantum numbers."""
        interactions = _nn_chain(4)
        spc, ops = build_heisenberg(interactions, spin=spin)
        assert spc is not None

    def test_nnn_cpl_zero_leaves_tensors_none(self):
        """NNN bonds with Jp=0 leave tensors as None (skipped by autompo)."""
        interactions = _nn_nnn_chain(6)
        build_heisenberg(interactions, J=1.0, Jp=0.0)

        nnn = [i for i in interactions if 'NNN' in i.label]
        assert len(nnn) > 0
        for intr in nnn:
            assert intr.cpl == 0.0
            assert intr.leading_tnsr  is None
            assert intr.terminal_tnsr is None

    def test_nnn_cpl_nonzero_adjacent(self):
        """NNN bonds with Jp != 0 and terminal_site == leading_site + 1 get tensors."""
        # Construct a synthetic adjacent NNN interaction to test the non-zero path.
        intr = Interaction2Site(label=['NNN'], leading_site=0, terminal_site=1)
        build_heisenberg([intr], Jp=0.3)
        assert intr.cpl == pytest.approx(0.3)
        assert intr.leading_tnsr  is not None
        assert intr.terminal_tnsr is not None

    def test_space_fn_override(self):
        """Custom space_fn replaces build_bosonic."""
        called = {}

        def custom_space(symmetry, spin):
            called['symmetry'] = symmetry
            called['spin'] = spin
            return build_bosonic(symmetry, spin)

        interactions = _nn_chain(4)
        build_heisenberg(interactions, symmetry='U1', spin=0.5, space_fn=custom_space)
        assert called['symmetry'] == 'U1'
        assert called['spin'] == 0.5

    def test_adjacent_nn_no_intermid_tnsr(self):
        """Adjacent NN bonds (gap=1) must leave `intermid_tnsr` as `None`."""
        intr = _synthetic('NN', gap=1)
        build_heisenberg([intr], J=1.0)
        assert intr.intermid_tnsr is None

    def test_long_range_nn_has_intermid_tnsr(self):
        """Long-range NN bonds (gap > 1) must have `intermid_tnsr` set."""
        intr = _synthetic('NN', gap=5)
        build_heisenberg([intr], J=1.0)
        assert intr.intermid_tnsr is not None

    def test_long_range_nn_intermid_tnsr_axis_count(self):
        """`intermid_tnsr` on a long-range NN bond must have exactly 4 indices."""
        intr = _synthetic('NN', gap=5)
        build_heisenberg([intr], J=1.0)
        assert len(intr.intermid_tnsr.indices) == 4

    def test_long_range_nnn_has_intermid_tnsr(self):
        """Long-range NNN bonds with `Jp != 0` must have `intermid_tnsr` set."""
        intr = _synthetic('NNN', gap=5)
        build_heisenberg([intr], Jp=0.3)
        assert intr.intermid_tnsr is not None

    def test_long_range_nnn_zero_jp_no_intermid_tnsr(self):
        """Long-range NNN bonds with `Jp=0` must leave `intermid_tnsr` as `None`."""
        intr = _synthetic('NNN', gap=5)
        build_heisenberg([intr], Jp=0.0)
        assert intr.intermid_tnsr is None


# ---------------------------------------------------------------------------
# build_free_fermion
# ---------------------------------------------------------------------------

class TestBuildFreeFermion:
    """Tests for `build_free_fermion`."""

    def test_returns_spc_and_ops(self):
        """Builder returns a (spc, ops) tuple."""
        interactions = _nn_chain(6)
        spc, ops = build_free_fermion(interactions)
        assert spc is not None

    def test_nn_cpl_is_negative_t(self):
        """NN interactions get cpl == -t (hopping sign convention)."""
        t = 1.0
        interactions = _nn_chain(6)
        build_free_fermion(interactions, t=t)
        for intr in interactions:
            assert intr.cpl == pytest.approx(-t)

    def test_nn_tensors_populated(self):
        """NN interactions get non-None tensors."""
        interactions = _nn_chain(6)
        build_free_fermion(interactions, t=1.0)
        for intr in interactions:
            assert intr.leading_tnsr  is not None
            assert intr.terminal_tnsr is not None

    def test_tensors_not_scaled_by_cpl(self):
        """Terminal tensor norm is independent of t."""
        interactions_a = _nn_chain(4)
        interactions_b = _nn_chain(4)
        build_free_fermion(interactions_a, t=1.0)
        build_free_fermion(interactions_b, t=3.0)
        for a, b in zip(interactions_a, interactions_b):
            assert abs(a.terminal_tnsr.norm() - b.terminal_tnsr.norm()) < 1e-12

    def test_mu_appends_onsite_interactions(self):
        """Non-zero mu appends L on-site Interaction1Site objects tagged 'mu'."""
        L = 6
        interactions = _nn_chain(L)
        build_free_fermion(interactions, L, mu=0.5)

        mu_intrs = [i for i in interactions if isinstance(i, Interaction1Site)
                    and 'mu' in i.label]
        assert len(mu_intrs) == L

    def test_mu_cpl(self):
        """Chemical potential Interaction1Site carries cpl == -mu."""
        L = 4
        mu = 0.7
        interactions = _nn_chain(L)
        build_free_fermion(interactions, L, mu=mu)

        mu_intrs = [i for i in interactions if isinstance(i, Interaction1Site)
                    and 'mu' in i.label]
        for intr in mu_intrs:
            assert intr.cpl == pytest.approx(-mu)

    def test_mu_zero_no_onsite(self):
        """mu=0 produces no chemical-potential Interaction1Site objects."""
        L = 4
        interactions = _nn_chain(L)
        build_free_fermion(interactions, L, mu=0.0)

        mu_intrs = [i for i in interactions if isinstance(i, Interaction1Site)
                    and 'mu' in i.label]
        assert len(mu_intrs) == 0

    def test_space_fn_override(self):
        """Custom space_fn replaces build_fermionic."""
        called = {}

        def custom_space(symmetry):
            called['symmetry'] = symmetry
            return build_fermionic(symmetry)

        interactions = _nn_chain(4)
        build_free_fermion(interactions, symmetry='U1', space_fn=custom_space)
        assert called['symmetry'] == 'U1'

    def test_nnn_zero_tp_leaves_tensors_none(self):
        """NNN bonds with tp=0 leave tensors as None."""
        interactions = _nn_nnn_chain(6)
        build_free_fermion(interactions, t=1.0, tp=0.0)

        nnn = [i for i in interactions if 'NNN' in i.label]
        assert len(nnn) > 0
        for intr in nnn:
            assert intr.cpl == 0.0
            assert intr.leading_tnsr  is None

    def test_adjacent_nn_no_intermid_tnsr(self):
        """Adjacent NN bonds (gap=1) must leave `intermid_tnsr` as `None`."""
        intr = _synthetic('NN', gap=1)
        build_free_fermion([intr], t=1.0)
        assert intr.intermid_tnsr is None

    def test_long_range_nn_has_intermid_tnsr(self):
        """Long-range NN bonds (gap > 1) must have `intermid_tnsr` set."""
        intr = _synthetic('NN', gap=5)
        build_free_fermion([intr], t=1.0)
        assert intr.intermid_tnsr is not None

    def test_long_range_nn_intermid_tnsr_axis_count(self):
        """`intermid_tnsr` on a long-range NN bond must have exactly 4 indices."""
        intr = _synthetic('NN', gap=5)
        build_free_fermion([intr], t=1.0)
        assert len(intr.intermid_tnsr.indices) == 4


# ---------------------------------------------------------------------------
# build_hubbard
# ---------------------------------------------------------------------------

class TestBuildHubbard:
    """Tests for `build_hubbard`."""

    def test_returns_spc_and_ops(self):
        """Builder returns a (spc, ops) tuple."""
        L = 6
        interactions = _nn_chain(L)
        spc, ops = build_hubbard(interactions, L)
        assert spc is not None

    def test_nn_hopping_cpl(self):
        """NN 2-site interactions get cpl == -t."""
        L = 6
        t = 2.0
        interactions = _nn_chain(L)
        build_hubbard(interactions, L, t=t)

        nn = [i for i in interactions if isinstance(i, Interaction2Site) and 'NN' in i.label]
        assert len(nn) == L - 1
        for intr in nn:
            assert intr.cpl == pytest.approx(-t)

    def test_nn_tensors_populated(self):
        """NN hopping interactions have non-None tensors."""
        L = 6
        interactions = _nn_chain(L)
        build_hubbard(interactions, L)

        nn = [i for i in interactions if isinstance(i, Interaction2Site)]
        for intr in nn:
            assert intr.leading_tnsr  is not None
            assert intr.terminal_tnsr is not None

    def test_U_interactions_appended(self):
        """Exactly L Hubbard-U Interaction1Site objects (labelled 'U') are appended."""
        L = 6
        interactions = _nn_chain(L)
        build_hubbard(interactions, L, U=4.0, mu=0.0)

        u_intrs = [i for i in interactions
                   if isinstance(i, Interaction1Site) and 'U' in i.label]
        assert len(u_intrs) == L

    def test_U_cpl_equals_U(self):
        """Hubbard-U Interaction1Site objects carry cpl == U."""
        L = 4
        U = 3.5
        interactions = _nn_chain(L)
        build_hubbard(interactions, L, U=U, mu=0.0)

        u_intrs = [i for i in interactions
                   if isinstance(i, Interaction1Site) and 'U' in i.label]
        for intr in u_intrs:
            assert intr.cpl == pytest.approx(U)

    def test_onsite_tensor_populated(self):
        """All on-site Interaction1Site objects have non-None tnsr."""
        L = 4
        interactions = _nn_chain(L)
        build_hubbard(interactions, L)

        onsite = [i for i in interactions if isinstance(i, Interaction1Site)]
        for intr in onsite:
            assert intr.tnsr is not None

    def test_U_tensors_not_scaled_by_U(self):
        """Hubbard-U tensor norm is independent of U (coupling not baked in)."""
        L = 4
        interactions_a = _nn_chain(L)
        interactions_b = _nn_chain(L)
        # Use mu=0 and U values that give the same mu_eff so only U terms differ.
        build_hubbard(interactions_a, L, U=1.0, mu=0.0)
        build_hubbard(interactions_b, L, U=8.0, mu=0.0)

        u_a = [i for i in interactions_a
               if isinstance(i, Interaction1Site) and 'U' in i.label]
        u_b = [i for i in interactions_b
               if isinstance(i, Interaction1Site) and 'U' in i.label]
        for a, b in zip(u_a, u_b):
            assert abs(a.tnsr.norm() - b.tnsr.norm()) < 1e-12

    def test_U_sites_cover_all_sites(self):
        """Hubbard-U Interaction1Site objects cover every site 0 to L-1."""
        L = 6
        interactions = _nn_chain(L)
        build_hubbard(interactions, L)

        sites = sorted(i.site for i in interactions
                       if isinstance(i, Interaction1Site) and 'U' in i.label)
        assert sites == list(range(L))

    def test_mu_half_filling_shift(self):
        """At mu=0, effective chemical potential equals U/2 (half-filling shift)."""
        L = 4
        U = 4.0
        interactions = _nn_chain(L)
        build_hubbard(interactions, L, U=U, mu=0.0)

        mu_intrs = [i for i in interactions if isinstance(i, Interaction1Site)
                    and 'mu' in i.label]
        # mu_eff = mu + U/2 = 0 + 2 = 2; cpl = -mu_eff = -2
        assert len(mu_intrs) == L
        for intr in mu_intrs:
            assert intr.cpl == pytest.approx(-(U / 2))

    def test_mu_relative_to_half_filling(self):
        """cpl on chemical-potential terms equals -(mu + U/2)."""
        L = 4
        U = 4.0
        mu = 0.5
        interactions = _nn_chain(L)
        build_hubbard(interactions, L, U=U, mu=mu)

        mu_intrs = [i for i in interactions if isinstance(i, Interaction1Site)
                    and 'mu' in i.label]
        for intr in mu_intrs:
            assert intr.cpl == pytest.approx(-(mu + U / 2))

    def test_mu_zero_U_zero_no_mu_onsite(self):
        """When mu=0 and U=0, mu_eff=0 so no chemical-potential term is added."""
        L = 4
        interactions = _nn_chain(L)
        build_hubbard(interactions, L, U=0.0, mu=0.0)

        mu_intrs = [i for i in interactions if isinstance(i, Interaction1Site)
                    and 'mu' in i.label]
        assert len(mu_intrs) == 0

    def test_mu_tensor_not_scaled(self):
        """Chemical-potential tensor norm is independent of mu_eff."""
        L = 4
        interactions_a = _nn_chain(L)
        interactions_b = _nn_chain(L)
        build_hubbard(interactions_a, L, U=2.0, mu=0.0)
        build_hubbard(interactions_b, L, U=2.0, mu=1.0)

        mu_a = [i for i in interactions_a if isinstance(i, Interaction1Site)
                and 'mu' in i.label]
        mu_b = [i for i in interactions_b if isinstance(i, Interaction1Site)
                and 'mu' in i.label]
        for a, b in zip(mu_a, mu_b):
            assert abs(a.tnsr.norm() - b.tnsr.norm()) < 1e-12

    def test_space_fn_override(self):
        """Custom space_fn replaces build_conductor."""
        called = {}

        def custom_space(symmetry):
            called['symmetry'] = symmetry
            return build_conductor(symmetry)

        L = 4
        interactions = _nn_chain(L)
        build_hubbard(interactions, L, symmetry='U1,U1', space_fn=custom_space)
        assert called['symmetry'] == 'U1,U1'

    def test_adjacent_nn_no_intermid_tnsr(self):
        """Adjacent NN bonds (gap=1) must leave `intermid_tnsr` as `None`."""
        intr = _synthetic('NN', gap=1)
        build_hubbard([intr], L=1, t=1.0)
        assert intr.intermid_tnsr is None

    def test_long_range_nn_has_intermid_tnsr(self):
        """Long-range NN bonds (gap > 1) must have `intermid_tnsr` set."""
        intr = _synthetic('NN', gap=5)
        build_hubbard([intr], L=1, t=1.0)
        assert intr.intermid_tnsr is not None

    def test_long_range_nn_intermid_tnsr_axis_count(self):
        """`intermid_tnsr` on a long-range NN bond must have exactly 4 indices."""
        intr = _synthetic('NN', gap=5)
        build_hubbard([intr], L=1, t=1.0)
        assert len(intr.intermid_tnsr.indices) == 4
