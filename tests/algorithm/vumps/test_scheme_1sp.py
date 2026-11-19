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


"""Tests for the bond expansion and the one-site split that follows it."""

from __future__ import annotations

import pytest
from nicole import Tensor, einsum

from alice.algorithm.vumps.complement import (
    _center_isometry_left,
    _has_complement,
    _noise_floor,
    discarded_weight,
    expand_bond,
    split_center,
)
from alice.algorithm.vumps.davidson import _inner_product
from alice.algorithm.vumps.scheme_1s import matvec_center, optimize_center
from alice.network import iMPS

from .conftest import (
    HEISENBERG,
    _bond,
    bond_environments,
    ihamiltonian,
    random_imps,
)

DAVIDSON = {'max_iter': 100, 'tol': 1e-10, 'max_subspace': 20}


def _setup(imps):
    """Return the iMPO and the environment blocks of a gauge-fixed state.

    Both block lists are indexed by bond, so the step that widens bond 1
    reads the left block on bond 0 and the right one on bond 2.
    """
    impo = ihamiltonian(HEISENBERG, imps.N)
    lefts, rights = bond_environments(imps, impo)
    return impo, lefts, rights


def _expand(imps, k_expand=4, alpha=None):
    """Widen bond 1 of a gauge-fixed cell and return everything in play.

    Bond 1 is met with the center tensor of site 0 behind it and the
    right-orthonormal tensor of site 1 in front, which is the geometry
    every step of the pass sees.
    """
    impo, lefts, rights = _setup(imps)
    center = imps.center_tensor(0)
    right_iso = imps.right_isometries()[1]
    widened = expand_bond(
        center, right_iso, impo[0], impo[1],
        lefts[0], rights[2 % imps.N], k_expand, alpha,
    )
    return (center, right_iso, impo, lefts, rights) + widened


def _saturated_imps(Spc):
    """Build a two-site cell whose bond 1 is as wide as it can be.

    A spin-1/2 site shifts the bond charge by one, so a bond carrying both
    charges adjacent to a one-dimensional neighbor spans the entire
    (bond, physical) space of that neighbor. Nothing is left over, so the
    complement of bond 1 is empty by construction, not by accident.
    """
    trivial = _bond(Spc.group, (0,))
    both = _bond(Spc.group, (-1, 1))
    imps = iMPS([
        Tensor.random(
            [trivial, both.flip(), Spc], seed=1, itags=['A00', 'A01', 's00'],
        ),
        Tensor.random(
            [both, trivial.flip(), Spc], seed=2, itags=['A01', 'A00', 's01'],
        ),
    ])
    imps.canonical()
    return imps


# ---------------------------------------------------------------------------
# Bond expansion
# ---------------------------------------------------------------------------

class TestExpansion:
    """The expansion must widen the bond without moving the state."""

    @pytest.mark.parametrize('N', [2, 4])
    def test_the_state_is_unchanged(self, spin_half, N):
        # The whole scheme rests on this: the solve that follows starts from
        # the state the iteration began with, merely described on a wider
        # bond, so the new directions enter with zero amplitude.
        imps = random_imps(spin_half, N, width=1, dim=1)
        center, right_iso, *_, center_exp, right_iso_exp, _ = _expand(imps)
        before = einsum('abr,bcs->acrs', center, right_iso)
        after = einsum('abr,bcs->acrs', center_exp, right_iso_exp)
        assert float((after - before).norm()) < 1e-12

    def test_the_bond_grows_by_at_most_what_was_asked_for(self, spin_half):
        imps = random_imps(spin_half, 2, width=1, dim=1)
        before = imps.bond_matrix(1).indices[0].dim
        _, _, *_, right_iso_exp, _ = _expand(imps, k_expand=3)
        after = right_iso_exp.indices[0].dim
        assert before < after <= before + 3

    def test_the_neighbor_stays_orthonormal(self, spin_half):
        # `oplus` widens the bond with directions the projection put in the
        # complement of what the neighbor already spans. Were they not, the
        # widened neighbor would stop being an isometry and the fill-in that
        # preserves the state would stop being exact.
        imps = random_imps(spin_half, 2, width=1, dim=1)
        _, _, *_, right_iso_exp, _ = _expand(imps)
        overlap = einsum('acr,bcr->ab', right_iso_exp.conj(), right_iso_exp)
        assert float((overlap - einsum('ab,bc->ac', overlap, overlap)).norm()) < 1e-10

    def test_the_center_basis_spans_the_center_tensor(self, spin_half):
        # Both halves of that matter. Spanning less would leave kept
        # directions in the complement, which `oplus` would then duplicate;
        # spanning more would project away the complement itself.
        imps = random_imps(spin_half, 2)
        center = imps.center_tensor(0)
        basis = _center_isometry_left(center)
        remainder = center - einsum('xcz,xbz,abr->acr', center, basis.conj(), basis)
        assert float(remainder.norm()) < 1e-10
        assert basis.indices[1].dim == center.indices[1].dim

    def test_a_saturated_bond_is_left_alone(self, spin_half):
        # There is nothing to expand into, and the SVD cannot say so: handed
        # a vanishing block it returns a unit vector lying in the kept space,
        # which `oplus` would duplicate. The noise floor catches it.
        imps = _saturated_imps(spin_half)
        center, right_iso, *_, center_exp, right_iso_exp, env_right_exp = _expand(imps)
        assert center_exp is center
        assert right_iso_exp is right_iso
        # The environment still has to be the one the plain one-site step
        # would have run against, since the caller has no other source.
        assert env_right_exp.indices[2].dim == center.indices[1].dim

    def test_the_floor_rejects_a_vanishing_complement(self, spin_half):
        imps = random_imps(spin_half, 2, width=1, dim=1)
        _, _, impo, lefts, rights, *_ = _expand(imps)
        center = imps.center_tensor(0)
        half = matvec_center(center, impo[0], lefts[0], rights[1])
        floor = _noise_floor(half, half)
        assert floor > 0.0
        assert not _has_complement(center - center, floor)
        assert _has_complement(center, floor)

    def test_a_cheaper_expansion_still_widens_the_bond(self, spin_half):
        # `alpha` truncates the two-site tensor the iMPO is applied to, which
        # keeps the expansion cheap against the solve it precedes.
        imps = random_imps(spin_half, 2, width=1, dim=1)
        before = imps.bond_matrix(1).indices[0].dim
        _, _, *_, right_iso_exp, _ = _expand(imps, k_expand=2, alpha=1)
        assert right_iso_exp.indices[0].dim > before


# ---------------------------------------------------------------------------
# What the widened bond buys
# ---------------------------------------------------------------------------

class TestSolveOnTheWidenedBond:
    """The point of widening: a lower Ritz value than the bond allowed."""

    def test_the_widened_solve_cannot_do_worse(self, spin_half):
        # The expanded space contains the old one, so its lowest eigenvalue
        # is bounded by the one the unwidened bond admits. A complement drawn
        # from the kept space, or an expansion that moved the state, would
        # both show up here.
        imps = random_imps(spin_half, 2, width=1, dim=1)
        center, _, impo, lefts, rights, center_exp, _, env_right_exp = _expand(imps)

        plain, _, _ = optimize_center(
            center, impo[0], lefts[0], rights[1], DAVIDSON,
        )
        widened, _, _ = optimize_center(
            center_exp, impo[0], lefts[0], env_right_exp, DAVIDSON,
        )
        assert widened <= plain + 1e-10

    def test_the_expanded_operator_is_hermitian(self, spin_half):
        # The expanded environment is one exact step over the widened
        # neighbor and not two blocks stacked block-diagonally, which keeps
        # the cross terms and so the operator symmetric.
        imps = random_imps(spin_half, 2, width=1, dim=1)
        _, _, impo, lefts, _, center_exp, _, env_right_exp = _expand(imps)
        u = center_exp
        v = matvec_center(center_exp, impo[0], lefts[0], env_right_exp)
        left = _inner_product(u, matvec_center(v, impo[0], lefts[0], env_right_exp))
        right = _inner_product(matvec_center(u, impo[0], lefts[0], env_right_exp), v)
        assert abs(left - right) < 1e-10 * max(1.0, abs(left))


# ---------------------------------------------------------------------------
# Splitting the widened bond back down
# ---------------------------------------------------------------------------

class TestSplit:
    """The split must be exact when nothing is cut, and lossy by a
    measured amount when something is."""

    @pytest.mark.parametrize('N', [2, 4])
    def test_untruncated_split_reconstructs_the_center(self, spin_half, N):
        imps = random_imps(spin_half, N)
        center = imps.center_tensor(0)
        tag = imps.bond_matrix(1).itags[0]
        left_iso, bond, rotation = split_center(center, tag)
        back = einsum('abr,bc,cd->adr', left_iso, bond, rotation)
        assert float((back - center).norm()) < 1e-12

    def test_factors_are_orthonormal_and_the_bond_carries_the_rest(self, spin_half):
        imps = random_imps(spin_half, 2)
        center = imps.center_tensor(0)
        tag = imps.bond_matrix(1).itags[0]
        left_iso, bond, rotation = split_center(center, tag)

        overlap = einsum('abr,acr->bc', left_iso.conj(), left_iso)
        assert float((overlap - einsum('ab,bc->ac', overlap, overlap)).norm()) < 1e-10
        overlap = einsum('ab,cb->ac', rotation, rotation.conj())
        assert float((overlap - einsum('ab,bc->ac', overlap, overlap)).norm()) < 1e-10
        assert abs(float(bond.norm()) - 1.0) < 1e-10

    def test_the_split_cuts_the_widened_bond_back_down(self, spin_half):
        # The widened bond lives only for the length of one solve; what the
        # cell keeps is whatever the cap allows.
        imps = random_imps(spin_half, 2, width=1, dim=1)
        *_, center_exp, _, _ = _expand(imps)
        tag = imps.bond_matrix(1).itags[0]
        trunc = {'nkeep': 2}
        _, bond, _ = split_center(center_exp, tag, trunc)
        assert bond.indices[0].dim <= 2
        assert discarded_weight(center_exp, trunc) > 0.0

    def test_nothing_is_discarded_without_a_cap(self, spin_half):
        imps = random_imps(spin_half, 2)
        assert discarded_weight(imps.center_tensor(0), None) == 0.0
