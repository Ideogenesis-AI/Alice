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


"""Tests for the two-site eigenproblem and the split that follows it."""

from __future__ import annotations

import pytest
from nicole import einsum

from alice.algorithm.idmrg import cell_environments
from alice.algorithm.idmrg.davidson import _inner_product
from alice.algorithm.idmrg.scheme_1s import gauge_tensors
from alice.algorithm.idmrg.scheme_2s import (
    discarded_weight,
    matvec_two_site,
    optimize_two_site,
    split_two_site,
)

from .conftest import HEISENBERG, ihamiltonian, random_imps

DAVIDSON = {'max_iter': 100, 'tol': 1e-10, 'max_subspace': 20}


def _pair(imps, k=0):
    """Return the two-site tensor spanning sites `k` and `k+1`."""
    return einsum(
        'abr,bcs->acrs',
        imps.left_isometries()[k],
        imps.center_tensor((k + 1) % imps.N),
    )


def _setup(imps):
    """Return the iMPO and the environment blocks for a gauge-fixed state."""
    impo = ihamiltonian(HEISENBERG, imps.N)
    env_left, env_right, _ = cell_environments(imps, impo)
    return impo, env_left, env_right


# ---------------------------------------------------------------------------
# Effective Hamiltonian
# ---------------------------------------------------------------------------

class TestEffectiveHamiltonian:
    """The two-site operator must be Hermitian and shape-preserving."""

    @pytest.mark.parametrize('N', [2, 4])
    def test_layout_is_preserved(self, spin_half, N):
        imps = random_imps(spin_half, N)
        impo, env_left, env_right = _setup(imps)
        AC2 = _pair(imps)
        out = matvec_two_site(AC2, impo[0], impo[1], env_left[0], env_right[1])
        assert out.itags == AC2.itags
        assert [i.dim for i in out.indices] == [i.dim for i in AC2.indices]

    def test_operator_is_hermitian(self, spin_half):
        # Davidson projects onto a real symmetric matrix, so an operator
        # that is not Hermitian would be silently symmetrized and the
        # eigenvector would be wrong rather than merely inaccurate.
        imps = random_imps(spin_half, 2)
        impo, env_left, env_right = _setup(imps)
        u = _pair(imps)
        v = _pair(random_imps(spin_half, 2, seed=7))
        left = _inner_product(
            u, matvec_two_site(v, impo[0], impo[1], env_left[0], env_right[1]),
        )
        right = _inner_product(
            matvec_two_site(u, impo[0], impo[1], env_left[0], env_right[1]), v,
        )
        assert abs(left - right) < 1e-10 * max(1.0, abs(left))


# ---------------------------------------------------------------------------
# Local eigensolve
# ---------------------------------------------------------------------------

class TestLocalSolve:
    """The Davidson wrapper must return a normalized, improving vector."""

    def test_solve_does_not_raise_the_ritz_value(self, spin_half):
        imps = random_imps(spin_half, 2)
        impo, env_left, env_right = _setup(imps)
        AC2 = _pair(imps)
        before = _inner_product(
            AC2,
            matvec_two_site(AC2, impo[0], impo[1], env_left[0], env_right[1]),
        ).real
        energy, optimized, _ = optimize_two_site(
            AC2, impo[0], impo[1], env_left[0], env_right[1], DAVIDSON,
        )
        assert energy <= before + 1e-10
        assert abs(float(optimized.norm()) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

class TestSplit:
    """The split must be exact when nothing is cut, and lossy by a
    measured amount when something is."""

    @pytest.mark.parametrize('N', [2, 4])
    def test_untruncated_split_reconstructs_the_pair(self, spin_half, N):
        imps = random_imps(spin_half, N)
        AC2 = _pair(imps)
        tag = imps.bond_matrix(1).itags[0]
        left_iso, bond, right_iso = split_two_site(AC2, tag)
        back = einsum(
            'abr,bcs->acrs', einsum('abr,bc->acr', left_iso, bond), right_iso,
        )
        assert float((back - AC2).norm()) < 1e-12

    def test_split_widens_the_bond(self, spin_half):
        # The point of the scheme: the bipartition of a pair admits more
        # states than the bond between them carried.
        imps = random_imps(spin_half, 2, width=1, dim=1)
        AC2 = _pair(imps)
        tag = imps.bond_matrix(1).itags[0]
        _, bond, _ = split_two_site(AC2, tag)
        assert bond.indices[0].dim > imps.bond_matrix(1).indices[0].dim

    def test_factors_are_orthonormal_and_the_bond_carries_the_rest(self, spin_half):
        imps = random_imps(spin_half, 2)
        AC2 = _pair(imps)
        tag = imps.bond_matrix(1).itags[0]
        left_iso, bond, right_iso = split_two_site(AC2, tag)

        overlap = einsum('abr,acr->bc', left_iso.conj(), left_iso)
        assert float((overlap - einsum('ab,bc->ac', overlap, overlap)).norm()) < 1e-10
        overlap = einsum('bar,car->bc', right_iso.conj(), right_iso)
        assert float((overlap - einsum('ab,bc->ac', overlap, overlap)).norm()) < 1e-10
        assert abs(float(bond.norm()) - 1.0) < 1e-10

    def test_cap_is_obeyed_and_what_it_costs_is_reported(self, spin_half):
        imps = random_imps(spin_half, 2)
        AC2 = _pair(imps)
        tag = imps.bond_matrix(1).itags[0]
        trunc = {'nkeep': 2}
        _, bond, _ = split_two_site(AC2, tag, trunc)
        assert bond.indices[0].dim <= 2
        assert discarded_weight(AC2, trunc) > 0.0

    def test_nothing_is_discarded_without_a_cap(self, spin_half):
        imps = random_imps(spin_half, 2)
        assert discarded_weight(_pair(imps), None) == 0.0


# ---------------------------------------------------------------------------
# Handing the split back to the gauge step
# ---------------------------------------------------------------------------

class TestGaugeHandover:
    """The one-site gauge step must recognise the split for what it is."""

    def test_left_factor_comes_back_unchanged(self, spin_half):
        # `AL = U_l(U S) U_l(S)^dag` is `U`, which is why the split needs no
        # inverse and why only the far side of the pair carries a gradient.
        imps = random_imps(spin_half, 2)
        AC2 = _pair(imps)
        tag = imps.bond_matrix(1).itags[0]
        left_iso, bond, _ = split_two_site(AC2, tag)
        center = einsum('abr,bc->acr', left_iso, bond)
        gauged, _, error = gauge_tensors(center, imps.bond_matrix(0), bond)
        assert float((gauged - left_iso).norm()) < 1e-10
        assert error < 1e-10
