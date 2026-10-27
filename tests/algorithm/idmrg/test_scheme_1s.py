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


"""Tests for the VUMPS local eigenproblems and the gauge step."""

from __future__ import annotations

import pytest
from nicole import identity

from alice.algorithm.idmrg import transfer_left, transfer_right
from alice.algorithm.idmrg.davidson import _inner_product
from alice.algorithm.idmrg.scheme_1s import (
    gauge_tensors,
    matvec_bond,
    matvec_center,
    optimize_bond,
    optimize_center,
)

from .conftest import (
    HEISENBERG,
    MAJUMDAR_GHOSH,
    bond_environments,
    dimer_imps,
    ihamiltonian,
    random_imps,
)

DAVIDSON = {'max_iter': 100, 'tol': 1e-10, 'max_subspace': 20}


def _delta(idx, itag):
    """Build an identity on a bond index, tagged for both ends of a cell."""
    delta = identity(idx)
    delta.retag([0, 1], [itag, itag])
    return delta


def _setup(imps, model=HEISENBERG, n3x=False):
    """Return the iMPO and the environment blocks of a gauge-fixed state.

    Both block lists are indexed by bond, so the right block that goes
    with site `i` is `rights[(i + 1) % N]` and the one that goes with
    bond `k` is `rights[k]`.
    """
    impo = ihamiltonian(model, imps.N, n3x=n3x)
    lefts, rights = bond_environments(imps, impo)
    return impo, lefts, rights


# ---------------------------------------------------------------------------
# Effective Hamiltonians
# ---------------------------------------------------------------------------

class TestEffectiveHamiltonians:
    """Both local operators must be Hermitian and shape-preserving."""

    @pytest.mark.parametrize('N', [2, 4])
    def test_center_operator_preserves_layout(self, spin_half, N):
        imps = random_imps(spin_half, N)
        impo, lefts, rights = _setup(imps)
        center = imps.center_tensor(0)
        out = matvec_center(center, impo[0], lefts[0], rights[1])
        assert out.itags == center.itags
        assert [i.dim for i in out.indices] == [i.dim for i in center.indices]

    @pytest.mark.parametrize('N', [2, 4])
    def test_bond_operator_preserves_layout(self, spin_half, N):
        imps = random_imps(spin_half, N)
        impo, lefts, rights = _setup(imps)
        bond = imps.bond_matrix(0)
        out = matvec_bond(bond, lefts[0], rights[0])
        assert out.itags == bond.itags
        assert [i.dim for i in out.indices] == [i.dim for i in bond.indices]

    def test_center_operator_is_hermitian(self, spin_half):
        # Davidson projects onto a real symmetric matrix, so an operator
        # that is not Hermitian would be silently symmetrized and the
        # eigenvector would be wrong rather than merely inaccurate.
        imps = random_imps(spin_half, 2)
        impo, lefts, rights = _setup(imps)
        u = imps.center_tensor(0)
        v = random_imps(spin_half, 2, seed=7).center_tensor(0)
        left = _inner_product(u, matvec_center(v, impo[0], lefts[0], rights[1]))
        right = _inner_product(matvec_center(u, impo[0], lefts[0], rights[1]), v)
        assert abs(left - right) < 1e-10 * max(1.0, abs(left))

    def test_bond_operator_is_hermitian(self, spin_half):
        imps = random_imps(spin_half, 2)
        impo, lefts, rights = _setup(imps)
        u = imps.bond_matrix(0)
        v = random_imps(spin_half, 2, seed=7).bond_matrix(0)
        left = _inner_product(u, matvec_bond(v, lefts[0], rights[0]))
        right = _inner_product(matvec_bond(u, lefts[0], rights[0]), v)
        assert abs(left - right) < 1e-10 * max(1.0, abs(left))


# ---------------------------------------------------------------------------
# Local eigensolves
# ---------------------------------------------------------------------------

class TestLocalSolves:
    """The Davidson wrappers must return normalized, improving vectors."""

    def test_center_solve_does_not_raise_the_ritz_value(self, spin_half):
        imps = random_imps(spin_half, 2)
        impo, lefts, rights = _setup(imps)
        center = imps.center_tensor(0)
        before = _inner_product(
            center, matvec_center(center, impo[0], lefts[0], rights[1]),
        ).real
        energy, optimized, _ = optimize_center(
            center, impo[0], lefts[0], rights[1], DAVIDSON,
        )
        assert energy <= before + 1e-12
        assert abs(float(optimized.norm()) - 1.0) < 1e-10

    def test_bond_solve_does_not_raise_the_ritz_value(self, spin_half):
        imps = random_imps(spin_half, 2)
        impo, lefts, rights = _setup(imps)
        bond = imps.bond_matrix(0)
        before = _inner_product(bond, matvec_bond(bond, lefts[0], rights[0])).real
        energy, optimized, _ = optimize_bond(bond, lefts[0], rights[0], DAVIDSON)
        assert energy <= before + 1e-12
        assert abs(float(optimized.norm()) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Gauge step
# ---------------------------------------------------------------------------

class TestGaugeStep:
    """The polar reconciliation must produce genuine canonical forms."""

    @pytest.mark.parametrize('N', [2, 4])
    def test_factors_are_orthonormal(self, spin_half, N):
        # The polar factors are isometric by construction whatever is fed
        # in, so this holds away from the fixed point too.
        imps = random_imps(spin_half, N)
        impo, lefts, rights = _setup(imps)
        centers, bonds = [], []
        for i in range(N):
            _, center, _ = optimize_center(
                imps.center_tensor(i), impo[i], lefts[i], rights[(i + 1) % N],
                DAVIDSON,
            )
            centers.append(center)
        for k in range(N):
            _, bond, _ = optimize_bond(
                imps.bond_matrix(k), lefts[k], rights[k], DAVIDSON,
            )
            bonds.append(bond)

        left_isos, right_isos = [], []
        for i in range(N):
            left_iso, right_iso, _ = gauge_tensors(
                centers[i], bonds[i], bonds[(i + 1) % N],
            )
            left_isos.append(left_iso)
            right_isos.append(right_iso)

        delta = _delta(left_isos[0].indices[0], left_isos[0].itags[0])
        assert float((transfer_left(delta, left_isos) - delta).norm()) < 1e-10
        last = right_isos[N - 1]
        delta = _delta(last.indices[1], last.itags[1])
        assert float((transfer_right(delta, right_isos) - delta).norm()) < 1e-10

    @pytest.mark.parametrize('N', [2, 4])
    def test_gauge_fixed_state_is_a_fixed_point_of_the_step(self, spin_half, N):
        # A gauge-fixed cell already satisfies AC = AL C exactly, so feeding
        # its own center tensors and bond matrices back in must report no
        # gradient. This is the identity the convergence criterion rests on.
        imps = random_imps(spin_half, N)
        for i in range(N):
            _, _, error = gauge_tensors(
                imps.center_tensor(i), imps.bond_matrix(i), imps.bond_matrix(i + 1),
            )
            assert error < 1e-10

    def test_gradient_vanishes_on_an_exact_ground_state(self, spin_half):
        # The dimer state is the exact Majumdar-Ghosh ground state, so the
        # two eigenproblems must already agree on it.
        imps = dimer_imps(spin_half)
        impo, lefts, rights = _setup(imps, MAJUMDAR_GHOSH, n3x=True)
        gradient = 0.0
        for i in range(imps.N):
            _, center, _ = optimize_center(
                imps.center_tensor(i), impo[i], lefts[i],
                rights[(i + 1) % imps.N], DAVIDSON,
            )
            _, bond_left, _ = optimize_bond(
                imps.bond_matrix(i), lefts[i], rights[i], DAVIDSON,
            )
            _, bond_right, _ = optimize_bond(
                imps.bond_matrix(i + 1), lefts[(i + 1) % imps.N],
                rights[(i + 1) % imps.N], DAVIDSON,
            )
            _, _, error = gauge_tensors(center, bond_left, bond_right)
            gradient = max(gradient, error)
        assert gradient < 1e-7
