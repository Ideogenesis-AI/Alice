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


"""Unit tests for `alice.physics.system` builder functions.

Tests are lightweight — no iterative diagonalization or MPS/MPO contraction.
Each test calls a builder directly and verifies tensor properties such as
index count, axis directions, operator algebra identities, and internal
consistency of the hopping decomposition.
"""

from __future__ import annotations

from typing import Dict

import pytest

from nicole import Direction, Tensor, contract
from nicole.index import Index

from alice.physics.system import build_bosonic, build_fermionic, build_conductor

_ATOL = 1e-12

# All 4th-order templates share the same direction pattern regardless of
# whether they are leading-site, terminal-site, or on-site.  The physical
# (bra, ket) axes follow Nicole's operator convention: bra = IN, ket = OUT.
# Layout: (L_or_op = IN, R_or_op = OUT, bra = IN, ket = OUT).
_4TH_ORDER_DIRECTIONS = (Direction.IN, Direction.OUT, Direction.IN, Direction.OUT)


def _directions(t: Tensor):
    """Return a tuple of axis directions for tensor `t`."""
    return tuple(idx.direction for idx in t.indices)


# ---------------------------------------------------------------------------
# TestBosonic
# ---------------------------------------------------------------------------

class TestBosonic:
    """Tests for `build_bosonic`."""

    @pytest.mark.parametrize('symmetry', ['U1', 'SU2'])
    def test_spc_is_index(self, symmetry):
        """Returned `Spc` must be a Nicole `Index`."""
        Spc, _ = build_bosonic(symmetry=symmetry)
        assert isinstance(Spc, Index)

    @pytest.mark.parametrize('symmetry', ['U1', 'SU2'])
    def test_keys_present(self, symmetry):
        """All required keys must be present; `Sz4`/`Sz4dag` only for U1."""
        _, Op = build_bosonic(symmetry=symmetry)
        required = {'S', 'Sdag', 'S4', 'S4dag', 'I4', 'I4mid'}
        assert required <= Op.keys()
        if symmetry == 'U1':
            assert 'Sz4'    in Op
            assert 'Sz4dag' in Op
        else:
            assert 'Sz4'    not in Op
            assert 'Sz4dag' not in Op

    @pytest.mark.parametrize('symmetry', ['U1', 'SU2'])
    def test_4th_order_axis_count(self, symmetry):
        """Every 4th-order template must have exactly 4 indices."""
        _, Op = build_bosonic(symmetry=symmetry)
        keys_4 = ['S4', 'S4dag', 'I4', 'I4mid']
        if symmetry == 'U1':
            keys_4 += ['Sz4', 'Sz4dag']
        for key in keys_4:
            assert len(Op[key].indices) == 4, f"'{key}' has {len(Op[key].indices)} indices"

    @pytest.mark.parametrize('symmetry', ['U1', 'SU2'])
    def test_4th_order_axis_directions(self, symmetry):
        """Leading, terminal, on-site, and intermediate 4th-order templates must
        follow the `(IN, OUT, IN, OUT)` direction convention."""
        _, Op = build_bosonic(symmetry=symmetry)
        for key in ['S4', 'S4dag', 'I4', 'I4mid']:
            assert _directions(Op[key]) == _4TH_ORDER_DIRECTIONS, (
                f"'{key}' directions: {_directions(Op[key])}"
            )
        if symmetry == 'U1':
            for key in ['Sz4', 'Sz4dag']:
                assert _directions(Op[key]) == _4TH_ORDER_DIRECTIONS, (
                    f"'{key}' directions: {_directions(Op[key])}"
                )

    @pytest.mark.parametrize('symmetry', ['U1', 'SU2'])
    def test_i4mid_op_bond_matches_s4(self, symmetry):
        """`I4mid` op bond index (axes 0 and 1) must have the same dimension as
        the op_OUT of `S4` and op_IN of `S4dag`."""
        _, Op = build_bosonic(symmetry=symmetry)
        op_dim = Op['S4'].indices[1].dim
        assert Op['I4mid'].indices[0].dim == op_dim, (
            "I4mid axis 0 (op_IN) dim does not match S4 op_OUT dim"
        )
        assert Op['I4mid'].indices[1].dim == op_dim, (
            "I4mid axis 1 (op_OUT) dim does not match S4 op_OUT dim"
        )

    @pytest.mark.parametrize('symmetry', ['U1', 'SU2'])
    def test_s4dag_is_adjoint_of_s4(self, symmetry):
        """S4dag carries the same operator data as S4 up to permutation; their
        norms must therefore be equal."""
        _, Op = build_bosonic(symmetry=symmetry)
        assert abs(Op['S4'].norm() - Op['S4dag'].norm()) < _ATOL

    @pytest.mark.parametrize('symmetry', ['U1', 'SU2'])
    def test_sz4_absent_for_su2(self, symmetry):
        """`Sz4` and `Sz4dag` must not appear when SU2 symmetry is used."""
        _, Op = build_bosonic(symmetry=symmetry)
        if symmetry == 'SU2':
            assert 'Sz4'    not in Op
            assert 'Sz4dag' not in Op


# ---------------------------------------------------------------------------
# TestFermionic
# ---------------------------------------------------------------------------

class TestFermionic:
    """Tests for `build_fermionic`."""

    @pytest.mark.parametrize('symmetry', ['U1', 'Z2'])
    def test_spc_is_index(self, symmetry):
        """Returned `Spc` must be a Nicole `Index`."""
        Spc, _ = build_fermionic(symmetry=symmetry)
        assert isinstance(Spc, Index)

    @pytest.mark.parametrize('symmetry', ['U1', 'Z2'])
    def test_keys_present(self, symmetry):
        """All required keys must be present in `Op`."""
        _, Op = build_fermionic(symmetry=symmetry)
        required = {'F', 'C', 'Fd', 'Cd', 'G', 'Gdag', 'G4', 'G4dag',
                    'N', 'N4', 'I4', 'Z4', 'Z4mid'}
        assert required <= Op.keys()

    @pytest.mark.parametrize('symmetry', ['U1', 'Z2'])
    def test_4th_order_axis_count(self, symmetry):
        """Every 4th-order template must have exactly 4 indices."""
        _, Op = build_fermionic(symmetry=symmetry)
        for key in ['G4', 'G4dag', 'N4', 'I4', 'Z4', 'Z4mid']:
            assert len(Op[key].indices) == 4, f"'{key}' has {len(Op[key].indices)} indices"

    @pytest.mark.parametrize('symmetry', ['U1', 'Z2'])
    def test_4th_order_axis_directions(self, symmetry):
        """All 4th-order templates must follow `(IN, OUT, IN, OUT)`."""
        _, Op = build_fermionic(symmetry=symmetry)
        for key in ['G4', 'G4dag', 'N4', 'I4', 'Z4', 'Z4mid']:
            assert _directions(Op[key]) == _4TH_ORDER_DIRECTIONS, (
                f"'{key}' directions: {_directions(Op[key])}"
            )

    @pytest.mark.parametrize('symmetry', ['U1', 'Z2'])
    def test_z4mid_op_bond_matches_g4(self, symmetry):
        """`Z4mid` op bond (axes 0 and 1) must have the same dimension as the
        op_OUT of `G4` and op_IN of `G4dag`."""
        _, Op = build_fermionic(symmetry=symmetry)
        op_dim = Op['G4'].indices[1].dim
        assert Op['Z4mid'].indices[0].dim == op_dim, (
            "Z4mid axis 0 (op_IN) dim does not match G4 op_OUT dim"
        )
        assert Op['Z4mid'].indices[1].dim == op_dim, (
            "Z4mid axis 1 (op_OUT) dim does not match G4 op_OUT dim"
        )

    @pytest.mark.parametrize('symmetry', ['U1', 'Z2'])
    def test_n_is_2nd_order(self, symmetry):
        """`'N'` must be a 2nd-order tensor (bra, ket)."""
        _, Op = build_fermionic(symmetry=symmetry)
        assert len(Op['N'].indices) == 2

    @pytest.mark.parametrize('symmetry', ['U1', 'Z2'])
    def test_z_squared_identity(self, symmetry):
        """Z is its own inverse: Z² must equal the identity operator."""
        from nicole import identity
        _, Op = build_fermionic(symmetry=symmetry)
        Z    = Op['Z']
        Z_sq = contract(Z, Z, axes=(1, 0))
        # Compare with the plain 2nd-order identity (same index as bra/ket of Z).
        I2   = identity(Z.indices[0])
        assert (Z_sq - I2).norm() < _ATOL

    @pytest.mark.parametrize('symmetry', ['U1', 'Z2'])
    def test_n_idempotent(self, symmetry):
        """Number operator is a projector: N² must equal N."""
        _, Op = build_fermionic(symmetry=symmetry)
        N    = Op['N']
        N_sq = contract(N, N, axes=([1], [0]))
        assert (N_sq - N).norm() < _ATOL

    @pytest.mark.parametrize('symmetry', ['U1', 'Z2'])
    def test_f_squared_zero(self, symmetry):
        """Pauli exclusion: applying two annihilators gives zero."""
        _, Op = build_fermionic(symmetry=symmetry)
        F    = Op['F']
        # Contract ket of F1 (OUT, axis 1) with bra of F2 (IN, axis 0).
        # The resulting 4th-order tensor (bra_F1, op_F1, op_F2, ket_F2)
        # must vanish because c² = 0 for fermionic modes.
        F_sq = contract(F, F, axes=([1], [0]))
        assert F_sq.norm() < _ATOL

    @pytest.mark.parametrize('symmetry', ['U1', 'Z2'])
    def test_hopping_hermitian_consistency(self, symmetry):
        """G ⊗ Gdag must equal T + T†, where T = Fd_i ⊗ F_j.

        This verifies that the MPO hopping decomposition (G on the leading
        site, Gdag on the terminal site) correctly represents the Hermitian
        hopping operator `c†_i c_j + c†_j c_i` when viewed as a bipartite
        operator on the local tensor product space.

        Note: Fd (op_IN) and F (op_OUT) have opposite op directions and can be
        contracted directly. The result T has axes
        `(bra_i, ket_i, bra_j, ket_j)`. Its Hermitian adjoint is obtained by
        swapping bra ↔ ket on both sites via `.conj().permute([1, 0, 3, 2])`.
        """
        _, Op = build_fermionic(symmetry=symmetry)
        Fd   = Op['Fd']
        F    = Op['F']
        G    = Op['G']
        Gdag = Op['Gdag']

        # Direct bipartite operator: c†_i c_j.
        T     = contract(Fd, F,    axes=(2, 2))
        T_dag = T.conj().permute([1, 0, 3, 2])

        # MPO hopping decomposition (no coupling factor).
        H_hop = contract(G, Gdag, axes=(2, 2))

        assert (T + T_dag - H_hop).norm() < _ATOL


# ---------------------------------------------------------------------------
# TestConductor
# ---------------------------------------------------------------------------

class TestConductor:
    """Tests for `build_conductor`."""

    @pytest.mark.parametrize('symmetry', ['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'])
    def test_spc_is_index(self, symmetry):
        """Returned `Spc` must be a Nicole `Index`."""
        Spc, _ = build_conductor(symmetry=symmetry)
        assert isinstance(Spc, Index)

    @pytest.mark.parametrize('symmetry', ['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'])
    def test_keys_present(self, symmetry):
        """All required keys must be present; `Sz4`/`Sz4dag` only for Abelian."""
        _, Op = build_conductor(symmetry=symmetry)
        hopping = {'F', 'ZF', 'ZC', 'Fd', 'Cd', 'G', 'Gdag', 'G4', 'G4dag'}
        spin    = {'S', 'Sdag', 'S4', 'S4dag'}
        onsite  = {'N', 'NN', 'N4', 'NN4', 'I4', 'Z4', 'Z4mid'}
        assert hopping | spin | onsite <= Op.keys()
        if 'SU2' not in symmetry:
            assert 'Sz4'    in Op
            assert 'Sz4dag' in Op
        else:
            assert 'Sz4'    not in Op
            assert 'Sz4dag' not in Op

    @pytest.mark.parametrize('symmetry', ['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'])
    def test_4th_order_axis_count(self, symmetry):
        """Every 4th-order template must have exactly 4 indices."""
        _, Op = build_conductor(symmetry=symmetry)
        keys_4 = ['G4', 'G4dag', 'S4', 'S4dag', 'N4', 'NN4', 'I4', 'Z4', 'Z4mid']
        if 'SU2' not in symmetry:
            keys_4 += ['Sz4', 'Sz4dag']
        for key in keys_4:
            assert len(Op[key].indices) == 4, f"'{key}' has {len(Op[key].indices)} indices"

    @pytest.mark.parametrize('symmetry', ['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'])
    def test_4th_order_axis_directions(self, symmetry):
        """All 4th-order templates must follow `(IN, OUT, IN, OUT)`."""
        _, Op = build_conductor(symmetry=symmetry)
        keys_4 = ['G4', 'G4dag', 'S4', 'S4dag', 'N4', 'NN4', 'I4', 'Z4', 'Z4mid']
        if 'SU2' not in symmetry:
            keys_4 += ['Sz4', 'Sz4dag']
        for key in keys_4:
            assert _directions(Op[key]) == _4TH_ORDER_DIRECTIONS, (
                f"'{key}' directions: {_directions(Op[key])}"
            )

    @pytest.mark.parametrize('symmetry', ['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'])
    def test_z4mid_op_bond_matches_g4(self, symmetry):
        """`Z4mid` op bond (axes 0 and 1) must have the same dimension as the
        op_OUT of `G4` and op_IN of `G4dag`."""
        _, Op = build_conductor(symmetry=symmetry)
        op_dim = Op['G4'].indices[1].dim
        assert Op['Z4mid'].indices[0].dim == op_dim, (
            "Z4mid axis 0 (op_IN) dim does not match G4 op_OUT dim"
        )
        assert Op['Z4mid'].indices[1].dim == op_dim, (
            "Z4mid axis 1 (op_OUT) dim does not match G4 op_OUT dim"
        )

    @pytest.mark.parametrize('symmetry', ['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'])
    def test_n_is_2nd_order(self, symmetry):
        """`'N'` must be a 2nd-order tensor."""
        _, Op = build_conductor(symmetry=symmetry)
        assert len(Op['N'].indices) == 2

    @pytest.mark.parametrize('symmetry', ['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'])
    def test_nn_is_2nd_order(self, symmetry):
        """`'NN'` must be a 2nd-order tensor."""
        _, Op = build_conductor(symmetry=symmetry)
        assert len(Op['NN'].indices) == 2

    @pytest.mark.parametrize('symmetry', ['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'])
    def test_z_squared_identity(self, symmetry):
        """Z² must equal the identity operator."""
        from nicole import identity
        _, Op = build_conductor(symmetry=symmetry)
        Z    = Op['Z']
        Z_sq = contract(Z, Z, axes=(1, 0))
        I2   = identity(Z.indices[0])
        assert (Z_sq - I2).norm() < _ATOL

    @pytest.mark.parametrize('symmetry', ['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'])
    def test_n_idempotent(self, symmetry):
        """Total number operator is a projector: N² must equal N.

        For a single spinful site the local occupation number
        n = n_up + n_dn satisfies n² = n only if the site has at most one
        electron, which is NOT the case here (double occupancy is allowed).
        Instead we verify the weaker idempotence identity
        n² - n = 2 * NN (double occupancy), i.e. the difference N² - N
        equals 2*NN.
        """
        _, Op = build_conductor(symmetry=symmetry)
        N     = Op['N']
        NN    = Op['NN']
        N_sq  = contract(N, N, axes=([1], [0]))
        assert (N_sq - N - NN * 2.0).norm() < _ATOL

    @pytest.mark.parametrize('symmetry', ['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'])
    def test_nn_idempotent(self, symmetry):
        """Double-occupancy operator is a projector: NN² must equal NN."""
        _, Op = build_conductor(symmetry=symmetry)
        NN    = Op['NN']
        NN_sq = contract(NN, NN, axes=([1], [0]))
        assert (NN_sq - NN).norm() < _ATOL

    @pytest.mark.parametrize('symmetry', ['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'])
    def test_nn_bounded_by_n(self, symmetry):
        """Double occupancy cannot exceed total occupation: Tr[NN] ≤ Tr[N].

        Both traces are computed by contracting bra with ket on the 2nd-order
        operator, giving a scalar representing the sum of diagonal elements
        over all local states.
        """
        from nicole import trace
        _, Op = build_conductor(symmetry=symmetry)
        N   = Op['N']
        NN  = Op['NN']
        tr_N  = trace(N).norm()
        tr_NN = trace(NN).norm()
        assert tr_NN <= tr_N + _ATOL

    @pytest.mark.parametrize('symmetry', ['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'])
    def test_sz4_absent_for_su2(self, symmetry):
        """`Sz4` and `Sz4dag` must not appear when SU2 is present."""
        _, Op = build_conductor(symmetry=symmetry)
        if 'SU2' in symmetry:
            assert 'Sz4'    not in Op
            assert 'Sz4dag' not in Op

    @pytest.mark.parametrize('symmetry', ['U1,U1', 'Z2,U1', 'U1,SU2', 'Z2,SU2'])
    def test_hopping_hermitian_consistency(self, symmetry):
        """G ⊗ Gdag must equal T + T†, where T = ZC_i ⊗ Cd_j.

        `ZC` (op_OUT after `capcup`) and `Cd` (op_IN after `capcup`)
        have opposite op directions and can be contracted directly. T represents
        `(ZF)†_i ⊗ F_j = (JW creator at i) ⊗ (bare annihilator at j)`, the
        leading hopping channel. T† swaps bra ↔ ket on both sites, recovering
        the conjugate channel. Their sum must equal the full G ⊗ Gdag.
        """
        _, Op = build_conductor(symmetry=symmetry)
        ZC   = Op['ZC']
        Cd   = Op['Cd']
        G    = Op['G']
        Gdag = Op['Gdag']

        # Direct bipartite operator: (ZF)†_i F_j.
        T     = contract(ZC, Cd, axes=(2, 2))
        T_dag = T.conj().permute([1, 0, 3, 2])

        H_hop = contract(G, Gdag, axes=(2, 2))

        assert (T + T_dag - H_hop).norm() < _ATOL
