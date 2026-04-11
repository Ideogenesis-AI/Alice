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


"""Tests for Network, MPS, and MPO classes in alice.network."""

from __future__ import annotations

import math

import pytest
import torch
from nicole import contract, conj

from alice.network import MPS, MPO, Network


def _assert_identity_blocks(mat, label: str, atol: float = 1e-7) -> None:
    """Assert every block of a 2-index symmetric tensor equals the identity."""
    for key, block in mat.data.items():
        d = block.shape[0]
        assert torch.allclose(block, torch.eye(d, dtype=block.dtype), atol=atol), (
            f"{label} block {key}: expected identity, got\n{block}"
        )


class TestNetwork:
    """Tests for the shared Network base class interface."""

    def test_L(self, mps_tensors):
        assert Network(mps_tensors).L == 10

    def test_len(self, mps_tensors):
        assert len(Network(mps_tensors)) == 10

    def test_bc_default(self, mps_tensors):
        assert Network(mps_tensors).bc == 'OBC'

    def test_bc_uppercase_normalised(self, mps_tensors):
        assert Network(mps_tensors, bc='obc').bc == 'OBC'

    def test_invalid_bc_raises(self, mps_tensors):
        with pytest.raises(ValueError, match="bc"):
            Network(mps_tensors, bc='INVALID')

    def test_empty_tensors_raises(self):
        with pytest.raises(ValueError):
            Network([])

    def test_bond_dims_length(self, mps_tensors):
        net = Network(mps_tensors)
        assert len(net.bond_dims) == net.L - 1

    def test_bond_dims_positive(self, mps_tensors):
        assert all(d > 0 for d in Network(mps_tensors).bond_dims)

    def test_phys_dims_length(self, mps_tensors):
        net = Network(mps_tensors)
        assert len(net.phys_dims) == net.L

    def test_phys_dims_spin_half(self, mps_tensors):
        """Spin-1/2 physical dimension is 2 at every site."""
        assert all(d == 2 for d in Network(mps_tensors).phys_dims)

    def test_getitem(self, mps_tensors):
        net = Network(mps_tensors)
        assert net[0] is net._tensors[0]

    def test_negative_getitem(self, mps_tensors):
        net = Network(mps_tensors)
        assert net[-1] is net[9]

    def test_setitem(self, mps_tensors):
        net = Network(mps_tensors)
        clone = net[0].clone()
        net[0] = clone
        assert net[0] is clone

    def test_repr_contains_class_name(self, mps_tensors):
        assert 'Network' in repr(Network(mps_tensors))

    def test_repr_contains_L(self, mps_tensors):
        assert 'L=10' in repr(Network(mps_tensors))


class TestMPS:
    """Tests for MPS-specific validation, norm, and canonical."""

    # ------------------------------------------------------------------
    # Constructor validation
    # ------------------------------------------------------------------

    def test_validation_wrong_axis_count(self, mpo_tensors):
        """4-axis MPO tensors must fail MPS validation."""
        with pytest.raises(ValueError, match="3 axes"):
            MPS(mpo_tensors)

    def test_validation_wrong_phys_itag(self, mps_tensors):
        """Incorrect physical itag on site 0 must raise."""
        mps_tensors[0].retag(2, 'wrong_tag')
        with pytest.raises(ValueError, match="itag"):
            MPS(mps_tensors)

    # ------------------------------------------------------------------
    # center attribute
    # ------------------------------------------------------------------

    def test_center_default_is_none(self, mps_tensors):
        assert MPS(mps_tensors).center is None

    def test_center_stored(self, mps_tensors):
        assert MPS(mps_tensors, center=1).center == 1

    # ------------------------------------------------------------------
    # norm()
    # ------------------------------------------------------------------

    def test_norm_returns_float(self, mps_tensors):
        mps = MPS([t.clone() for t in mps_tensors], center=9)
        assert isinstance(mps.norm(), float)

    def test_norm_positive(self, mps_tensors):
        mps = MPS([t.clone() for t in mps_tensors], center=9)
        assert mps.norm() > 0

    def test_norm_sets_center_when_none(self, mps_tensors):
        mps = MPS([t.clone() for t in mps_tensors])
        assert mps.center is None
        mps.norm()
        assert mps.center == 0

    # ------------------------------------------------------------------
    # canonical()
    # ------------------------------------------------------------------

    def test_canonical_sets_center(self, mps_tensors):
        mps = MPS([t.clone() for t in mps_tensors])
        for target in [0, 5, 9]:
            mps.canonical(target)
            assert mps.center == target

    def test_canonical_preserves_norm(self, mps_tensors):
        """The Frobenius norm of the state must be invariant under canonical()."""
        mps = MPS([t.clone() for t in mps_tensors])
        mps.canonical(0)
        n0 = mps.norm()
        mps.canonical(9)
        n9 = mps.norm()
        assert math.isclose(n0, n9, rel_tol=1e-10)

    def test_canonical_negative_target(self, mps_tensors):
        """Negative target index must resolve like standard Python negative indexing."""
        mps = MPS([t.clone() for t in mps_tensors])
        mps.canonical(-1)
        assert mps.center == 9

    def test_canonical_out_of_range_raises(self, mps_tensors):
        mps = MPS([t.clone() for t in mps_tensors])
        with pytest.raises(IndexError):
            mps.canonical(15)

    # ------------------------------------------------------------------
    # Isometry conditions after canonical()
    # ------------------------------------------------------------------

    def test_left_canonical_isometry(self, mps_tensors):
        """Sites left of the center must satisfy A†A = I in the right-bond space.

        Verified by checking that every block of A†A equals the identity matrix.
        """
        mps = MPS([t.clone() for t in mps_tensors])
        mps.canonical(9)  # sites 0..8 become left-canonical

        for i in range(9):  # all left-canonical sites
            A = mps[i]
            # Contract over (left=0, phys=2) leaving the right-bond pair.
            AcA = contract(conj(A), A, axes=([0, 2], [0, 2]))
            # Each block of A†A must equal the identity of that sector's dimension.
            _assert_identity_blocks(AcA, f"MPS site {i} A†A")

    def test_right_canonical_isometry(self, mps_tensors):
        """Sites right of the center must satisfy BB† = I in the left-bond space.

        Verified by checking that every block of BB† equals the identity matrix.
        """
        mps = MPS([t.clone() for t in mps_tensors])
        mps.canonical(0)  # sites 1..9 become right-canonical

        for i in range(1, 10):  # all right-canonical sites
            B = mps[i]
            # Contract over (right=1, phys=2) leaving the left-bond pair.
            BBc = contract(B, conj(B), axes=([1, 2], [1, 2]))
            # Each block of BB† must equal the identity of that sector's dimension.
            _assert_identity_blocks(BBc, f"MPS site {i} BB†")

    def test_mixed_canonical_isometry(self, mps_tensors):
        """Mixed canonical form with center=5: sites 0..4 are left-canonical,
        sites 6..9 are right-canonical.

        Verified by checking that every block of the respective isometry equals
        the identity matrix.
        """
        center = 5
        mps = MPS([t.clone() for t in mps_tensors])
        mps.canonical(center)

        for i in range(center):  # left-canonical sites
            A = mps[i]
            # Contract over (left=0, phys=2) leaving the right-bond pair.
            AcA = contract(conj(A), A, axes=([0, 2], [0, 2]))
            _assert_identity_blocks(AcA, f"MPS site {i} A†A")

        for i in range(center + 1, 10):  # right-canonical sites
            B = mps[i]
            # Contract over (right=1, phys=2) leaving the left-bond pair.
            BBc = contract(B, conj(B), axes=([1, 2], [1, 2]))
            _assert_identity_blocks(BBc, f"MPS site {i} BB†")


class TestMPO:
    """Tests for MPO-specific validation, norm, and canonical."""

    # ------------------------------------------------------------------
    # Constructor validation
    # ------------------------------------------------------------------

    def test_validation_wrong_axis_count(self, mps_tensors):
        """3-axis (MPS) tensors must fail MPO validation."""
        with pytest.raises(ValueError, match="4 axes"):
            MPO(mps_tensors)

    def test_validation_wrong_phys_itag(self, mpo_tensors):
        """Incorrect physical itag must raise."""
        mpo_tensors[0].retag(2, 'wrong_tag')
        with pytest.raises(ValueError, match="itag"):
            MPO(mpo_tensors)

    def test_validation_same_direction_raises(self, mpo_tensors):
        """Physical axes with the same direction must raise."""
        # Flip axis 3 so it matches axis 2 (both become IN).
        mpo_tensors[0].invert(3)
        with pytest.raises(ValueError, match="direction"):
            MPO(mpo_tensors)

    # ------------------------------------------------------------------
    # center attribute
    # ------------------------------------------------------------------

    def test_center_default_is_none(self, mpo_tensors):
        assert MPO(mpo_tensors).center is None

    def test_center_stored(self, mpo_tensors):
        assert MPO(mpo_tensors, center=1).center == 1

    # ------------------------------------------------------------------
    # bond_dims and phys_dims
    # ------------------------------------------------------------------

    def test_bond_dims_length(self, mpo_tensors):
        assert len(MPO(mpo_tensors).bond_dims) == 9  # L=10 → 9 internal bonds

    def test_bond_dims_nontrivial(self, mpo_tensors):
        """Bulk MPO bond dimensions must be strictly greater than 1 (non-trivial)."""
        assert all(d > 1 for d in MPO(mpo_tensors).bond_dims[1:])

    def test_phys_dims_spin_half(self, mpo_tensors):
        """Ket physical dimension is 2 at every site (spin-1/2)."""
        assert all(d == 2 for d in MPO(mpo_tensors).phys_dims)

    # ------------------------------------------------------------------
    # norm() and canonical()
    # ------------------------------------------------------------------

    def test_norm_positive(self, mpo_tensors):
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(0)
        n = mpo.norm()
        assert isinstance(n, float) and n > 0

    def test_canonical_sets_center(self, mpo_tensors):
        mpo = MPO([t.clone() for t in mpo_tensors])
        for target in [0, 5, 9]:
            mpo.canonical(target)
            assert mpo.center == target

    def test_canonical_preserves_norm(self, mpo_tensors):
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(0)
        n0 = mpo.norm()
        mpo.canonical(9)
        n9 = mpo.norm()
        assert math.isclose(n0, n9, rel_tol=1e-10)

    # ------------------------------------------------------------------
    # Isometry conditions after canonical()
    # ------------------------------------------------------------------

    def test_left_canonical_isometry(self, mpo_tensors):
        """Sites left of center must satisfy W†W = I (contracting left, phys_in, phys_out).

        Verified by checking that every block of W†W equals the identity matrix.
        """
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(9)  # sites 0..8 become left-canonical

        for i in range(9):  # all left-canonical sites
            W = mpo[i]
            # Contract over left(0), phys_in(2), phys_out(3) leaving right-bond pair.
            WcW = contract(conj(W), W, axes=([0, 2, 3], [0, 2, 3]))
            # Each block of W†W must equal the identity of that sector's dimension.
            _assert_identity_blocks(WcW, f"MPO site {i} W†W")

    def test_right_canonical_isometry(self, mpo_tensors):
        """Sites right of center must satisfy WW† = I (contracting right, phys_in, phys_out).

        Verified by checking that every block of WW† equals the identity matrix.
        """
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(0)  # sites 1..9 become right-canonical

        for i in range(1, 10):  # all right-canonical sites
            W = mpo[i]
            # Contract over right(1), phys_in(2), phys_out(3) leaving left-bond pair.
            WWc = contract(W, conj(W), axes=([1, 2, 3], [1, 2, 3]))
            # Each block of WW† must equal the identity of that sector's dimension.
            _assert_identity_blocks(WWc, f"MPO site {i} WW†")

    def test_mixed_canonical_isometry(self, mpo_tensors):
        """Mixed canonical form with center=5: sites 0..4 are left-canonical,
        sites 6..9 are right-canonical.

        Verified by checking that every block of the respective isometry equals
        the identity matrix.
        """
        center = 5
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(center)

        for i in range(center):  # left-canonical sites
            W = mpo[i]
            # Contract over left(0), phys_in(2), phys_out(3) leaving right-bond pair.
            WcW = contract(conj(W), W, axes=([0, 2, 3], [0, 2, 3]))
            _assert_identity_blocks(WcW, f"MPO site {i} W†W")

        for i in range(center + 1, 10):  # right-canonical sites
            W = mpo[i]
            # Contract over right(1), phys_in(2), phys_out(3) leaving left-bond pair.
            WWc = contract(W, conj(W), axes=([1, 2, 3], [1, 2, 3]))
            _assert_identity_blocks(WWc, f"MPO site {i} WW†")
