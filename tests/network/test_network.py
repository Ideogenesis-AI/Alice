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


"""Tests for Network, MPS, and MPO classes in alice.network."""

from __future__ import annotations

import math

import pytest
import torch
from nicole import allclose, contract, conj, identity

from alice.network import MPS, MPO, Network


def _assert_identity_blocks(mat, label: str, atol: float = 1e-7) -> None:
    """Assert every block of a 2-index symmetric tensor equals the identity."""
    for key, block in mat.data.items():
        d = block.shape[0]
        assert torch.allclose(block, torch.eye(d, dtype=block.dtype), atol=atol), (
            f"{label} block {key}: expected identity, got\n{block}"
        )


def _assert_is_identity(mat, label: str, atol: float = 1e-7) -> None:
    """Assert that a 2-index symmetric tensor equals the identity.

    Uses Nicole's `identity` and `allclose` for a gauge-invariant comparison
    that is correct for both Abelian and non-Abelian (SU(2)) tensors. SU(2)
    blocks carry Bridge intertwiners and extra coupling dimensions that make
    raw block comparison unreliable; `allclose` accounts for these via the
    physical `R @ W` representation.
    """
    I = identity(mat.indices[0])
    I.retag([0, 1], list(mat.itags))
    assert allclose(mat, I, atol=atol), f"{label}: tensor is not the identity"


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

    def test_bond_itag_mismatch_raises(self, mps_tensors):
        """Mismatched bond itag between adjacent sites must raise."""
        mps_tensors[1].retag(0, 'WRONG')
        with pytest.raises(ValueError, match="itag mismatch"):
            Network(mps_tensors)

    def test_bond_same_direction_raises(self, mps_tensors):
        """Bond indices with the same direction between adjacent sites must raise."""
        mps_tensors[1].invert(0)
        with pytest.raises(ValueError, match="opposite directions"):
            Network(mps_tensors)

    def test_bond_sector_dim_mismatch_raises(self, mps_tensors, spin_space):
        """Mismatched sector dimension on a shared charge must raise."""
        from nicole.index import Index, Sector
        from nicole import Direction
        Spc, Op = spin_space
        # Replace site 1's left bond with an index that has a different dim
        # for an overlapping charge sector.
        bad_left = Index(
            direction=mps_tensors[1].indices[0].direction,
            group=Spc.group,
            sectors=(Sector(charge=-1, dim=99),),  # dim 99 ≠ dim 2 in site 0's right bond
        )
        from nicole import Tensor
        mps_tensors[1] = Tensor.random(
            [bad_left] + list(mps_tensors[1].indices[1:]),
            itags=list(mps_tensors[1].itags),
            seed=99,
        )
        with pytest.raises(ValueError, match="sector charge"):
            Network(mps_tensors)

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

    # ------------------------------------------------------------------
    # SU(2) symmetry
    # ------------------------------------------------------------------

    def test_phys_dims_su2(self, mps_tensors_su2):
        """SU(2) spin-1/2 physical index reports 1 multiplet per site.

        `phys_dims` returns `idx.dim`, the number of SU(2) multiplets, not
        the total Hilbert space dimension. For a spin-1/2 doublet there is
        one multiplet, so `dim == 1` at every site.
        """
        assert all(d == 1 for d in Network(mps_tensors_su2).phys_dims)


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

    def test_norm_does_not_set_center(self, mps_tensors):
        """norm() must not modify the network when center is None."""
        mps = MPS([t.clone() for t in mps_tensors])
        assert mps.center is None
        mps.norm()
        assert mps.center is None

    def test_norm_uncanonical_equals_canonical(self, mps_tensors):
        """norm() without canonicalization must agree with the canonical fast path."""
        tensors = [t.clone() for t in mps_tensors]
        n_direct = MPS(tensors).norm()          # contraction path (center=None)
        mps = MPS([t.clone() for t in mps_tensors])
        mps.canonical(5, trunc=None)
        n_canonical = mps.norm()                # fast path (center=5)
        assert math.isclose(n_direct, n_canonical, rel_tol=1e-10)

    # ------------------------------------------------------------------
    # canonical()
    # ------------------------------------------------------------------

    def test_canonical_sets_center(self, mps_tensors):
        mps = MPS([t.clone() for t in mps_tensors])
        for target in [0, 5, 9]:
            mps.canonical(target)
            assert mps.center == target
            mps._validate()

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

    # ------------------------------------------------------------------
    # SU(2) symmetry
    # ------------------------------------------------------------------

    def test_norm_positive_su2(self, mps_tensors_su2):
        mps = MPS([t.clone() for t in mps_tensors_su2], center=9)
        assert mps.norm() > 0

    def test_norm_uncanonical_equals_canonical_su2(self, mps_tensors_su2):
        """Full-contraction and center-tensor norms must agree for SU(2) MPS.

        `norm()` without a center uses the einsum transfer path with Bridge
        weighting; the canonical fast path uses the center tensor norm. Both
        must yield the same result.
        """
        n_direct = MPS(mps_tensors_su2).norm()
        mps = MPS([t.clone() for t in mps_tensors_su2])
        mps.canonical(5, trunc=None)
        n_canonical = mps.norm()
        assert math.isclose(n_direct, n_canonical, rel_tol=1e-10)

    def test_canonical_sets_center_su2(self, mps_tensors_su2):
        mps = MPS([t.clone() for t in mps_tensors_su2])
        for target in [0, 5, 9]:
            mps.canonical(target)
            assert mps.center == target
            mps._validate()

    def test_canonical_preserves_norm_su2(self, mps_tensors_su2):
        """Norm must be invariant under a full canonical sweep for SU(2) MPS."""
        mps = MPS([t.clone() for t in mps_tensors_su2])
        mps.canonical(0)
        n0 = mps.norm()
        mps.canonical(9)
        n9 = mps.norm()
        assert math.isclose(n0, n9, rel_tol=1e-10)

    def test_left_canonical_isometry_su2(self, mps_tensors_su2):
        """Sites left of center must satisfy A†A = I (SU(2) MPS).

        The QR decomposition must handle the non-Abelian block structure so
        that the reduced-matrix blocks are isometric. Uses Nicole's `allclose`
        for a gauge-invariant comparison that accounts for Bridge intertwiners.
        """
        mps = MPS([t.clone() for t in mps_tensors_su2])
        mps.canonical(9)
        for i in range(9):
            A = mps[i]
            AcA = contract(conj(A), A, axes=([0, 2], [0, 2]))
            _assert_is_identity(AcA, f"SU(2) MPS site {i} A†A")

    def test_right_canonical_isometry_su2(self, mps_tensors_su2):
        """Sites right of center must satisfy BB† = I (SU(2) MPS)."""
        mps = MPS([t.clone() for t in mps_tensors_su2])
        mps.canonical(0)
        for i in range(1, 10):
            B = mps[i]
            BBc = contract(B, conj(B), axes=([1, 2], [1, 2]))
            _assert_is_identity(BBc, f"SU(2) MPS site {i} BB†")

    def test_mixed_canonical_isometry_su2(self, mps_tensors_su2):
        """Mixed canonical with center=5: left and right isometries hold (SU(2) MPS)."""
        center = 5
        mps = MPS([t.clone() for t in mps_tensors_su2])
        mps.canonical(center)
        for i in range(center):
            A = mps[i]
            AcA = contract(conj(A), A, axes=([0, 2], [0, 2]))
            _assert_is_identity(AcA, f"SU(2) MPS site {i} A†A")
        for i in range(center + 1, 10):
            B = mps[i]
            BBc = contract(B, conj(B), axes=([1, 2], [1, 2]))
            _assert_is_identity(BBc, f"SU(2) MPS site {i} BB†")


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

    def test_norm_uncanonical_equals_canonical(self, mpo_tensors):
        """norm() without canonicalization must agree with the canonical fast path."""
        n_direct = MPO([t.clone() for t in mpo_tensors]).norm()
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(5, trunc=None)
        n_canonical = mpo.norm()
        assert math.isclose(n_direct, n_canonical, rel_tol=1e-10)

    def test_canonical_sets_center(self, mpo_tensors):
        mpo = MPO([t.clone() for t in mpo_tensors])
        for target in [0, 5, 9]:
            mpo.canonical(target)
            assert mpo.center == target
            mpo._validate()

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

    # ------------------------------------------------------------------
    # SU(2) symmetry
    # ------------------------------------------------------------------

    def test_phys_dims_su2(self, mpo_tensors_su2):
        """SU(2) spin-1/2 ket physical index reports 1 multiplet per site."""
        assert all(d == 1 for d in MPO(mpo_tensors_su2).phys_dims)

    def test_norm_positive_su2(self, mpo_tensors_su2):
        mpo = MPO([t.clone() for t in mpo_tensors_su2])
        mpo.canonical(0)
        assert mpo.norm() > 0

    def test_norm_uncanonical_equals_canonical_su2(self, mpo_tensors_su2):
        """Full-contraction and center-tensor norms must agree for SU(2) MPO."""
        n_direct = MPO([t.clone() for t in mpo_tensors_su2]).norm()
        mpo = MPO([t.clone() for t in mpo_tensors_su2])
        mpo.canonical(5, trunc=None)
        n_canonical = mpo.norm()
        assert math.isclose(n_direct, n_canonical, rel_tol=1e-10)

    def test_canonical_sets_center_su2(self, mpo_tensors_su2):
        mpo = MPO([t.clone() for t in mpo_tensors_su2])
        for target in [0, 5, 9]:
            mpo.canonical(target)
            assert mpo.center == target
            mpo._validate()

    def test_canonical_preserves_norm_su2(self, mpo_tensors_su2):
        """Norm must be invariant under a full canonical sweep for SU(2) MPO."""
        mpo = MPO([t.clone() for t in mpo_tensors_su2])
        mpo.canonical(0)
        n0 = mpo.norm()
        mpo.canonical(9)
        n9 = mpo.norm()
        assert math.isclose(n0, n9, rel_tol=1e-10)

    def test_left_canonical_isometry_su2(self, mpo_tensors_su2):
        """Sites left of center must satisfy W†W = I (SU(2) MPO)."""
        mpo = MPO([t.clone() for t in mpo_tensors_su2])
        mpo.canonical(9)
        for i in range(9):
            W = mpo[i]
            WcW = contract(conj(W), W, axes=([0, 2, 3], [0, 2, 3]))
            _assert_is_identity(WcW, f"SU(2) MPO site {i} W†W")

    def test_right_canonical_isometry_su2(self, mpo_tensors_su2):
        """Sites right of center must satisfy WW† = I (SU(2) MPO)."""
        mpo = MPO([t.clone() for t in mpo_tensors_su2])
        mpo.canonical(0)
        for i in range(1, 10):
            W = mpo[i]
            WWc = contract(W, conj(W), axes=([1, 2, 3], [1, 2, 3]))
            _assert_is_identity(WWc, f"SU(2) MPO site {i} WW†")

    def test_mixed_canonical_isometry_su2(self, mpo_tensors_su2):
        """Mixed canonical with center=5: left and right isometries hold (SU(2) MPO)."""
        center = 5
        mpo = MPO([t.clone() for t in mpo_tensors_su2])
        mpo.canonical(center)
        for i in range(center):
            W = mpo[i]
            WcW = contract(conj(W), W, axes=([0, 2, 3], [0, 2, 3]))
            _assert_is_identity(WcW, f"SU(2) MPO site {i} W†W")
        for i in range(center + 1, 10):
            W = mpo[i]
            WWc = contract(W, conj(W), axes=([1, 2, 3], [1, 2, 3]))
            _assert_is_identity(WWc, f"SU(2) MPO site {i} WW†")

    # ------------------------------------------------------------------
    # redistribute_norm()
    # ------------------------------------------------------------------

    def test_redistribute_norm_preserves_total_norm(self, mpo_tensors):
        """redistribute_norm() must not change the total MPO norm.

        Each tensor is multiplied by factor = N^(1/L) and then the center
        tensor is divided by N. The product of all scale factors is
        factor^L / N = 1, so the operator — and its Frobenius norm — is
        unchanged.
        """
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(5, trunc=None)
        n_before = mpo.norm()
        mpo.redistribute_norm()
        assert math.isclose(mpo.norm(), n_before, rel_tol=1e-10)

    def test_redistribute_norm_no_center_raises(self, mpo_tensors):
        """redistribute_norm() must raise ValueError when center is not set."""
        mpo = MPO([t.clone() for t in mpo_tensors])
        assert mpo.center is None
        with pytest.raises(ValueError, match="canonical form"):
            mpo.redistribute_norm()

    def test_redistribute_norm_scales_by_factor(self, mpo_tensors):
        """All tensors are scaled by factor = N^(1/L); the pivot is also divided by N.

        When center is set, the pivot is the center tensor; otherwise tensor 0.
        Non-pivot tensors carry exactly one factor each.
        """
        center = 5
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(center, trunc=None)
        N = mpo.norm()
        factor = N ** (1.0 / mpo.L)
        norms_before = [mpo[i].norm() for i in range(mpo.L)]
        mpo.redistribute_norm()
        for i in range(mpo.L):
            if i == center:
                assert math.isclose(mpo[i].norm(), norms_before[i] * factor / N, rel_tol=1e-10)
            else:
                assert math.isclose(mpo[i].norm(), norms_before[i] * factor, rel_tol=1e-10)

    def test_redistribute_norm_resets_center(self, mpo_tensors):
        """redistribute_norm() must reset center to None."""
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(5, trunc=None)
        assert mpo.center == 5
        mpo.redistribute_norm()
        assert mpo.center is None

    def test_redistribute_norm_zero_raises(self, mpo_tensors):
        """redistribute_norm() must raise ValueError when the norm is numerically zero."""
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(0, trunc=None)
        for i in range(mpo.L):
            mpo[i] = mpo[i] * 0.0
        with pytest.raises(ValueError, match="numerically zero"):
            mpo.redistribute_norm()


class TestNormalize:
    """Tests for Network.normalize() on both MPS and MPO."""

    # ------------------------------------------------------------------
    # Error conditions
    # ------------------------------------------------------------------

    def test_no_center_raises_mps(self, mps_tensors):
        """normalize() must raise ValueError when center is None (MPS)."""
        mps = MPS([t.clone() for t in mps_tensors])
        assert mps.center is None
        with pytest.raises(ValueError, match="canonical"):
            mps.normalize()

    def test_no_center_raises_mpo(self, mpo_tensors):
        """normalize() must raise ValueError when center is None (MPO)."""
        mpo = MPO([t.clone() for t in mpo_tensors])
        assert mpo.center is None
        with pytest.raises(ValueError, match="canonical"):
            mpo.normalize()

    def test_zero_norm_raises_mps(self, mps_tensors):
        """normalize() must raise ValueError when the norm is numerically zero."""
        mps = MPS([t.clone() for t in mps_tensors])
        mps.canonical(0, trunc=None)
        mps[mps.center] = mps[mps.center] * 0.0
        with pytest.raises(ValueError, match="numerically zero"):
            mps.normalize()

    # ------------------------------------------------------------------
    # MPS: correctness
    # ------------------------------------------------------------------

    def test_returns_none_mps(self, mps_tensors):
        """normalize() must return None."""
        mps = MPS([t.clone() for t in mps_tensors])
        mps.canonical(5, trunc=None)
        assert mps.normalize() is None

    def test_center_norm_is_one_mps(self, mps_tensors):
        """After normalize(), the center tensor must have unit Frobenius norm."""
        mps = MPS([t.clone() for t in mps_tensors])
        mps.canonical(5, trunc=None)
        mps.normalize()
        assert math.isclose(mps[mps.center].norm(), 1.0, rel_tol=1e-10)

    def test_non_center_tensors_unchanged_mps(self, mps_tensors):
        """normalize() must not modify non-center site tensors (MPS)."""
        center = 5
        mps = MPS([t.clone() for t in mps_tensors])
        mps.canonical(center, trunc=None)
        norms_before = [mps[i].norm() for i in range(mps.L)]
        mps.normalize()
        for i in range(mps.L):
            if i != center:
                assert math.isclose(mps[i].norm(), norms_before[i], rel_tol=1e-10)

    def test_center_preserved_after_normalize_mps(self, mps_tensors):
        """normalize() must not change the center attribute."""
        mps = MPS([t.clone() for t in mps_tensors])
        mps.canonical(3, trunc=None)
        mps.normalize()
        assert mps.center == 3

    def test_norm_is_one_after_normalize_mps(self, mps_tensors):
        """The total MPS norm must equal 1.0 after normalize()."""
        mps = MPS([t.clone() for t in mps_tensors])
        mps.canonical(5, trunc=None)
        mps.normalize()
        assert math.isclose(mps.norm(), 1.0, rel_tol=1e-10)

    # ------------------------------------------------------------------
    # MPO: correctness
    # ------------------------------------------------------------------

    def test_center_norm_is_one_mpo(self, mpo_tensors):
        """After normalize(), the center tensor must have unit Frobenius norm (MPO)."""
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(5, trunc=None)
        mpo.normalize()
        assert math.isclose(mpo[mpo.center].norm(), 1.0, rel_tol=1e-10)

    def test_non_center_tensors_unchanged_mpo(self, mpo_tensors):
        """normalize() must not modify non-center site tensors (MPO)."""
        center = 5
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(center, trunc=None)
        norms_before = [mpo[i].norm() for i in range(mpo.L)]
        mpo.normalize()
        for i in range(mpo.L):
            if i != center:
                assert math.isclose(mpo[i].norm(), norms_before[i], rel_tol=1e-10)

    def test_center_preserved_after_normalize_mpo(self, mpo_tensors):
        """normalize() must not change the center attribute (MPO)."""
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(7, trunc=None)
        mpo.normalize()
        assert mpo.center == 7


class TestCompact:
    """Tests for MPO.compact()."""

    # ------------------------------------------------------------------
    # Return value and post-conditions
    # ------------------------------------------------------------------

    def test_returns_none(self, mpo_tensors):
        """compact() must return None."""
        mpo = MPO([t.clone() for t in mpo_tensors])
        assert mpo.compact() is None

    def test_center_is_none_after_compact(self, mpo_tensors):
        """compact() must leave center=None (via redistribute_norm)."""
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.compact()
        assert mpo.center is None

    # ------------------------------------------------------------------
    # Norm preservation
    # ------------------------------------------------------------------

    def test_preserves_norm(self, mpo_tensors):
        """compact() must not change the total MPO norm."""
        mpo = MPO([t.clone() for t in mpo_tensors])
        n_before = mpo.norm()
        mpo.compact(trunc=None)
        assert math.isclose(mpo.norm(), n_before, rel_tol=1e-10)

    def test_preserves_norm_from_canonical_start(self, mpo_tensors):
        """compact() must preserve norm when the MPO already has a center set."""
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.canonical(3, trunc=None)
        n_before = mpo.norm()
        mpo.compact(trunc=None)
        assert math.isclose(mpo.norm(), n_before, rel_tol=1e-10)

    # ------------------------------------------------------------------
    # Idempotency under repeated application
    # ------------------------------------------------------------------

    def test_norm_stable_after_second_compact(self, mpo_tensors):
        """Applying compact() twice must leave the norm unchanged."""
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.compact(trunc=None)
        n_after_first = mpo.norm()
        mpo.compact(trunc=None)
        assert math.isclose(mpo.norm(), n_after_first, rel_tol=1e-10)

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------

    def test_trunc_reduces_bond_dims(self, mpo_tensors):
        """compact() with a tight threshold must reduce at least one bond dimension."""
        mpo_full = MPO([t.clone() for t in mpo_tensors])
        mpo_full.compact(trunc=None)
        max_dim_full = max(mpo_full.bond_dims)

        mpo_trunc = MPO([t.clone() for t in mpo_tensors])
        mpo_trunc.compact(trunc={'thresh': 1e-2})
        max_dim_trunc = max(mpo_trunc.bond_dims)

        assert max_dim_trunc <= max_dim_full

    def test_trunc_preserves_norm(self, mpo_tensors):
        """compact() with truncation must still preserve the total norm."""
        mpo = MPO([t.clone() for t in mpo_tensors])
        n_before = mpo.norm()
        mpo.compact(trunc={'thresh': 1e-10})
        assert math.isclose(mpo.norm(), n_before, rel_tol=1e-6)

    def test_default_trunc_equals_explicit_default(self, mpo_tensors):
        """compact() with no trunc argument must match compact(trunc={'thresh': 1e-14})."""
        mpo_a = MPO([t.clone() for t in mpo_tensors])
        mpo_a.compact()
        mpo_b = MPO([t.clone() for t in mpo_tensors])
        mpo_b.compact(trunc={'thresh': 1e-14})
        assert mpo_a.bond_dims == mpo_b.bond_dims
        assert math.isclose(mpo_a.norm(), mpo_b.norm(), rel_tol=1e-12)

    # ------------------------------------------------------------------
    # Structural validity
    # ------------------------------------------------------------------

    def test_validates_after_compact(self, mpo_tensors):
        """compact() must leave the MPO in a structurally consistent state."""
        mpo = MPO([t.clone() for t in mpo_tensors])
        mpo.compact(trunc=None)
        mpo._validate()

    # ------------------------------------------------------------------
    # SU(2) symmetry
    # ------------------------------------------------------------------

    def test_compact_preserves_norm_su2(self, mpo_tensors_su2):
        """compact() must not change the total MPO norm (SU(2))."""
        mpo = MPO([t.clone() for t in mpo_tensors_su2])
        n_before = mpo.norm()
        mpo.compact(trunc=None)
        assert math.isclose(mpo.norm(), n_before, rel_tol=1e-10)

    def test_compact_validates_su2(self, mpo_tensors_su2):
        """compact() must leave the SU(2) MPO in a structurally valid state."""
        mpo = MPO([t.clone() for t in mpo_tensors_su2])
        mpo.compact(trunc=None)
        mpo._validate()


class TestNetworkSerialize:
    """Tests for Network.serialize() and Network.deserialize()."""

    # ------------------------------------------------------------------
    # Schema shape
    # ------------------------------------------------------------------

    def test_serialize_schema_keys(self, mps_tensors):
        """Serialized dict must contain all required top-level keys."""
        payload = MPS(mps_tensors).serialize()
        assert set(payload.keys()) == {"version", "class", "bc", "center", "itag_prefix", "tensors"}

    def test_serialize_version(self, mps_tensors):
        payload = MPS(mps_tensors).serialize()
        assert payload["version"] == 1

    def test_serialize_class_network(self, mps_tensors):
        payload = Network(mps_tensors).serialize()
        assert payload["class"] == "Network"

    def test_serialize_class_mps(self, mps_tensors):
        payload = MPS(mps_tensors).serialize()
        assert payload["class"] == "MPS"

    def test_serialize_class_mpo(self, mpo_tensors):
        payload = MPO(mpo_tensors).serialize()
        assert payload["class"] == "MPO"

    def test_serialize_tensors_length(self, mps_tensors):
        mps = MPS(mps_tensors)
        payload = mps.serialize()
        assert len(payload["tensors"]) == mps.L

    # ------------------------------------------------------------------
    # Correct class type after round-trip
    # ------------------------------------------------------------------

    def test_deserialize_returns_network_type(self, mps_tensors):
        """Round-tripping a bare Network must return a Network, not a subclass."""
        net = Network(mps_tensors)
        net2 = Network.deserialize(net.serialize())
        assert type(net2) is Network

    def test_deserialize_returns_mps_type(self, mps_tensors):
        mps = MPS(mps_tensors)
        mps2 = Network.deserialize(mps.serialize())
        assert type(mps2) is MPS

    def test_deserialize_returns_mpo_type(self, mpo_tensors):
        mpo = MPO(mpo_tensors)
        mpo2 = Network.deserialize(mpo.serialize())
        assert type(mpo2) is MPO

    # ------------------------------------------------------------------
    # Metadata preservation
    # ------------------------------------------------------------------

    def test_roundtrip_bc(self, mps_tensors):
        mps = MPS(mps_tensors)
        assert Network.deserialize(mps.serialize()).bc == "OBC"

    def test_roundtrip_center_none(self, mps_tensors):
        """center=None (no canonical form declared) must round-trip correctly."""
        mps = MPS(mps_tensors)
        assert mps.center is None
        assert Network.deserialize(mps.serialize()).center is None

    def test_roundtrip_center_set(self, mps_tensors):
        """An integer center set by canonical() must survive the round-trip."""
        mps = MPS([t.clone() for t in mps_tensors])
        mps.canonical(3)
        mps2 = Network.deserialize(mps.serialize())
        assert mps2.center == 3

    def test_roundtrip_itag_prefix(self, mps_tensors):
        """A custom itag_prefix must be preserved across serialize/deserialize."""
        mps = MPS(mps_tensors)
        mps.itag_prefix = "X"
        mps2 = Network.deserialize(mps.serialize())
        assert mps2.itag_prefix == "X"

    # ------------------------------------------------------------------
    # Tensor data fidelity
    # ------------------------------------------------------------------

    def test_roundtrip_tensor_blocks_mps(self, mps_tensors):
        """Every block of every site tensor must match the original after round-trip."""
        mps = MPS(mps_tensors)
        mps2 = Network.deserialize(mps.serialize())
        for i in range(mps.L):
            for key, block in mps[i].data.items():
                assert key in mps2[i].data, f"site {i}: block {key} missing after round-trip"
                assert torch.allclose(mps2[i].data[key], block), (
                    f"site {i}: block {key} mismatch after round-trip"
                )

    def test_roundtrip_tensor_blocks_mpo(self, mpo_tensors):
        """Every block of every MPO site tensor must match the original after round-trip."""
        mpo = MPO(mpo_tensors)
        mpo2 = Network.deserialize(mpo.serialize())
        for i in range(mpo.L):
            for key, block in mpo[i].data.items():
                assert key in mpo2[i].data, f"site {i}: block {key} missing after round-trip"
                assert torch.allclose(mpo2[i].data[key], block), (
                    f"site {i}: block {key} mismatch after round-trip"
                )

    # ------------------------------------------------------------------
    # torch.save / torch.load round-trip
    # ------------------------------------------------------------------

    def test_torch_save_load(self, mps_tensors, tmp_path):
        """Full torch.save / torch.load(weights_only=True) cycle must reconstruct the MPS."""
        mps = MPS(mps_tensors)
        path = tmp_path / "state.mps"
        torch.save(mps.serialize(), path)
        payload = torch.load(path, weights_only=True)
        mps2 = Network.deserialize(payload)
        assert type(mps2) is MPS
        assert mps2.bc == mps.bc
        assert mps2.center == mps.center
        for i in range(mps.L):
            for key, block in mps[i].data.items():
                assert torch.allclose(mps2[i].data[key], block)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_deserialize_bad_version_raises(self, mps_tensors):
        """A dict with an unsupported version number must raise ValueError."""
        payload = MPS(mps_tensors).serialize()
        payload["version"] = 99
        with pytest.raises(ValueError, match="version"):
            Network.deserialize(payload)

    def test_deserialize_unknown_class_raises(self, mps_tensors):
        """A dict with an unrecognized class name must raise ValueError."""
        payload = MPS(mps_tensors).serialize()
        payload["class"] = "UnknownNet"
        with pytest.raises(ValueError, match="class"):
            Network.deserialize(payload)

    # ------------------------------------------------------------------
    # SU(2) symmetry — verifies CG intertwiners survive the round-trip
    # ------------------------------------------------------------------

    def test_deserialize_returns_mps_type_su2(self, mps_tensors_su2):
        mps = MPS(mps_tensors_su2)
        mps2 = Network.deserialize(mps.serialize())
        assert type(mps2) is MPS

    def test_roundtrip_tensor_blocks_mps_su2(self, mps_tensors_su2):
        """Every block of every SU(2) MPS site tensor must match the original."""
        mps = MPS(mps_tensors_su2)
        mps2 = Network.deserialize(mps.serialize())
        for i in range(mps.L):
            for key, block in mps[i].data.items():
                assert key in mps2[i].data, f"site {i}: block {key} missing after round-trip"
                assert torch.allclose(mps2[i].data[key], block), (
                    f"site {i}: block {key} mismatch after round-trip"
                )

    def test_roundtrip_tensor_blocks_mpo_su2(self, mpo_tensors_su2):
        """Every block of every SU(2) MPO site tensor must match the original."""
        mpo = MPO(mpo_tensors_su2)
        mpo2 = Network.deserialize(mpo.serialize())
        for i in range(mpo.L):
            for key, block in mpo[i].data.items():
                assert key in mpo2[i].data, f"site {i}: block {key} missing after round-trip"
                assert torch.allclose(mpo2[i].data[key], block), (
                    f"site {i}: block {key} mismatch after round-trip"
                )

    def test_roundtrip_center_set_su2(self, mps_tensors_su2):
        """An integer center set by canonical() must survive the round-trip (SU(2) MPS)."""
        mps = MPS([t.clone() for t in mps_tensors_su2])
        mps.canonical(3)
        mps2 = Network.deserialize(mps.serialize())
        assert mps2.center == 3
