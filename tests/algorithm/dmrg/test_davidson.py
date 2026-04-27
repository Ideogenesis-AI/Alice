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


"""Tests for alice.algorithm.dmrg.davidson.

Nicole does not allow rank-1 tensors. We therefore represent test vectors as
rank-2 tensors with a trivial dim-1 left index (charge 0) and a dim-n right
index (charge 0). The single data block has shape (1, n) and stores the
actual vector in its second axis. The Davidson helpers (_inner_product, _axpy)
work on these in the same way they work on rank-3 MPS site tensors — the
einsum pattern is built dynamically.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from nicole import Direction, Tensor, load_space
from nicole.index import Index, Sector

from alice.algorithm.dmrg.davidson import _axpy, _inner_product, davidson


# ---------------------------------------------------------------------------
# Shared test-vector utilities (rank-2, charge-0 throughout)
# ---------------------------------------------------------------------------

def _indices_for(n: int, group):
    """Return (left_idx, right_idx) for a charge-0 rank-2 test vector of size n."""
    left = Index(direction=Direction.IN,  group=group, sectors=(Sector(charge=0, dim=1),))
    right = Index(direction=Direction.OUT, group=group, sectors=(Sector(charge=0, dim=n),))
    return left, right


def _make_vec(values: np.ndarray, group) -> Tensor:
    """Build a complex128 rank-2 test tensor from a 1-D numpy array.

    The block has shape (1, n); the vector lives in `block[0, :]`.
    Using complex128 avoids dtype mismatches when Nicole's `.conj()` promotes
    the tensor from real to complex during the Davidson iteration.
    """
    n = len(values)
    left, right = _indices_for(n, group)
    t = Tensor.zeros([left, right], itags=['l', 'r'], dtype=torch.complex128)
    _, block = next(iter(t.data.items()))
    block[0, :] = torch.tensor(values, dtype=torch.complex128)
    return t


def _mat_matvec(matrix: np.ndarray, group):
    """Return a matvec_fn that applies a dense symmetric matrix to a rank-2 test vector."""
    n = matrix.shape[0]
    assert matrix.shape == (n, n)
    A = matrix.astype(np.complex128)
    left, right = _indices_for(n, group)

    def mv(v: Tensor) -> Tensor:
        _, block = next(iter(v.data.items()))
        vec = block[0, :].numpy().astype(np.complex128)
        out_vec = A @ vec
        t = Tensor.zeros([left, right], itags=['l', 'r'], dtype=torch.complex128)
        _, blk_out = next(iter(t.data.items()))
        blk_out[0, :] = torch.tensor(out_vec, dtype=torch.complex128)
        return t

    return mv


def _get_group():
    Spc, _ = load_space("Spin", "U1", {"J": 0.5})
    return Spc.group


# ---------------------------------------------------------------------------
# _inner_product
# ---------------------------------------------------------------------------

class TestInnerProduct:
    """Tests for the _inner_product helper."""

    def test_self_inner_product_equals_norm_squared(self):
        """⟨v|v⟩ == ‖v‖²  (3² + 4² = 25)."""
        group = _get_group()
        v = _make_vec(np.array([3.0, 4.0]), group)
        ip = _inner_product(v, v)
        assert abs(ip.real - 25.0) < 1e-10

    def test_orthogonal_inner_product_is_zero(self):
        group = _get_group()
        v1 = _make_vec(np.array([1.0, 0.0]), group)
        v2 = _make_vec(np.array([0.0, 1.0]), group)
        ip = _inner_product(v1, v2)
        assert abs(ip) < 1e-10

    def test_inner_product_with_rank3_tensors(self, heisenberg_L2):
        """_inner_product also works on actual rank-3 MPS site tensors."""
        mps, _ = heisenberg_L2
        M = mps[0]
        ip = _inner_product(M, M)
        assert abs(ip.real - M.norm() ** 2) < 1e-9


# ---------------------------------------------------------------------------
# _axpy
# ---------------------------------------------------------------------------

class TestAxpy:
    """Tests for the _axpy helper."""

    def test_axpy_values(self):
        """y + 2*x == [2, 1] for x=[1,0], y=[0,1]."""
        group = _get_group()
        x = _make_vec(np.array([1.0, 0.0]), group)
        y = _make_vec(np.array([0.0, 1.0]), group)
        z = _axpy(2.0, x, y)
        _, block = next(iter(z.data.items()))
        np.testing.assert_allclose(block[0].numpy(), [2.0, 1.0], atol=1e-10)

    def test_axpy_zero_coefficient(self):
        """0 * x + y == y."""
        group = _get_group()
        x = _make_vec(np.array([1.0, 2.0]), group)
        y = _make_vec(np.array([3.0, 4.0]), group)
        z = _axpy(0.0, x, y)
        _, block_z = next(iter(z.data.items()))
        _, block_y = next(iter(y.data.items()))
        np.testing.assert_allclose(block_z[0].numpy(), block_y[0].numpy(), atol=1e-10)


# ---------------------------------------------------------------------------
# Davidson solver
# ---------------------------------------------------------------------------

class TestDavidson:
    """Tests for the davidson() function."""

    def test_2x2_converges_to_min_eigenvalue(self):
        """2×2 symmetric matrix: converges to known minimum eigenvalue."""
        group = _get_group()
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        expected_min = float(np.linalg.eigvalsh(A)[0])

        mv = _mat_matvec(A, group)
        v0 = _make_vec(np.array([1.0, 0.0]), group)

        theta, q = davidson(mv, v0, max_iter=50, tol=1e-10, max_subspace=10)
        assert abs(theta - expected_min) < 1e-8, (
            f"Davidson returned {theta}, expected {expected_min}"
        )

    def test_4x4_converges_to_min_eigenvalue(self):
        """4×4 symmetric matrix: converges to known minimum eigenvalue."""
        group = _get_group()
        rng = np.random.default_rng(42)
        B = rng.standard_normal((4, 4))
        A = B @ B.T + np.eye(4)
        expected_min = float(np.linalg.eigvalsh(A)[0])

        mv = _mat_matvec(A, group)
        v0 = _make_vec(np.ones(4) / 2.0, group)

        theta, q = davidson(mv, v0, max_iter=100, tol=1e-10, max_subspace=20)
        assert abs(theta - expected_min) < 1e-7

    def test_subspace_collapse_does_not_crash(self):
        """Davidson with max_subspace=1 forces a restart without crashing."""
        group = _get_group()
        A = np.array([[1.0, 0.5], [0.5, 2.0]])
        mv = _mat_matvec(A, group)
        v0 = _make_vec(np.array([1.0, 0.0]), group)

        theta, q = davidson(mv, v0, max_iter=50, tol=1e-9, max_subspace=1)
        expected_min = float(np.linalg.eigvalsh(A)[0])
        assert abs(theta - expected_min) < 1e-6

    def test_zero_initial_guess_raises(self):
        """A zero initial guess must raise ValueError."""
        group = _get_group()
        A = np.eye(2)
        mv = _mat_matvec(A, group)
        v0 = _make_vec(np.zeros(2), group)
        with pytest.raises(ValueError, match="zero norm"):
            davidson(mv, v0)

    def test_already_converged_on_eigenvector(self):
        """Starting exactly on an eigenvector converges in one iteration."""
        group = _get_group()
        # Diagonal matrix — any unit vector is an eigenvector.
        A = np.diag([1.0, 5.0])
        mv = _mat_matvec(A, group)
        v0 = _make_vec(np.array([1.0, 0.0]), group)  # eigenvector of eigenvalue 1

        theta, _ = davidson(mv, v0, max_iter=10, tol=1e-10, max_subspace=5)
        assert abs(theta - 1.0) < 1e-10
