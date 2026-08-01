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


"""Comprehensive tests for NormalMPO and thermal_mpo.

All physics tests use a product Hamiltonian H = h * Σ_i n_i (no
inter-site coupling) on L sites of spin-1/2 U(1), where n_i is the
number operator (eigenvalues 0 and 1 per site). For this model:

- Spectrum: eigenvalue ``E_m = h * m`` with degeneracy ``C(L, m)``.
- Moments: ``Tr[H^k] = Σ_{m=0}^{L} C(L, m) * (h*m)^k``.
- Partition function: ``Z(β) = (1 + e^{-βh})^L``.
- Thermal energy: ``⟨H⟩_β = L * h * e^{-βh} / (1 + e^{-βh})``.

All reference values are computed analytically, avoiding the need for
a `to_dense()` conversion.
"""

from __future__ import annotations

import math

import pytest

from nicole import Direction, load_space, identity

from alice.network import (
    Interaction1Site,
    Interaction2Site,
    NormalMPO,
    build_hamiltonian,
    observe,
    thermal_mpo,
)
from alice.physics.system import build_fermionic
from alice.network.thermal import _identity_mpo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_product_h(Spc, Op, L, h=1.0):
    """Build H = h * Σ_i n_i as an MPO.

    Returns ``(H_mpo, Spc)`` where ``H_mpo`` is the plain `MPO` and ``Spc``
    is the physical index.
    """
    Sz = Op['Sz']
    I_op = identity(Spc)
    I_op.retag([0, 1], list(Sz.itags))
    n_op = Sz + I_op * 0.5  # eigenvalues 0 (down), 1 (up)

    n_full = n_op.clone()
    n_full.insert_index(0, direction=Direction.IN, itag='L')
    n_full.insert_index(1, direction=Direction.OUT, itag='R')

    interactions = [
        Interaction1Site(site=i, tnsr=n_full.clone(), cpl=h)
        for i in range(L)
    ]
    return build_hamiltonian(interactions, L, Spc)


def _tr_H_power(L, h, k):
    """Compute ``Tr[H^k]`` analytically for H = h * Σ_i n_i."""
    return sum(math.comb(L, m) * (h * m) ** k for m in range(L + 1))


def _Z_exact(L, h, beta):
    """Exact partition function ``Z(β) = (1 + e^{-βh})^L``."""
    return (1.0 + math.exp(-beta * h)) ** L


def _E_exact(L, h, beta):
    """Exact thermal energy ``⟨H⟩_β = L h e^{-βh} / (1 + e^{-βh})``."""
    return L * h * math.exp(-beta * h) / (1.0 + math.exp(-beta * h))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def spin_u1():
    """Spin-1/2 U(1) space and operators (module scope for performance)."""
    return load_space('Spin', 'U1', {'J': 0.5})


@pytest.fixture
def product_h_L2(spin_u1):
    """Product Hamiltonian H = 1 * Σ n_i, L=2."""
    Spc, Op = spin_u1
    return _make_product_h(Spc, Op, L=2, h=1.0), Spc


@pytest.fixture
def product_h_L4(spin_u1):
    """Product Hamiltonian H = 1 * Σ n_i, L=4."""
    Spc, Op = spin_u1
    return _make_product_h(Spc, Op, L=4, h=1.0), Spc


# ---------------------------------------------------------------------------
# Tests: NormalMPO construction
# ---------------------------------------------------------------------------

class TestNormalMPOConstruction:
    """Tests for `from_mpo`, `compact()`, `__mul__`, and `__rmul__`."""

    def test_from_mpo_unit_norm(self, product_h_L2):
        """After `from_mpo`, the internal MPO has unit Frobenius norm."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        assert math.isclose(H_n.norm(), 1.0, rel_tol=1e-10)

    def test_from_mpo_scale_equals_original_norm(self, product_h_L2):
        """``scale`` returned by `from_mpo` equals the original MPO norm."""
        H_mpo, Spc = product_h_L2
        original_norm = H_mpo.norm()
        H_n = NormalMPO.from_mpo(H_mpo)
        assert math.isclose(H_n.scale, original_norm, rel_tol=1e-10)

    def test_from_mpo_does_not_modify_input(self, product_h_L2):
        """``from_mpo`` leaves the source MPO unchanged."""
        H_mpo, Spc = product_h_L2
        norm_before = H_mpo.norm()
        NormalMPO.from_mpo(H_mpo)
        assert math.isclose(H_mpo.norm(), norm_before, rel_tol=1e-14)

    def test_compact_restores_unit_norm(self, product_h_L4):
        """After `compact()` on a raw product MPO, ``norm() ≈ 1``."""
        H_mpo, Spc = product_h_L4
        H_n = NormalMPO.from_mpo(H_mpo)
        I_n = NormalMPO.from_mpo(_identity_mpo(Spc, H_mpo.L))
        # Raw product (not canonical, norm ≠ 1)
        raw = H_n @ I_n
        raw.compact()
        assert math.isclose(raw.norm(), 1.0, rel_tol=1e-9)

    def test_compact_scale_consistent_with_trace(self, product_h_L2):
        """After `compact()`, ``scale * norm`` is consistent with trace."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        I_n = NormalMPO.from_mpo(_identity_mpo(Spc, H_mpo.L))
        raw = H_n @ I_n
        # Trace of H @ I should equal trace of H (before compact)
        tr_before_compact = raw.trace()
        tr_H = H_n.trace()
        raw.compact()
        # After compact, trace should be preserved
        assert math.isclose(raw.trace(), tr_H, rel_tol=1e-8)
        assert math.isclose(raw.trace(), tr_before_compact, rel_tol=1e-8)

    def test_mul_scales_scale_only(self, product_h_L2):
        """``c * H`` has ``scale = c * H.scale`` and identical tensors."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        c = 3.7
        scaled = c * H_n
        assert math.isclose(scaled.scale, c * H_n.scale, rel_tol=1e-14)
        # Tensors are unchanged (same data as original)
        from nicole import allclose
        for i in range(H_n.L):
            assert allclose(scaled[i], H_n[i])

    def test_rmul_same_as_mul(self, product_h_L2):
        """``c * H`` and ``H * c`` give identical results."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        c = 2.5
        assert math.isclose((c * H_n).scale, (H_n * c).scale, rel_tol=1e-14)

    def test_scale_property_is_readonly_float(self, product_h_L2):
        """``scale`` property returns a float."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        assert isinstance(H_n.scale, float)


# ---------------------------------------------------------------------------
# Tests: log-scale representation (log_scale, scale_by, sign-in-tensor)
# ---------------------------------------------------------------------------

class TestNormalMPOLogScale:
    """Tests for `log_scale`, `scale_by`, sign-folded-into-tensor, and overflow safety."""

    def test_log_scale_consistent_with_scale(self, product_h_L2):
        """`exp(log_scale)` reconstructs `scale` at moderate magnitude."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        assert math.isclose(math.exp(H_n.log_scale), H_n.scale, rel_tol=1e-12)

    def test_scale_by_matches_manual_multiplication(self, product_h_L2):
        """`scale_by` gives the same result as multiplying the raw magnitude."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        original_scale = H_n.scale
        H_n.scale_by(math.log(2.5))
        assert math.isclose(H_n.scale, original_scale * 2.5, rel_tol=1e-12)

    def test_mul_negative_scalar_folds_sign_into_tensor(self, product_h_L2):
        """`c * H` for `c < 0` flips the sign observed via `trace()`.

        `.scale` stays a non-negative magnitude, since sign lives in the
        tensor data and only shows up once it is actually contracted, e.g.
        via `trace()`.
        """
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        c = -2.0
        scaled = c * H_n
        assert scaled.scale >= 0.0
        assert math.isclose(scaled.scale, abs(c) * H_n.scale, rel_tol=1e-14)
        assert math.isclose(scaled.trace(), c * H_n.trace(), rel_tol=1e-10)

    def test_double_negative_mul_restores_sign(self, product_h_L2):
        """`(-1) * ((-1) * H)` has the same trace as `H` (sign flips cancel)."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        twice_negated = -1.0 * (-1.0 * H_n)
        assert math.isclose(twice_negated.trace(), H_n.trace(), rel_tol=1e-10)

    def test_repeated_squaring_overflows_scale_but_not_log_scale(self, product_h_L2):
        """Driving `.scale` to `inf` via repeated squaring keeps `log_scale` finite.

        Reproduces, at unit-test speed, the class of bug behind XTRG's
        β=256 overflow: repeated squaring (`@` + `compact()`) eventually
        pushes the raw magnitude past float64 range, but `log_scale` (and
        `log_trace()`) must remain finite and internally consistent
        throughout.
        """
        H_mpo, Spc = product_h_L2
        rho = NormalMPO.from_mpo(H_mpo)
        # Even starting from an O(1) magnitude, ~40 squarings (roughly
        # doubling log_scale each time) is far more than enough to exceed
        # float64's ~1.8e308 range.
        for _ in range(40):
            rho = rho @ rho
            rho.compact()
        assert math.isinf(rho.scale)
        assert math.isfinite(rho.log_scale)
        log_abs, sign = rho.log_trace()
        assert math.isfinite(log_abs)
        assert sign in (1.0, -1.0, 0.0)


# ---------------------------------------------------------------------------
# Tests: trace()
# ---------------------------------------------------------------------------

class TestNormalMPOTrace:
    """Tests for `trace()`."""

    def test_identity_trace_L2(self, spin_u1):
        """Trace of the identity MPO equals d^L (total Hilbert space dim)."""
        Spc, _ = spin_u1
        L = 2
        I_n = NormalMPO.from_mpo(_identity_mpo(Spc, L))
        # d=2 (spin-1/2), d^L = 4
        assert math.isclose(I_n.trace(), 4.0, rel_tol=1e-10)

    def test_identity_trace_L4(self, spin_u1):
        """Trace of the identity MPO equals d^L for L=4."""
        Spc, _ = spin_u1
        L = 4
        I_n = NormalMPO.from_mpo(_identity_mpo(Spc, L))
        assert math.isclose(I_n.trace(), 16.0, rel_tol=1e-10)

    def test_product_h_trace_L2(self, product_h_L2):
        """``Tr[H] = Σ_m C(L,m) m*h`` for a product Hamiltonian."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        L, h = 2, 1.0
        expected = _tr_H_power(L, h, k=1)
        assert math.isclose(H_n.trace(), expected, rel_tol=1e-9)

    def test_product_h_trace_L4(self, product_h_L4):
        """``Tr[H] = Σ_m C(L,m) m*h`` for L=4."""
        H_mpo, Spc = product_h_L4
        H_n = NormalMPO.from_mpo(H_mpo)
        L, h = 4, 1.0
        expected = _tr_H_power(L, h, k=1)
        assert math.isclose(H_n.trace(), expected, rel_tol=1e-9)

    def test_scaled_trace(self, product_h_L2):
        """``Tr[c * H] = c * Tr[H]``."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        c = 4.0
        assert math.isclose((c * H_n).trace(), c * H_n.trace(), rel_tol=1e-10)

    def test_trace_after_compact(self, product_h_L4):
        """``trace()`` is preserved through `compact()`."""
        H_mpo, Spc = product_h_L4
        H_n = NormalMPO.from_mpo(H_mpo)
        tr_before = H_n.trace()
        H_n.compact()
        tr_after = H_n.trace()
        assert math.isclose(tr_before, tr_after, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Tests: __matmul__ consistency
# ---------------------------------------------------------------------------

class TestNormalMPOMatmul:
    """Consistency tests for `__matmul__` via trace checks."""

    def test_right_identity(self, product_h_L2):
        """``Tr[H @ I] ≈ Tr[H]``."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        I_n = NormalMPO.from_mpo(_identity_mpo(Spc, H_mpo.L))
        prod = H_n @ I_n
        prod.compact()
        assert math.isclose(prod.trace(), H_n.trace(), rel_tol=1e-8)

    def test_left_identity(self, product_h_L2):
        """``Tr[I @ H] ≈ Tr[H]``."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        I_n = NormalMPO.from_mpo(_identity_mpo(Spc, H_mpo.L))
        prod = I_n @ H_n
        prod.compact()
        assert math.isclose(prod.trace(), H_n.trace(), rel_tol=1e-8)

    def test_power_moment_L2(self, product_h_L2):
        """``Tr[H²] = Σ_m C(L,m) (hm)²`` (L=2, h=1)."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        L, h = 2, 1.0
        HH = H_n @ H_n
        HH.compact()
        expected = _tr_H_power(L, h, k=2)
        assert math.isclose(HH.trace(), expected, rel_tol=1e-8)

    def test_power_moment_L4(self, product_h_L4):
        """``Tr[H²] = Σ_m C(L,m) (hm)²`` (L=4, h=1)."""
        H_mpo, Spc = product_h_L4
        H_n = NormalMPO.from_mpo(H_mpo)
        L, h = 4, 1.0
        HH = H_n @ H_n
        HH.compact()
        expected = _tr_H_power(L, h, k=2)
        assert math.isclose(HH.trace(), expected, rel_tol=1e-8)

    def test_cubic_moment_L2(self, product_h_L2):
        """``Tr[H³] = Σ_m C(L,m) (hm)³`` (L=2, h=1)."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        L, h = 2, 1.0
        HHH = H_n @ H_n @ H_n
        HHH.compact()
        expected = _tr_H_power(L, h, k=3)
        assert math.isclose(HHH.trace(), expected, rel_tol=1e-8)

    def test_associativity_L2(self, product_h_L2):
        """``Tr[(H @ H) @ H] = Tr[H @ (H @ H)]``."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        left = (H_n @ H_n) @ H_n
        left.compact()
        right = H_n @ (H_n @ H_n)
        right.compact()
        assert math.isclose(left.trace(), right.trace(), rel_tol=1e-8)

    def test_associativity_L4(self, product_h_L4):
        """``Tr[(H @ H) @ H] = Tr[H @ (H @ H)]`` for L=4."""
        H_mpo, Spc = product_h_L4
        H_n = NormalMPO.from_mpo(H_mpo)
        left = (H_n @ H_n) @ H_n
        left.compact()
        right = H_n @ (H_n @ H_n)
        right.compact()
        assert math.isclose(left.trace(), right.trace(), rel_tol=1e-7)

    def test_scale_multiplicative(self, product_h_L2):
        """``(c * H) @ (d * I)`` has scale consistent with trace."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        I_n = NormalMPO.from_mpo(_identity_mpo(Spc, H_mpo.L))
        c, d = 2.0, 3.0
        prod = (c * H_n) @ (d * I_n)
        prod.compact()
        # Tr[(cH)(dI)] = c*d*Tr[H]
        expected = c * d * H_n.trace()
        assert math.isclose(prod.trace(), expected, rel_tol=1e-8)


# ---------------------------------------------------------------------------
# Tests: __add__ consistency
# ---------------------------------------------------------------------------

class TestNormalMPOAdd:
    """Consistency tests for `__add__` via trace checks."""

    def test_linearity_L2(self, product_h_L2):
        """``Tr[H + I] = Tr[H] + Tr[I]``."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        I_n = NormalMPO.from_mpo(_identity_mpo(Spc, H_mpo.L))
        S = H_n + I_n
        S.compact()
        expected = H_n.trace() + I_n.trace()
        assert math.isclose(S.trace(), expected, rel_tol=1e-8)

    def test_linearity_L4(self, product_h_L4):
        """``Tr[H + I] = Tr[H] + Tr[I]`` for L=4."""
        H_mpo, Spc = product_h_L4
        H_n = NormalMPO.from_mpo(H_mpo)
        I_n = NormalMPO.from_mpo(_identity_mpo(Spc, H_mpo.L))
        S = H_n + I_n
        S.compact()
        expected = H_n.trace() + I_n.trace()
        assert math.isclose(S.trace(), expected, rel_tol=1e-8)

    def test_scalar_consistency_L2(self, product_h_L2):
        """``Tr[c*H + H] = (c+1) * Tr[H]``."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        c = 2.5
        S = c * H_n + H_n
        S.compact()
        expected = (c + 1.0) * H_n.trace()
        assert math.isclose(S.trace(), expected, rel_tol=1e-8)

    def test_scalar_consistency_negative_coeff(self, product_h_L2):
        """``Tr[H + (-0.5)*I] = Tr[H] - 0.5*Tr[I]`` (negative coefficient)."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        I_n = NormalMPO.from_mpo(_identity_mpo(Spc, H_mpo.L))
        c = -0.5
        S = H_n + c * I_n
        S.compact()
        expected = H_n.trace() + c * I_n.trace()
        assert math.isclose(S.trace(), expected, rel_tol=1e-8)

    def test_distributivity_L2(self, product_h_L2):
        """``Tr[(H + I) @ H] = Tr[H @ H] + Tr[I @ H]``."""
        H_mpo, Spc = product_h_L2
        H_n = NormalMPO.from_mpo(H_mpo)
        I_n = NormalMPO.from_mpo(_identity_mpo(Spc, H_mpo.L))

        lhs = (H_n + I_n) @ H_n
        lhs.compact()

        rhs_1 = H_n @ H_n
        rhs_1.compact()
        rhs_2 = I_n @ H_n
        rhs_2.compact()

        expected = rhs_1.trace() + rhs_2.trace()
        assert math.isclose(lhs.trace(), expected, rel_tol=1e-7)

    def test_add_with_scaled_term(self, product_h_L4):
        """``Tr[H + 2*H] = 3 * Tr[H]`` for L=4."""
        H_mpo, Spc = product_h_L4
        H_n = NormalMPO.from_mpo(H_mpo)
        S = H_n + 2.0 * H_n
        S.compact()
        expected = 3.0 * H_n.trace()
        assert math.isclose(S.trace(), expected, rel_tol=1e-7)


# ---------------------------------------------------------------------------
# Tests: thermal_mpo physics
# ---------------------------------------------------------------------------

class TestThermalMPO:
    """Physics correctness tests for `thermal_mpo`."""

    _BETA_VALS = [0.1, 0.5, 1.0]
    _ORDER = 25   # sufficient for convergence up to β=1.0, L=4 (max eigenvalue 4)

    @pytest.mark.parametrize('beta', _BETA_VALS)
    def test_partition_function_L2(self, product_h_L2, beta):
        """``Z(β) = (1 + e^{-βh})^L`` for L=2."""
        H_mpo, Spc = product_h_L2
        L, h = 2, 1.0
        rho = thermal_mpo(H_mpo, beta, self._ORDER, Spc)
        Z = rho.trace()
        assert math.isclose(Z, _Z_exact(L, h, beta), rel_tol=1e-8)

    @pytest.mark.parametrize('beta', _BETA_VALS)
    def test_partition_function_L4(self, product_h_L4, beta):
        """``Z(β) = (1 + e^{-βh})^L`` for L=4."""
        H_mpo, Spc = product_h_L4
        L, h = 4, 1.0
        rho = thermal_mpo(H_mpo, beta, self._ORDER, Spc)
        Z = rho.trace()
        assert math.isclose(Z, _Z_exact(L, h, beta), rel_tol=1e-8)

    @pytest.mark.parametrize('beta', _BETA_VALS)
    def test_thermal_energy_L2(self, product_h_L2, beta):
        """``⟨H⟩_β = L h e^{-βh} / (1 + e^{-βh})`` for L=2."""
        H_mpo, Spc = product_h_L2
        L, h = 2, 1.0
        rho = thermal_mpo(H_mpo, beta, self._ORDER, Spc)
        E = observe(rho, H_mpo)
        assert math.isclose(E, _E_exact(L, h, beta), rel_tol=1e-8)

    @pytest.mark.parametrize('beta', _BETA_VALS)
    def test_thermal_energy_L4(self, product_h_L4, beta):
        """``⟨H⟩_β = L h e^{-βh} / (1 + e^{-βh})`` for L=4."""
        H_mpo, Spc = product_h_L4
        L, h = 4, 1.0
        rho = thermal_mpo(H_mpo, beta, self._ORDER, Spc)
        E = observe(rho, H_mpo)
        assert math.isclose(E, _E_exact(L, h, beta), rel_tol=1e-8)

    def test_z_order1_approximation(self, product_h_L2):
        """First-order Taylor gives ``Z ≈ d^L - β Tr[H]`` for small β."""
        H_mpo, Spc = product_h_L2
        L, h = 2, 1.0
        beta = 0.05
        rho = thermal_mpo(H_mpo, beta, 1, Spc)
        Z = rho.trace()
        d = 2  # spin-1/2
        Z_order1 = d ** L - beta * _tr_H_power(L, h, k=1)
        assert math.isclose(Z, Z_order1, rel_tol=1e-10)

    def test_convergence_with_order(self, product_h_L2):
        """Partition function converges to the exact value as Taylor order grows."""
        H_mpo, Spc = product_h_L2
        L, h, beta = 2, 1.0, 0.5
        Z_exact_val = _Z_exact(L, h, beta)
        prev_err = float('inf')
        for order in [3, 6, 10, 15]:
            rho = thermal_mpo(H_mpo, beta, order, Spc)
            Z = rho.trace()
            err = abs(Z - Z_exact_val) / Z_exact_val
            assert err < prev_err or math.isclose(err, 0.0, abs_tol=1e-12), (
                f"Error did not decrease at order={order}: "
                f"prev={prev_err:.2e}, curr={err:.2e}"
            )
            prev_err = err

    def test_high_temperature_limit(self, product_h_L2):
        """At small β, the Taylor expansion quickly converges to the exact Z."""
        H_mpo, Spc = product_h_L2
        L, h = 2, 1.0
        beta = 0.01
        rho = thermal_mpo(H_mpo, beta, 4, Spc)
        Z = rho.trace()
        # At small β, order=4 is more than enough for machine-precision accuracy.
        assert math.isclose(Z, _Z_exact(L, h, beta), rel_tol=1e-10)

    def test_observe_dispatch_for_normal_mpo(self, product_h_L2):
        """``observe(rho, H)`` dispatches to the thermal branch when rho is a NormalMPO."""
        H_mpo, Spc = product_h_L2
        rho = thermal_mpo(H_mpo, 0.3, self._ORDER, Spc)
        E = observe(rho, H_mpo)
        assert isinstance(E, float)


# ---------------------------------------------------------------------------
# Fixture: spinless fermion U(1)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def ferm_u1():
    """Spinless fermion U(1) space and operators (module scope for performance)."""
    return build_fermionic(symmetry='U1')


# ---------------------------------------------------------------------------
# Tests: free fermion tight-binding chain
# ---------------------------------------------------------------------------

class TestFreeFermionThermal:
    """Physics tests for `thermal_mpo` on a spinless tight-binding chain.

    The Hamiltonian is H = -t Σ_i (c†_i c_{i+1} + h.c.) with OBC.
    Single-particle energies are ε_k = -2t cos(kπ/(L+1)).
    The exact partition function and thermal energy follow from free-fermion
    statistics: Z = ∏_k (1 + e^{-β ε_k}) and ⟨H⟩ = Σ_k ε_k n_F(ε_k).
    """

    _L = 4
    _t = 1.0
    _ORDER = 25

    def _exact_energies(self):
        return [
            -2 * self._t * math.cos(k * math.pi / (self._L + 1))
            for k in range(1, self._L + 1)
        ]

    def _Z_exact(self, beta):
        return math.prod(1 + math.exp(-beta * e) for e in self._exact_energies())

    def _E_exact(self, beta):
        return sum(e / (math.exp(beta * e) + 1) for e in self._exact_energies())

    @pytest.fixture(scope='class')
    def tight_binding_h(self, ferm_u1):
        """Tight-binding Hamiltonian MPO for an L-site chain."""
        Spc, Op = ferm_u1
        L = self._L
        interactions = [
            Interaction2Site(
                cpl=-self._t,
                leading_site=i,
                terminal_site=i + 1,
                leading_tnsr=Op['G4'].clone(),
                terminal_tnsr=Op['G4dag'].clone(),
            )
            for i in range(L - 1)
        ]
        return build_hamiltonian(interactions, L, Spc), Spc

    @pytest.mark.parametrize('beta', [0.5, 1.0])
    def test_partition_function(self, tight_binding_h, beta):
        """Partition function ``Tr[ρ]`` matches the exact free-fermion result."""
        H_mpo, Spc = tight_binding_h
        rho = thermal_mpo(H_mpo, beta, self._ORDER, Spc)
        assert math.isclose(rho.trace(), self._Z_exact(beta), rel_tol=1e-6)

    @pytest.mark.parametrize('beta', [0.5, 1.0])
    def test_thermal_energy(self, tight_binding_h, beta):
        """Thermal energy ``⟨H⟩_β`` matches the exact free-fermion result."""
        H_mpo, Spc = tight_binding_h
        rho = thermal_mpo(H_mpo, beta, self._ORDER, Spc)
        E = observe(rho, H_mpo)
        assert math.isclose(E, self._E_exact(beta), rel_tol=1e-4)
