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


"""Tests for alice.algorithm.tgtrg.krylov.

The reference throughout is a dense `eigh` exponential of the same effective
Hamiltonian, rendered in the flattened block basis the fixtures provide. The
propagator being real, the reference is built from `exp(-tau Λ)` directly,
without a matrix-exponential routine.

Because `lanczos_expv` returns its answer factored as `exp(log_scale) * w`,
every comparison recombines the two — except the tests that exist precisely
to check that the factoring keeps `w` on the scale of the input.
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np
import pytest

from alice.algorithm.tgtrg.krylov import inner_product, lanczos_expv
from alice.algorithm.tgtrg.scheme_1s import matvec_1s

from .conftest import dense, flatten, fresh_envs, thermal_state, unflatten


@pytest.fixture
def site_operator(spinless_fermion_L2):
    """A 1-site effective Hamiltonian, its dense matrix, and a seed tensor."""
    mpo, spc = spinless_fermion_L2
    rho = thermal_state(mpo, spc, 0.1)
    env_left, env_right = fresh_envs(rho, mpo)
    mv = partial(matvec_1s, W=mpo[0], E_left=env_left[0], E_right=env_right[0])
    return mv, dense(mv, rho[0]), rho[0]


def _reference(H, v, tau):
    """Return `exp(-tau H) v` densely, via the spectral decomposition of H."""
    evals, evecs = np.linalg.eigh(H)
    vec, _ = flatten(v)
    return evecs @ (np.exp(-tau * evals) * (evecs.T @ vec))


def _combine(out, log_scale):
    """Recombine a factored propagator result into a dense vector."""
    return math.exp(log_scale) * flatten(out)[0]


# ---------------------------------------------------------------------------
# inner_product
# ---------------------------------------------------------------------------

class TestInnerProduct:
    """Tests for the Frobenius inner product helper."""

    def test_norm_squared_is_real_and_positive(self, spinless_fermion_L2):
        """⟨v|v⟩ is real, positive, and equals ‖v‖²."""
        mpo, spc = spinless_fermion_L2
        v = thermal_state(mpo, spc, 0.1)[0]
        ip = inner_product(v, v)
        assert abs(ip.imag) < 1e-14
        assert ip.real > 0.0
        assert ip.real == pytest.approx(v.norm() ** 2, rel=1e-12)

    def test_symmetric_for_a_real_tensor(self, site_operator):
        """⟨a|b⟩ = ⟨b|a⟩ when both tensors are real."""
        mv, _, a = site_operator
        b = mv(a)
        assert inner_product(a, b).real == pytest.approx(inner_product(b, a).real,
                                                         rel=1e-12)

    def test_linear_in_the_ket(self, site_operator):
        """Scaling the ket scales the product."""
        _, _, v = site_operator
        assert inner_product(v, v * 2.5).real == pytest.approx(
            2.5 * inner_product(v, v).real, rel=1e-12)


# ---------------------------------------------------------------------------
# The effective Hamiltonian itself
# ---------------------------------------------------------------------------

class TestEffectiveHamiltonian:
    """The property the Lanczos recurrence relies on."""

    def test_is_symmetric(self, site_operator):
        """H_eff is symmetric in the Choi inner product, as Lanczos assumes."""
        _, H, _ = site_operator
        assert np.abs(H - H.T).max() < 1e-12

    def test_is_real(self, site_operator):
        """A real Hamiltonian gives a real effective Hamiltonian."""
        _, H, _ = site_operator
        assert H.dtype == np.float64


# ---------------------------------------------------------------------------
# lanczos_expv against the dense exponential
# ---------------------------------------------------------------------------

class TestLanczosExpv:
    """Tests for the Krylov imaginary-time propagator."""

    @pytest.mark.parametrize('tau', [0.01, 0.1, 1.0, 5.0])
    def test_matches_dense_exponential(self, site_operator, tau):
        """exp(-tau H)v matches the dense reference over a wide tau range."""
        mv, H, v = site_operator
        reference = _reference(H, v, tau)

        out, log_scale, _ = lanczos_expv(mv, v, tau, max_krylov=40, tol=1e-14)
        assert np.abs(_combine(out, log_scale) - reference).max() < 1e-10 * max(
            1.0, np.abs(reference).max())

    @pytest.mark.parametrize('tau', [-0.05, -1.0])
    def test_matches_dense_exponential_backwards(self, site_operator, tau):
        """A negative tau heats, as the bond update asks it to."""
        mv, H, v = site_operator
        reference = _reference(H, v, tau)

        out, log_scale, _ = lanczos_expv(mv, v, tau, max_krylov=40, tol=1e-14)
        assert np.abs(_combine(out, log_scale) - reference).max() < 1e-10 * max(
            1.0, np.abs(reference).max())

    def test_stays_real(self, site_operator):
        """A real input stays real: nothing here promotes the density matrix."""
        mv, _, v = site_operator
        out, _, _ = lanczos_expv(mv, v, 0.3, max_krylov=20, tol=1e-12)
        assert not out.dtype.is_complex

    def test_zero_tau_is_the_identity(self, site_operator):
        """exp(0) leaves the tensor untouched and removes no scale."""
        mv, _, v = site_operator
        out, log_scale, _ = lanczos_expv(mv, v, 0.0, max_krylov=40, tol=1e-14)
        assert log_scale == pytest.approx(0.0)
        assert np.abs(flatten(out)[0] - flatten(v)[0]).max() < 1e-13

    def test_reversible(self, site_operator):
        """Cooling by tau then heating by tau returns the original tensor."""
        mv, _, v = site_operator
        fwd, log_f, _ = lanczos_expv(mv, v, 0.4, max_krylov=40, tol=1e-14)
        back, log_b, _ = lanczos_expv(mv, fwd, -0.4, max_krylov=40, tol=1e-14)
        combined = math.exp(log_f + log_b) * flatten(back)[0]
        assert np.abs(combined - flatten(v)[0]).max() < 1e-11

    def test_composition(self, site_operator):
        """exp(-t1 H) ∘ exp(-t2 H) equals exp(-(t1+t2) H)."""
        mv, _, v = site_operator
        a, log_a, _ = lanczos_expv(mv, v, 0.25, max_krylov=40, tol=1e-14)
        ab, log_ab, _ = lanczos_expv(mv, a, 0.35, max_krylov=40, tol=1e-14)
        direct, log_d, _ = lanczos_expv(mv, v, 0.6, max_krylov=40, tol=1e-14)
        lhs = math.exp(log_a + log_ab) * flatten(ab)[0]
        rhs = math.exp(log_d) * flatten(direct)[0]
        assert np.abs(lhs - rhs).max() < 1e-10 * max(1.0, np.abs(rhs).max())

    def test_linear_in_the_tensor(self, site_operator):
        """The propagator is linear: U(c v) = c U(v), scale factored alike."""
        mv, _, v = site_operator
        scaled, log_s, _ = lanczos_expv(mv, v * 3.0, 0.3, max_krylov=40, tol=1e-14)
        plain, log_p, _ = lanczos_expv(mv, v, 0.3, max_krylov=40, tol=1e-14)
        lhs = math.exp(log_s) * flatten(scaled)[0]
        rhs = 3.0 * math.exp(log_p) * flatten(plain)[0]
        assert np.abs(lhs - rhs).max() < 1e-11 * max(1.0, np.abs(rhs).max())

    def test_output_keeps_the_index_structure(self, site_operator):
        """The propagated tensor is interchangeable with the input."""
        mv, _, v = site_operator
        out, _, _ = lanczos_expv(mv, v, 0.3, max_krylov=20, tol=1e-12)
        assert out.itags == v.itags
        assert [i.direction for i in out.indices] == [i.direction for i in v.indices]
        assert sorted(out.data.keys(), key=str) == sorted(v.data.keys(), key=str)

    def test_eigenvector_picks_up_a_pure_factor(self, site_operator):
        """An eigenvector of H_eff only acquires exp(-tau E)."""
        mv, H, v = site_operator
        evals, evecs = np.linalg.eigh(H)
        _, keys = flatten(v)
        eigvec = unflatten(evecs[:, 0].copy(), v, keys)

        tau = 0.9
        out, log_scale, _ = lanczos_expv(mv, eigvec, tau, max_krylov=40, tol=1e-14)
        expected = math.exp(-tau * evals[0]) * evecs[:, 0]
        assert np.abs(_combine(out, log_scale) - expected).max() < 1e-10


# ---------------------------------------------------------------------------
# The factoring that keeps a large step representable
# ---------------------------------------------------------------------------

class TestScaleFactoring:
    """Tests for the magnitude held in `log_scale`."""

    def test_log_scale_is_the_magnitude_change(self, site_operator):
        """exp(log_scale) is ‖exp(-tau H)v‖ / ‖v‖, the whole magnitude."""
        mv, H, v = site_operator
        for tau in (0.3, 2.0, -0.5):
            _, log_scale, _ = lanczos_expv(mv, v, tau, max_krylov=40, tol=1e-14)
            expected = np.linalg.norm(_reference(H, v, tau)) / v.norm()
            assert math.exp(log_scale) == pytest.approx(expected, rel=1e-9)

    def test_returned_tensor_keeps_the_input_norm(self, site_operator):
        """‖w‖ = ‖v‖ exactly: the direction is all `w` carries."""
        mv, _, v = site_operator
        for tau in (0.5, 5.0, 50.0, -5.0):
            out, _, _ = lanczos_expv(mv, v, tau, max_krylov=40, tol=1e-14)
            assert out.norm() == pytest.approx(v.norm(), rel=1e-12)

    def test_huge_step_stays_finite(self, site_operator):
        """A step whose magnitude leaves float64 range still returns a direction.

        `exp(-tau E)` for `tau = 2000` on this spectrum is far outside
        float64 range — the regime the exponential stage of a cooling run
        reaches — so the magnitude can only live in the log.
        """
        mv, _, v = site_operator
        out, log_scale, _ = lanczos_expv(mv, v, 2000.0, max_krylov=40, tol=1e-14)
        assert out.norm() == pytest.approx(v.norm(), rel=1e-12)
        with pytest.raises(OverflowError):
            math.exp(log_scale)

    def test_tiny_tensor_does_not_collapse(self, site_operator):
        """A step that would underflow leaves a usable tensor, not a zero one.

        Subtracting a bare eigenvalue bounds the coefficients above but not
        below: with little overlap on the favored eigenvector they can all
        collapse together, and a sweep handed that tensor has nothing left to
        propagate. The normalization rules that out.
        """
        mv, _, v = site_operator
        out, _, _ = lanczos_expv(mv, v, -2000.0, max_krylov=40, tol=1e-14)
        assert out.norm() > 0.0
        assert math.isfinite(out.norm())


# ---------------------------------------------------------------------------
# Error estimate and termination
# ---------------------------------------------------------------------------

class TestErrorEstimate:
    """Tests for the reported Krylov error."""

    def test_single_direction_reports_no_information(self, site_operator):
        """A one-dimensional Krylov space puts all weight on its only vector."""
        mv, _, v = site_operator
        _, _, error = lanczos_expv(mv, v, 1.0, max_krylov=1, tol=1e-14)
        assert error == pytest.approx(1.0)

    def test_error_falls_with_subspace_size(self, site_operator):
        """Enlarging the Krylov space does not increase the reported error."""
        mv, _, v = site_operator
        errors = [lanczos_expv(mv, v, 1.0, max_krylov=k, tol=1e-16)[2]
                  for k in (1, 2, 3)]
        assert errors == sorted(errors, reverse=True)

    def test_accuracy_tracks_the_error_estimate(self, site_operator):
        """A smaller reported error means a result closer to the reference."""
        mv, H, v = site_operator
        reference = _reference(H, v, 1.0)
        deviations = []
        for k in (1, 2, 4):
            out, log_scale, _ = lanczos_expv(mv, v, 1.0, max_krylov=k, tol=1e-16)
            deviations.append(np.abs(_combine(out, log_scale) - reference).max())
        assert deviations == sorted(deviations, reverse=True)

    def test_rejects_an_empty_subspace(self, site_operator):
        """max_krylov must leave room for at least one direction."""
        mv, _, v = site_operator
        with pytest.raises(ValueError, match='max_krylov'):
            lanczos_expv(mv, v, 0.1, max_krylov=0)

    def test_rejects_a_zero_tensor(self, site_operator):
        """A zero tensor has no direction to build a Krylov space on."""
        mv, _, v = site_operator
        with pytest.raises(ValueError, match='zero norm'):
            lanczos_expv(mv, v * 0.0, 0.1)
