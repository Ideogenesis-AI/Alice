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


"""Tests for alice.algorithm.tdvp.krylov.

The reference throughout is `torch.matrix_exp` applied to a dense rendering
of the same effective Hamiltonian. Building that rendering requires flattening
a block-sparse symmetric tensor into a plain vector, which `_flatten` and
`_unflatten` do by fixing an order on the surviving blocks — only the allowed
blocks appear, so the flattened space is the variational space and nothing
else.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch

from alice.algorithm.tdvp.krylov import inner_product, lanczos_expv
from alice.algorithm.tdvp.scheme_1s import matvec_1s

from .conftest import fresh_envs


# ---------------------------------------------------------------------------
# Dense rendering helpers
# ---------------------------------------------------------------------------

def _flatten(t):
    """Flatten the surviving blocks of `t` into a vector plus their key order."""
    keys = sorted(t.data.keys(), key=str)
    return np.concatenate([t.data[k].reshape(-1).numpy() for k in keys]), keys


def _unflatten(vec, ref, keys):
    """Rebuild a tensor shaped like `ref` from a flat vector."""
    out = ref.clone()
    offset = 0
    for k in keys:
        n = ref.data[k].numel()
        out.data[k] = torch.tensor(
            vec[offset:offset + n].reshape(tuple(ref.data[k].shape)),
            dtype=torch.complex128,
        )
        offset += n
    return out


def _dense(matvec_fn, ref):
    """Render `matvec_fn` as a dense matrix in the flattened block basis."""
    _, keys = _flatten(ref)
    n = sum(ref.data[k].numel() for k in keys)
    cols = [
        _flatten(matvec_fn(_unflatten(np.eye(n, dtype=np.complex128)[j], ref, keys)))[0]
        for j in range(n)
    ]
    return np.stack(cols, axis=1)


@pytest.fixture
def site_operator(heisenberg_L2):
    """A 1-site effective Hamiltonian, its dense matrix, and a seed vector."""
    mps, mpo = heisenberg_L2
    env_left, env_right = fresh_envs(mps, mpo)
    mv = partial(matvec_1s, W=mpo[0], E_left=env_left[0], E_right=env_right[0])
    return mv, _dense(mv, mps[0]), mps[0]


# ---------------------------------------------------------------------------
# inner_product
# ---------------------------------------------------------------------------

class TestInnerProduct:
    """Tests for the complex inner product helper."""

    def test_norm_squared_is_real_and_positive(self, heisenberg_L2):
        """⟨v|v⟩ is real, positive, and equals ‖v‖²."""
        mps, _ = heisenberg_L2
        v = mps[0]
        ip = inner_product(v, v)
        assert abs(ip.imag) < 1e-14
        assert ip.real > 0.0
        assert abs(ip.real - v.norm() ** 2) < 1e-12

    def test_conjugate_symmetry(self, heisenberg_L2):
        """⟨a|b⟩ = conj(⟨b|a⟩), which requires the bra to be conjugated."""
        mps, mpo = heisenberg_L2
        env_left, env_right = fresh_envs(mps, mpo)
        a = mps[0]
        b = matvec_1s(a, mpo[0], env_left[0], env_right[0])
        assert abs(inner_product(a, b) - inner_product(b, a).conjugate()) < 1e-13

    def test_antilinear_in_the_bra(self, heisenberg_L2):
        """Scaling the bra by `z` scales the result by `conj(z)`."""
        mps, _ = heisenberg_L2
        v = mps[0]
        z = 0.6 + 0.8j
        assert abs(inner_product(v * z, v) - z.conjugate() * inner_product(v, v)) < 1e-13


# ---------------------------------------------------------------------------
# lanczos_expv against torch.matrix_exp
# ---------------------------------------------------------------------------

class TestLanczosExpv:
    """Tests for the Krylov matrix exponential."""

    @pytest.mark.parametrize('dt', [0.01, 0.1, 1.0, 5.0])
    def test_matches_matrix_exp(self, site_operator, dt):
        """exp(-i dt H)v matches a dense torch.matrix_exp over a wide dt range."""
        mv, H, v = site_operator
        vec, _ = _flatten(v)
        reference = torch.matrix_exp(torch.tensor(-1j * dt * H)).numpy() @ vec

        out, _ = lanczos_expv(mv, v, dt, max_krylov=40, tol=1e-14)
        assert np.abs(_flatten(out)[0] - reference).max() < 1e-9

    def test_effective_hamiltonian_is_hermitian(self, site_operator):
        """The dense rendering is Hermitian, as Lanczos assumes."""
        _, H, _ = site_operator
        assert np.abs(H - H.conj().T).max() < 1e-13

    def test_preserves_norm(self, site_operator):
        """A unitary propagator leaves the norm unchanged."""
        mv, _, v = site_operator
        out, _ = lanczos_expv(mv, v, 0.7, max_krylov=40, tol=1e-14)
        assert abs(out.norm() - v.norm()) < 1e-12

    def test_zero_dt_is_the_identity(self, site_operator):
        """exp(0) leaves the vector untouched."""
        mv, _, v = site_operator
        out, _ = lanczos_expv(mv, v, 0.0, max_krylov=40, tol=1e-14)
        assert np.abs(_flatten(out)[0] - _flatten(v)[0]).max() < 1e-13

    def test_reversible(self, site_operator):
        """Evolving by +dt then -dt returns the original vector."""
        mv, _, v = site_operator
        forward, _ = lanczos_expv(mv, v, 0.4, max_krylov=40, tol=1e-14)
        back, _ = lanczos_expv(mv, forward, -0.4, max_krylov=40, tol=1e-14)
        assert np.abs(_flatten(back)[0] - _flatten(v)[0]).max() < 1e-12

    def test_linear_in_the_vector(self, site_operator):
        """The propagator is linear: U(z v) = z U(v)."""
        mv, _, v = site_operator
        z = 0.3 - 1.1j
        scaled, _ = lanczos_expv(mv, v * z, 0.3, max_krylov=40, tol=1e-14)
        plain, _ = lanczos_expv(mv, v, 0.3, max_krylov=40, tol=1e-14)
        assert np.abs(_flatten(scaled)[0] - z * _flatten(plain)[0]).max() < 1e-12

    def test_composition(self, site_operator):
        """U(dt1) ∘ U(dt2) equals U(dt1 + dt2) for a time-independent H."""
        mv, _, v = site_operator
        a, _ = lanczos_expv(mv, v, 0.25, max_krylov=40, tol=1e-14)
        ab, _ = lanczos_expv(mv, a, 0.35, max_krylov=40, tol=1e-14)
        direct, _ = lanczos_expv(mv, v, 0.6, max_krylov=40, tol=1e-14)
        assert np.abs(_flatten(ab)[0] - _flatten(direct)[0]).max() < 1e-11

    def test_output_keeps_the_index_structure(self, site_operator):
        """The propagated tensor is interchangeable with the input."""
        mv, _, v = site_operator
        out, _ = lanczos_expv(mv, v, 0.3, max_krylov=20, tol=1e-12)
        assert out.itags == v.itags
        assert [i.direction for i in out.indices] == [i.direction for i in v.indices]
        assert sorted(out.data.keys(), key=str) == sorted(v.data.keys(), key=str)

    def test_truncated_subspace_is_less_accurate(self, site_operator):
        """A one-dimensional Krylov space reports no convergence."""
        mv, _, v = site_operator
        _, error = lanczos_expv(mv, v, 1.0, max_krylov=1, tol=1e-14)
        # With a single direction the propagator is a pure phase on that
        # direction, so all its weight sits on the newest (only) vector.
        assert error == pytest.approx(1.0)

    def test_error_estimate_falls_with_subspace_size(self, site_operator):
        """Enlarging the Krylov space does not increase the reported error."""
        mv, _, v = site_operator
        errors = [lanczos_expv(mv, v, 1.0, max_krylov=k, tol=1e-16)[1]
                  for k in (1, 2, 3)]
        assert errors == sorted(errors, reverse=True)

    def test_eigenvector_picks_up_a_pure_phase(self, site_operator):
        """An eigenvector of H_eff only acquires exp(-i dt E)."""
        mv, H, v = site_operator
        evals, evecs = np.linalg.eigh(H)
        _, keys = _flatten(v)
        eigvec = _unflatten(evecs[:, 0].copy(), v, keys)

        dt = 0.9
        out, _ = lanczos_expv(mv, eigvec, dt, max_krylov=40, tol=1e-14)
        expected = np.exp(-1j * dt * evals[0]) * evecs[:, 0]
        assert np.abs(_flatten(out)[0] - expected).max() < 1e-11
