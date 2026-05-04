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


"""Tests for alice.algorithm.dmrg.scheme_2s (build_bulk, matvec_2s, optimize_2site)."""

from __future__ import annotations

import torch

from alice.network import observe
from alice.algorithm.dmrg.davidson import _inner_product
from alice.algorithm.dmrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
)
from alice.algorithm.dmrg.scheme_2s import (
    build_bulk,
    discarded_weight,
    matvec_2s,
    optimize_2site,
    split_forward,
)


# ---------------------------------------------------------------------------
# build_bulk
# ---------------------------------------------------------------------------

class TestBuildBulk:
    """Tests for the build_bulk function."""

    def test_output_rank(self, heisenberg_L2):
        """build_bulk returns a rank-4 tensor."""
        mps, _ = heisenberg_L2
        theta = build_bulk(mps[0], mps[1])
        assert len(theta.indices) == 4

    def test_output_dims(self, heisenberg_L2):
        """Θ left/right bond dims match the MPS boundary dims; phys dims match each site."""
        mps, _ = heisenberg_L2
        theta = build_bulk(mps[0], mps[1])
        # Axis 0: ket_left of site 0.
        assert theta.indices[0].dim == mps[0].indices[0].dim
        # Axis 1: ket_right of site 1.
        assert theta.indices[1].dim == mps[1].indices[1].dim
        # Axis 2: physical of site 0.
        assert theta.indices[2].dim == mps[0].indices[2].dim
        # Axis 3: physical of site 1.
        assert theta.indices[3].dim == mps[1].indices[2].dim


# ---------------------------------------------------------------------------
# matvec_2s
# ---------------------------------------------------------------------------

class TestMatvec2s:
    """Tests for the matvec_2s function."""

    def _envs(self, mps, mpo):
        L = mps.L
        env_left = Environment(L)
        env_right = Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)
        return env_left, env_right

    def test_energy_matches_observe(self, heisenberg_L2):
        """⟨Θ|H_eff|Θ⟩ / ⟨Θ|Θ⟩ at bond (0,1) equals the normalised observe energy.

        For L=2 the bond tensor Θ covers the whole chain, so ⟨Θ|H_eff|Θ⟩ / ⟨Θ|Θ⟩
        equals ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ as computed by `observe`.
        """
        mps, mpo = heisenberg_L2
        env_left, env_right = self._envs(mps, mpo)

        theta = build_bulk(mps[0], mps[1])
        H_theta = matvec_2s(theta, mpo[0], mpo[1], env_left[0], env_right[1])
        energy = _inner_product(theta, H_theta).real / _inner_product(theta, theta).real

        obs = observe(mps, mpo)
        norm_sq = mps.norm() ** 2
        obs_normalised = obs / norm_sq
        assert abs(energy - obs_normalised) < 1e-9, (
            f"matvec_2s energy {energy} != normalised observe {obs_normalised}"
        )

    def test_hermiticity(self, heisenberg_L2):
        """⟨θ₁|H_eff|θ₂⟩ == ⟨θ₂|H_eff|θ₁⟩ (H_eff is Hermitian)."""
        mps, mpo = heisenberg_L2
        env_left, env_right = self._envs(mps, mpo)

        theta1 = build_bulk(mps[0], mps[1])
        # Use H_eff|θ₁⟩ as an independent second vector.
        theta2 = matvec_2s(theta1, mpo[0], mpo[1], env_left[0], env_right[1])

        H_theta2 = matvec_2s(theta2, mpo[0], mpo[1], env_left[0], env_right[1])
        H_theta1 = matvec_2s(theta1, mpo[0], mpo[1], env_left[0], env_right[1])

        lhs = _inner_product(theta1, H_theta2)
        rhs = _inner_product(theta2, H_theta1)
        assert abs(lhs - rhs) < 1e-9, (
            f"Hermiticity violated: ⟨θ₁|H|θ₂⟩={lhs}, ⟨θ₂|H|θ₁⟩={rhs}"
        )

    def test_output_shape_matches_input(self, heisenberg_L2):
        """matvec_2s output has the same axis dimensions as the input θ."""
        mps, mpo = heisenberg_L2
        env_left, env_right = self._envs(mps, mpo)

        theta = build_bulk(mps[0], mps[1])
        H_theta = matvec_2s(theta, mpo[0], mpo[1], env_left[0], env_right[1])

        assert len(H_theta.indices) == len(theta.indices)
        for idx_in, idx_out in zip(theta.indices, H_theta.indices):
            assert idx_in.dim == idx_out.dim


# ---------------------------------------------------------------------------
# optimize_2site
# ---------------------------------------------------------------------------

class TestOptimize2site:
    """Tests for the optimize_2site function."""

    def _envs(self, mps, mpo):
        L = mps.L
        env_left = Environment(L)
        env_right = Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)
        return env_left, env_right

    def test_energy_does_not_increase(self, heisenberg_L2):
        """After optimize_2site the returned energy ≤ initial Rayleigh quotient."""
        mps, mpo = heisenberg_L2
        env_left, env_right = self._envs(mps, mpo)

        theta = build_bulk(mps[0], mps[1])
        H_theta = matvec_2s(theta, mpo[0], mpo[1], env_left[0], env_right[1])
        energy_init = (
            _inner_product(theta, H_theta).real / _inner_product(theta, theta).real
        )

        davidson_opts = {'max_iter': 50, 'tol': 1e-10, 'max_subspace': 10}
        energy_opt, _, _ = optimize_2site(
            mps[0], mps[1],
            mpo[0], mpo[1], env_left[0], env_right[1],
            davidson_opts,
        )

        assert energy_opt <= energy_init + 1e-10, (
            f"optimize_2site increased energy: {energy_init} -> {energy_opt}"
        )

    def test_no_mutation_of_inputs(self, heisenberg_L2):
        """optimize_2site does not mutate M_i or M_{i+1}."""
        mps, mpo = heisenberg_L2
        env_left, env_right = self._envs(mps, mpo)

        before_0 = {k: v.clone() for k, v in mps[0].data.items()}
        before_1 = {k: v.clone() for k, v in mps[1].data.items()}

        davidson_opts = {'max_iter': 10, 'tol': 1e-8, 'max_subspace': 5}
        optimize_2site(
            mps[0], mps[1],
            mpo[0], mpo[1], env_left[0], env_right[1],
            davidson_opts,
        )

        for k, v_before in before_0.items():
            assert torch.allclose(mps[0].data[k], v_before), (
                f"optimize_2site mutated mps[0] at block {k}"
            )
        for k, v_before in before_1.items():
            assert torch.allclose(mps[1].data[k], v_before), (
                f"optimize_2site mutated mps[1] at block {k}"
            )


# ---------------------------------------------------------------------------
# discarded_weight
# ---------------------------------------------------------------------------

class TestDiscardedWeight:
    """Tests for the discarded_weight function."""

    def test_zero_when_no_truncation(self, heisenberg_L4):
        """discarded_weight returns 0.0 when trunc=None (nothing is discarded)."""
        mps, _ = heisenberg_L4
        theta = build_bulk(mps[1], mps[2])
        dw = discarded_weight(theta, trunc=None)
        assert dw == 0.0

    def test_non_negative(self, heisenberg_L4):
        """discarded_weight is always >= 0."""
        mps, _ = heisenberg_L4
        theta = build_bulk(mps[1], mps[2])
        dw = discarded_weight(theta, trunc={'thresh': 1e-15})
        assert dw >= 0.0

    def test_positive_under_aggressive_truncation(self, heisenberg_L4):
        """discarded_weight > 0 when nkeep=1 forces singular values to be dropped."""
        mps, _ = heisenberg_L4
        theta = build_bulk(mps[1], mps[2])
        dw = discarded_weight(theta, trunc={'nkeep': 1})
        assert dw > 0.0

    def test_consistent_with_split_bond_dim(self, heisenberg_L4):
        """When nkeep=1 discards weight, the split bond dimension is also 1."""
        mps, _ = heisenberg_L4
        theta = build_bulk(mps[1], mps[2])
        trunc = {'nkeep': 1}
        dw = discarded_weight(theta, trunc=trunc)
        itag = mps._bond_itag(2)
        U, R = split_forward(theta, itag, trunc)
        # The new bond (axis 1 of U) must have dimension 1.
        assert U.indices[1].dim == 1
        # And something was discarded.
        assert dw > 0.0
