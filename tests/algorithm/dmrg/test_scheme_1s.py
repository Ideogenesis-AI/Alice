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


"""Tests for alice.algorithm.dmrg.scheme_1s (matvec and optimize_1site)."""

from __future__ import annotations

import pytest

from alice.network import observe
from alice.algorithm.dmrg.davidson import _inner_product
from alice.algorithm.dmrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
)
from alice.algorithm.dmrg.scheme_1s import matvec, optimize_1site


# ---------------------------------------------------------------------------
# matvec: energy consistency with observe()
# ---------------------------------------------------------------------------

class TestMatvec:
    """Tests for the matvec function."""

    def test_energy_matches_observe_site0(self, heisenberg_L2):
        """⟨M|H_eff|M⟩ at site 0 equals observe(mps, mpo)."""
        mps, mpo = heisenberg_L2
        L = mps.L

        env_left = Environment(L)
        env_right = Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)

        M = mps[0]
        HM = matvec(M, mpo[0], env_left[0], env_right[0])
        energy = _inner_product(M, HM).real

        obs = observe(mps, mpo)
        assert abs(energy - obs) < 1e-9, f"matvec energy {energy} != observe {obs}"

    def test_hermiticity(self, heisenberg_L2):
        """⟨M|H_eff|H_eff|M⟩ is real (follows from H_eff being Hermitian).

        We use M1 = M and M2 = H_eff|M⟩ (no canonicalization on the fixture
        MPS) and verify ⟨M1|H|M2⟩ == ⟨M2|H|M1⟩*, which holds iff H is
        Hermitian.
        """
        mps, mpo = heisenberg_L2
        L = mps.L

        env_left = Environment(L)
        env_right = Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)

        M1 = mps[0]
        # Use H_eff|M1⟩ as a second independent vector (same index structure).
        M2 = matvec(M1, mpo[0], env_left[0], env_right[0])

        HM2 = matvec(M2, mpo[0], env_left[0], env_right[0])
        HM1 = matvec(M1, mpo[0], env_left[0], env_right[0])

        lhs = _inner_product(M1, HM2)
        rhs = _inner_product(M2, HM1)
        # For Hermitian H: ⟨M1|H|M2⟩ = ⟨M2|H|M1⟩* (or simply equal for real MPS).
        assert abs(lhs - rhs) < 1e-9, f"Hermiticity violated: ⟨v1|H|v2⟩={lhs}, ⟨v2|H|v1⟩={rhs}"

    def test_output_shape_matches_input(self, heisenberg_L2):
        """matvec output has the same axis layout as the input tensor."""
        mps, mpo = heisenberg_L2
        L = mps.L

        env_left = Environment(L)
        env_right = Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)

        M = mps[0]
        HM = matvec(M, mpo[0], env_left[0], env_right[0])
        # Same number of axes and same index dimensions.
        assert len(HM.indices) == len(M.indices)
        for idx_in, idx_out in zip(M.indices, HM.indices):
            assert idx_in.dim == idx_out.dim


# ---------------------------------------------------------------------------
# optimize_site
# ---------------------------------------------------------------------------

class TestOptimize1site:
    """Tests for the optimize_1site function."""

    def test_energy_does_not_increase(self, heisenberg_L2):
        """After optimize_1site the returned energy ≤ initial ⟨M|H_eff|M⟩."""
        mps, mpo = heisenberg_L2
        L = mps.L

        env_left = Environment(L)
        env_right = Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)

        M = mps[0]
        HM_init = matvec(M, mpo[0], env_left[0], env_right[0])
        energy_init = _inner_product(M, HM_init).real / _inner_product(M, M).real

        davidson_opts = {'max_iter': 50, 'tol': 1e-10, 'max_subspace': 10}
        energy_opt, M_opt, _ = optimize_1site(M, mpo[0], env_left[0], env_right[0], davidson_opts)

        # Variational principle: optimised energy ≤ initial Rayleigh quotient.
        assert energy_opt <= energy_init + 1e-10, (
            f"optimize_1site increased energy: {energy_init} -> {energy_opt}"
        )

    def test_output_is_pure_no_mutation(self, heisenberg_L2):
        """optimize_1site does not mutate the input tensor M."""
        import torch
        mps, mpo = heisenberg_L2
        L = mps.L

        env_left = Environment(L)
        env_right = Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        build_right_envs(mps, mpo, env_right)

        M = mps[0]
        # Snapshot the data blocks before optimisation.
        before = {k: v.clone() for k, v in M.data.items()}

        davidson_opts = {'max_iter': 10, 'tol': 1e-8, 'max_subspace': 5}
        optimize_1site(M, mpo[0], env_left[0], env_right[0], davidson_opts)

        # M must be unchanged.
        for k, v_before in before.items():
            assert torch.allclose(M.data[k], v_before), (
                f"optimize_1site mutated input tensor at block {k}"
            )
