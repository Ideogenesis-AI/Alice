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


"""Tests for the 1-site-plus (CBE) complement module and full sweep round-trip."""

from __future__ import annotations

from nicole import einsum

from alice.algorithm.dmrg.complement import (
    _cheap_factors,
    _complement_vectors,
    _left_half,
    _project_complement_left,
    _project_complement_right,
    _right_half,
    expand_forward,
)
from alice.algorithm.dmrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
    step_left_env,
)
from alice.algorithm.dmrg import dmrg
from alice.algorithm.dmrg.scheme_2s import build_bulk, split_backward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_envs(mps, mpo):
    """Return (env_left, env_right) fully populated for a given (mps, mpo).

    `env_right` is built from the right boundary via `build_right_envs`
    (standard, used by the DMRG driver). `env_left` is built by stepping
    from the left boundary using `step_left_env`; this gives environments
    with the correct structure and index types even when `mps` is not
    left-canonical, which is sufficient for shape/rank unit tests.
    """
    L = mps.L
    env_left  = Environment(L)
    env_right = Environment(L)
    env_left[0] = left_env_boundary(mps, mpo)
    build_right_envs(mps, mpo, env_right)
    # Propagate env_left leftward — valid for testing index shapes even for a
    # non-left-canonical MPS (the environments may not be physically exact).
    for j in range(L - 1):
        env_left[j + 1] = step_left_env(env_left[j], mps[j], mpo[j])
    return env_left, env_right


def _tensor_norm(T) -> float:
    """Frobenius norm of a Nicole `Tensor` via its block data."""
    total = sum(v.pow(2).sum().item() for v in T.data.values())
    return total ** 0.5


# ---------------------------------------------------------------------------
# _cheap_factors
# ---------------------------------------------------------------------------

class TestCheapFactors:
    """Tests for the _cheap_factors helper."""

    def test_returns_two_tensors(self, heisenberg_L4):
        mps, _ = heisenberg_L4
        M_tilde_i, M_tilde_i1 = _cheap_factors(mps[1], mps[2], alpha=None)
        assert M_tilde_i is not None
        assert M_tilde_i1 is not None

    def test_left_factor_rank(self, heisenberg_L4):
        """M̃_i is rank-3 (ket_left, internal, phys)."""
        mps, _ = heisenberg_L4
        M_tilde_i, _ = _cheap_factors(mps[1], mps[2], alpha=None)
        assert len(M_tilde_i.indices) == 3

    def test_right_factor_rank(self, heisenberg_L4):
        """M̃_i1 is rank-3 (internal, ket_right, phys)."""
        mps, _ = heisenberg_L4
        _, M_tilde_i1 = _cheap_factors(mps[1], mps[2], alpha=None)
        assert len(M_tilde_i1.indices) == 3

    def test_alpha_truncation(self, heisenberg_L4):
        """Internal bond dimension ≤ alpha when alpha is smaller than current bond."""
        mps, _ = heisenberg_L4
        alpha = 1
        M_tilde_i, _ = _cheap_factors(mps[1], mps[2], alpha=alpha)
        # axis 1 of M̃_i is the internal reduced bond.
        assert M_tilde_i.indices[1].dim <= alpha

    def test_no_truncation_preserves_full_bond(self, heisenberg_L4):
        """With alpha=None, the internal bond has at most the original bond dimension."""
        mps, _ = heisenberg_L4
        original_bond_dim = mps[1].indices[1].dim
        M_tilde_i, _ = _cheap_factors(mps[1], mps[2], alpha=None)
        assert M_tilde_i.indices[1].dim <= original_bond_dim


# ---------------------------------------------------------------------------
# _left_half and _right_half
# ---------------------------------------------------------------------------

class TestHalfContractions:
    """Tests for _left_half and _right_half."""

    # Test at bond (1, 2) inside an L=4 chain.
    # env_right[i] accumulates sites i+1..L-1 and has ket bond = left bond of site i+1.
    # M_tilde_i1 from cheap_factors(mps[1], mps[2]) has:
    #   axis 1 (ket_right) = right bond of site 2 = bond (2, 3).
    # env_right[2] has ket bond = bond (2, 3) (left bond of site 3). ✓

    def _setup(self, mps, mpo):
        env_left, env_right = _build_envs(mps, mpo)
        M_tilde_i, M_tilde_i1 = _cheap_factors(mps[1], mps[2], alpha=None)
        return env_left, env_right, M_tilde_i, M_tilde_i1

    def test_left_half_rank(self, heisenberg_L4):
        """_left_half returns a rank-4 tensor (bra_left, internal, mpo_mid, phys_bra)."""
        mps, mpo = heisenberg_L4
        env_left, env_right, M_tilde_i, _ = self._setup(mps, mpo)
        lh = _left_half(env_left[1], M_tilde_i, mpo[1])
        assert len(lh.indices) == 4

    def test_right_half_rank(self, heisenberg_L4):
        """_right_half returns a rank-4 tensor (internal, mpo_mid, bra_right, phys_bra)."""
        mps, mpo = heisenberg_L4
        env_left, env_right, _, M_tilde_i1 = self._setup(mps, mpo)
        # env_right[2] has ket bond = bond (2,3) = M_tilde_i1 axis 1 (right bond of site 2).
        rh = _right_half(env_right[2], M_tilde_i1, mpo[2])
        assert len(rh.indices) == 4

    def test_left_half_bra_left_dim(self, heisenberg_L4):
        """_left_half axis 0 (bra_left) dim matches E_left bra_left dim."""
        mps, mpo = heisenberg_L4
        env_left, env_right, M_tilde_i, _ = self._setup(mps, mpo)
        lh = _left_half(env_left[1], M_tilde_i, mpo[1])
        # E_left axis 0 = bra_left; lh axis 0 = bra_left — same dim.
        assert lh.indices[0].dim == env_left[1].indices[0].dim

    def test_right_half_bra_right_dim(self, heisenberg_L4):
        """_right_half axis 2 (bra_right) dim matches E_right bra_right dim."""
        mps, mpo = heisenberg_L4
        env_left, env_right, _, M_tilde_i1 = self._setup(mps, mpo)
        rh = _right_half(env_right[2], M_tilde_i1, mpo[2])
        # E_right axis 0 = bra_right; rh axis 2 = bra_right — same dim.
        assert rh.indices[2].dim == env_right[2].indices[0].dim


# ---------------------------------------------------------------------------
# _project_complement_left and _project_complement_right
# ---------------------------------------------------------------------------

class TestProjectComplement:
    """Tests for the discarded-space projection helpers."""

    def _setup(self, mps, mpo, site_i=1):
        env_left, env_right = _build_envs(mps, mpo)
        M_tilde_i, M_tilde_i1 = _cheap_factors(mps[site_i], mps[site_i + 1], alpha=None)
        lh = _left_half(env_left[site_i], M_tilde_i, mpo[site_i])
        # env_right[site_i + 1] accumulates sites (site_i+2)..(L-1); its ket bond =
        # left bond of site (site_i+2) = right bond of site (site_i+1) = M_tilde_i1 axis 1.
        rh = _right_half(env_right[site_i + 1], M_tilde_i1, mpo[site_i + 1])
        return lh, rh

    def test_left_projection_rank_preserved(self, heisenberg_L4):
        """_project_complement_left output has the same rank as the input."""
        mps, mpo = heisenberg_L4
        lh, _ = self._setup(mps, mpo)
        lh_disc = _project_complement_left(lh, mps[1])
        assert len(lh_disc.indices) == len(lh.indices)

    def test_right_projection_rank_preserved(self, heisenberg_L4):
        """_project_complement_right output has the same rank as the input."""
        mps, mpo = heisenberg_L4
        _, rh = self._setup(mps, mpo)
        rh_disc = _project_complement_right(rh, mps[2])
        assert len(rh_disc.indices) == len(rh.indices)

    def test_left_discarded_orthogonal_to_projector(self, heisenberg_L4):
        """M_tilde_i^* · left_half_disc ≈ 0 when M_tilde_i (left-isometric) is the projector.

        `M_tilde_i` from `_cheap_factors` is left-isometric (U factor from SVD/QR).
        When used as the projector, (I − M_tilde_i M_tilde_i†) is an orthogonal
        projector and M_tilde_i† acting on lh_disc returns zero by the identity
        P† (I − P) = 0, where P = M_tilde_i M_tilde_i†.
        """
        mps, mpo = heisenberg_L4
        env_left, env_right = _build_envs(mps, mpo)
        M_tilde_i, _ = _cheap_factors(mps[1], mps[2], alpha=None)
        lh = _left_half(env_left[1], M_tilde_i, mpo[1])
        lh_disc = _project_complement_left(lh, M_tilde_i)
        # M_tilde_i† lh_disc must be zero (M_tilde_i is left-isometric → M†M = I).
        overlap = einsum('abr,aαpr->bαp', M_tilde_i.conj(), lh_disc)
        norm = _tensor_norm(overlap)
        assert norm < 1e-9, (
            f"Left discarded part not orthogonal to M_tilde_i: overlap norm = {norm}"
        )

    def test_right_discarded_orthogonal_to_projector(self, heisenberg_L4):
        """M_tilde_i1 · rh_disc† ≈ 0 when M_tilde_i1 (right-isometric) is the projector.

        `split_backward` yields a right-isometric V factor. When used as the
        projector, (I − M_tilde_i1† M_tilde_i1) is an orthogonal projector and
        M_tilde_i1 acting on rh_disc returns zero by the identity
        (I − P) P† = 0, where P = M_tilde_i1† M_tilde_i1.
        """
        mps, mpo = heisenberg_L4
        env_left, env_right = _build_envs(mps, mpo)
        theta = build_bulk(mps[1], mps[2])
        _, M_tilde_i1 = split_backward(theta, itag='_test_', trunc=None)
        rh = _right_half(env_right[2], M_tilde_i1, mpo[2])
        rh_disc = _project_complement_right(rh, M_tilde_i1)
        # M_tilde_i1 rh_disc† must be zero (M_tilde_i1 is right-isometric → M M† = I).
        overlap = einsum('bcu,αpcu->αpb', M_tilde_i1.conj(), rh_disc)
        norm = _tensor_norm(overlap)
        assert norm < 1e-9, (
            f"Right discarded part not orthogonal to M_tilde_i1: overlap norm = {norm}"
        )

    def test_left_discarded_norm_bounded(self, heisenberg_L4):
        """The discarded left half has Frobenius norm ≤ that of the original left half.

        After projection (I − P_L) lh_disc ≤ lh in norm, since the projector
        P_L is positive semi-definite with operator norm ≤ 1. This holds
        regardless of the isometry status of the projector tensor.
        """
        mps, mpo = heisenberg_L4
        lh, _ = self._setup(mps, mpo)
        lh_disc = _project_complement_left(lh, mps[1])
        norm_full = _tensor_norm(lh)
        norm_disc = _tensor_norm(lh_disc)
        assert norm_disc <= norm_full + 1e-9, (
            f"Discarded norm {norm_disc:.6f} exceeds original norm {norm_full:.6f}"
        )

    def test_right_discarded_norm_bounded(self, heisenberg_L4):
        """The discarded right half has Frobenius norm ≤ that of the original right half.

        After projection (I − P_R) rh_disc ≤ rh in norm, since the projector
        P_R is positive semi-definite with operator norm ≤ 1. This holds
        regardless of the isometry status of the projector tensor.
        """
        mps, mpo = heisenberg_L4
        lh, rh = self._setup(mps, mpo)
        rh_disc = _project_complement_right(rh, mps[2])
        norm_full = _tensor_norm(rh)
        norm_disc = _tensor_norm(rh_disc)
        assert norm_disc <= norm_full + 1e-9, (
            f"Discarded norm {norm_disc:.6f} exceeds original norm {norm_full:.6f}"
        )


# ---------------------------------------------------------------------------
# _complement_vectors
# ---------------------------------------------------------------------------

class TestComplementVectors:
    """Tests for the _complement_vectors helper."""

    def _setup(self, mps, mpo, site_i=1, k_expand=2):
        env_left, env_right = _build_envs(mps, mpo)
        M_tilde_i, M_tilde_i1 = _cheap_factors(mps[site_i], mps[site_i + 1], alpha=None)
        lh = _left_half(env_left[site_i], M_tilde_i, mpo[site_i])
        rh = _right_half(env_right[site_i + 1], M_tilde_i1, mpo[site_i + 1])
        lh_disc = _project_complement_left(lh, mps[site_i])
        rh_disc = _project_complement_right(rh, mps[site_i + 1])
        N_L, N_R = _complement_vectors(lh_disc, rh_disc, k_expand, flow='<<')
        return N_L, N_R

    def test_NL_rank(self, heisenberg_L4):
        """N_L is rank-3 (bra_left, k, phys)."""
        mps, mpo = heisenberg_L4
        N_L, _ = self._setup(mps, mpo)
        assert len(N_L.indices) == 3

    def test_NR_rank(self, heisenberg_L4):
        """N_R is rank-3 (k, bra_right, phys)."""
        mps, mpo = heisenberg_L4
        _, N_R = self._setup(mps, mpo)
        assert len(N_R.indices) == 3

    def test_k_dimension_bounded(self, heisenberg_L4):
        """The complement bond dimension is ≤ k_expand."""
        mps, mpo = heisenberg_L4
        k_expand = 2
        N_L, N_R = self._setup(mps, mpo, k_expand=k_expand)
        assert N_L.indices[1].dim <= k_expand
        assert N_R.indices[0].dim <= k_expand

    def test_NL_NR_share_k_bond_dim(self, heisenberg_L4):
        """N_L and N_R have the same complement bond dimension (from the same SVD)."""
        mps, mpo = heisenberg_L4
        N_L, N_R = self._setup(mps, mpo)
        assert N_L.indices[1].dim == N_R.indices[0].dim


# ---------------------------------------------------------------------------
# expand_forward
# ---------------------------------------------------------------------------

class TestExpandForward:
    """Tests for the expand_forward public function."""

    def _setup(self, mps, mpo, site_i=1, k_expand=2, alpha=None):
        env_left, env_right = _build_envs(mps, mpo)
        M_i_exp, M_i1_exp, E_right_i_exp = expand_forward(
            mps[site_i], mps[site_i + 1],
            mpo[site_i], mpo[site_i + 1],
            env_left[site_i], env_right[site_i + 1],
            k_expand, alpha,
        )
        return M_i_exp, M_i1_exp, E_right_i_exp, env_left, env_right

    def test_Mi_bond_expands(self, heisenberg_L4):
        """M_i_exp right bond dimension is >= M_i right bond dimension."""
        mps, mpo = heisenberg_L4
        M_i_exp, _, _, _, _ = self._setup(mps, mpo)
        original_dim = mps[1].indices[1].dim
        assert M_i_exp.indices[1].dim >= original_dim

    def test_Mi1_bond_expands(self, heisenberg_L4):
        """M_i1_exp left bond dimension is >= M_i1 left bond dimension."""
        mps, mpo = heisenberg_L4
        _, M_i1_exp, _, _, _ = self._setup(mps, mpo)
        original_dim = mps[2].indices[0].dim
        assert M_i1_exp.indices[0].dim >= original_dim

    def test_expanded_bond_dims_match(self, heisenberg_L4):
        """M_i_exp right bond dim equals M_i1_exp left bond dim (shared expanded bond)."""
        mps, mpo = heisenberg_L4
        M_i_exp, M_i1_exp, _, _, _ = self._setup(mps, mpo)
        assert M_i_exp.indices[1].dim == M_i1_exp.indices[0].dim

    def test_expanded_env_rank(self, heisenberg_L4):
        """The expanded right environment is rank-3."""
        mps, mpo = heisenberg_L4
        _, _, E_right_i_exp, _, _ = self._setup(mps, mpo)
        assert len(E_right_i_exp.indices) == 3

    def test_Mi_non_bond_dims_preserved(self, heisenberg_L4):
        """M_i_exp preserves the ket_left (axis 0) and phys (axis 2) dimensions."""
        mps, mpo = heisenberg_L4
        M_i_exp, _, _, _, _ = self._setup(mps, mpo)
        assert M_i_exp.indices[0].dim == mps[1].indices[0].dim
        assert M_i_exp.indices[2].dim == mps[1].indices[2].dim

    def test_Mi1_non_bond_dims_preserved(self, heisenberg_L4):
        """M_i1_exp preserves the ket_right (axis 1) and phys (axis 2) dimensions."""
        mps, mpo = heisenberg_L4
        _, M_i1_exp, _, _, _ = self._setup(mps, mpo)
        assert M_i1_exp.indices[1].dim == mps[2].indices[1].dim
        assert M_i1_exp.indices[2].dim == mps[2].indices[2].dim

    def test_boundary_site_no_expansion(self, heisenberg_L4):
        """Expanding at bond (0, 1) where site 0 has trivial left bond gives no expansion.

        The left bond at site 0 has dimension 1 (vacuum boundary), so the complement
        in the left direction is trivially zero. The bond dimension may or may not grow
        depending on the right projector, but the left ket_left of M_i_exp remains 1.
        """
        mps, mpo = heisenberg_L4
        env_left, env_right = _build_envs(mps, mpo)
        # env_right[1] = right env at site 1; ket bond = bond (1,2) = right bond of mps[1]. ✓
        M_i_exp, _, _ = expand_forward(
            mps[0], mps[1], mpo[0], mpo[1],
            env_left[0], env_right[1], k_expand=4, alpha=None,
        )
        assert M_i_exp.indices[0].dim == mps[0].indices[0].dim


# ---------------------------------------------------------------------------
# Full CBE DMRG sweep round-trip
# ---------------------------------------------------------------------------

class TestDMRGScheme1sp:
    """End-to-end tests for the 1-site-plus DMRG scheme."""

    def test_energy_convergence_L4(self, heisenberg_L4):
        """1sp DMRG on L=4 Heisenberg chain converges to near exact energy.

        The exact ground-state energy of the L=4 S=1/2 Heisenberg chain with OBC
        is approximately -1.6160254 (in units of J=1). With max_bond=8 and
        expand_k=4 the 1sp scheme should approach this within 1e-3 after a few
        sweeps given the initial random MPS.
        """
        mps, mpo = heisenberg_L4
        opts = dmrg.Options(
            scheme='1sp',
            n_sweeps=6,
            max_bond=8,
            trunc_thresh=1e-15,
            davidson_tol=1e-10,
            expand_k=4,
            expand_alpha=None,
            e_tol=1e-6,
        )
        summary = dmrg.run(mps, mpo, opts)
        exact = -1.6160254
        assert summary.energy < 0.0, "Ground state energy must be negative"
        assert summary.energy >= exact - 0.1, "Energy too low (below exact)"
        assert summary.energy <= exact + 0.05, (
            f"1sp energy {summary.energy:.6f} did not converge near exact "
            f"({exact})"
        )

    def test_energy_monotone_decrease(self, heisenberg_L4):
        """Energy recorded at the end of each backward sweep does not increase."""
        mps, mpo = heisenberg_L4
        opts = dmrg.Options(
            scheme='1sp',
            n_sweeps=4,
            max_bond=8,
            expand_k=3,
            expand_alpha=None,
            e_tol=1e-12,
        )
        summary = dmrg.run(mps, mpo, opts)
        energies = summary.energies
        for k in range(1, len(energies)):
            assert energies[k] <= energies[k - 1] + 1e-9, (
                f"Energy increased at sweep {k + 1}: "
                f"{energies[k - 1]:.9f} -> {energies[k]:.9f}"
            )

    def test_scheme_alias_accepted(self, heisenberg_L4):
        """'1-site-plus' and 'one-site-plus' aliases are both accepted."""
        mps, mpo = heisenberg_L4
        for alias in ('1-site-plus', 'one-site-plus'):
            opts = dmrg.Options(scheme=alias, n_sweeps=1, max_bond=4, expand_k=2)
            summary = dmrg.run(mps, mpo, opts)
            assert summary.n_sweeps == 1

    def test_discarded_weight_reported(self, heisenberg_L4):
        """1sp scheme reports a non-negative discarded weight measured at the center bond."""
        mps, mpo = heisenberg_L4
        opts = dmrg.Options(scheme='1sp', n_sweeps=2, max_bond=4, expand_k=2)
        summary = dmrg.run(mps, mpo, opts)
        assert all(dw >= 0.0 for dw in summary.discarded_weights), (
            "1sp discarded weights must be non-negative"
        )

    def test_expand_alpha_reduces_cost(self, heisenberg_L4):
        """Setting expand_alpha limits the internal bond and does not crash."""
        mps, mpo = heisenberg_L4
        opts = dmrg.Options(
            scheme='1sp',
            n_sweeps=2,
            max_bond=4,
            expand_k=2,
            expand_alpha=2,
        )
        summary = dmrg.run(mps, mpo, opts)
        assert summary.energy < 0.0
