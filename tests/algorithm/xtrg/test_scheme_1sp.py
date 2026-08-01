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


"""Tests for the 1-site-plus (CBE) complement module for XTRG."""

from __future__ import annotations

from nicole import einsum

from alice.network.thermal import NormalMPO, thermal_mpo
from alice.algorithm.xtrg.complement import (
    _absorb_left,
    _absorb_right,
    _compress_bond,
    _new_direction_left,
    _new_direction_right,
    _project_complement_left,
    _project_complement_right,
    _reduce_operands,
    expand_backward,
    expand_forward,
)
from alice.algorithm.xtrg.environ import (
    Environment,
    build_left_envs,
    build_right_envs,
    left_env_boundary,
    right_env_boundary,
    step_left_env,
    step_right_env,
)
from alice.algorithm.xtrg.scheme_2s import left_partial, right_partial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_rho_and_c(mpo, spc, tau_0=2 ** -6, order=4):
    """Return `(rho, mpo_c)` for a self-squaring compression `mpo_c ≈ rho · rho`."""
    rho = thermal_mpo(mpo, tau_0, order, spc)
    mpo_c = NormalMPO.from_mpo(rho)
    mpo_c.canonical(0)
    return rho, mpo_c


def _build_envs_forward_upto(rho, mpo_c, site_i):
    """Return `(env_left, env_right)` valid for `expand_forward` at bond `(site_i, site_i+1)`.

    Mirrors `sweep._forward_1sp`'s state just before processing bond
    `site_i`: `env_right` is built once (requires `mpo_c.center == 0`,
    i.e. right-canonical); `env_left` is then built by progressively
    moving the orthogonality center rightward via `mpo_c.canonical` and
    calling `step_left_env` on the newly left-isometric site, up to
    (but not including) `site_i`. After the call `mpo_c.center == site_i`.
    """
    L = mpo_c.L
    env_left = Environment(L)
    env_right = Environment(L)
    env_left[0] = left_env_boundary(rho, rho, mpo_c)
    build_right_envs(rho, rho, mpo_c, env_right)
    for j in range(site_i):
        mpo_c.canonical(j + 1)
        env_left[j + 1] = step_left_env(env_left[j], rho[j], rho[j], mpo_c[j])
    return env_left, env_right


def _build_envs_backward_upto(rho, mpo_c, site_i):
    """Return `(env_left, env_right)` valid for `expand_backward` at bond `(site_i-1, site_i)`.

    Mirrors `sweep._backward_1sp`'s state just before processing bond
    `site_i`: `mpo_c` is first brought fully left-canonical (as after a
    completed forward sweep) and `env_left` is built in bulk via
    `build_left_envs`. `env_right` is then built by progressively moving
    the orthogonality center leftward via `mpo_c.canonical` and calling
    `step_right_env`, down to (but not including) `site_i`. After the
    call `mpo_c.center == site_i`.
    """
    L = mpo_c.L
    env_left = Environment(L)
    env_right = Environment(L)
    mpo_c.canonical(L - 1)
    build_left_envs(rho, rho, mpo_c, env_left)
    env_right[L - 1] = right_env_boundary(rho, rho, mpo_c)
    for k in range(L - 1, site_i, -1):
        mpo_c.canonical(k - 1)
        env_right[k - 1] = step_right_env(env_right[k], rho[k], rho[k], mpo_c[k])
    return env_left, env_right


def _tensor_norm(T) -> float:
    """Frobenius norm of a Nicole `Tensor` via its block data."""
    total = sum(v.pow(2).sum().item() for v in T.data.values())
    return total ** 0.5


# ---------------------------------------------------------------------------
# _compress_bond / _absorb_right / _absorb_left
# ---------------------------------------------------------------------------

class TestCompressBond:
    """Tests for the _compress_bond helper."""

    def test_isometry_rank(self, spinless_fermion_L4):
        """The isometry from _compress_bond is rank-2."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _build_rho_and_c(mpo, spc)
        iso = _compress_bond(rho[2], axis=0, alpha=2, itag='_test_bond_')
        assert len(iso.indices) == 2

    def test_alpha_truncation(self, spinless_fermion_L4):
        """The reduced bond dimension is at most alpha."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _build_rho_and_c(mpo, spc)
        alpha = 1
        iso = _compress_bond(rho[2], axis=0, alpha=alpha, itag='_test_bond_')
        assert iso.indices[1].dim <= alpha

    def test_connector_axis_dim_preserved(self, spinless_fermion_L4):
        """The isometry's first axis matches the connector bond dimension."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _build_rho_and_c(mpo, spc)
        iso = _compress_bond(rho[2], axis=0, alpha=4, itag='_test_bond_')
        assert iso.indices[0].dim == rho[2].indices[0].dim


class TestAbsorb:
    """Tests for _absorb_right and _absorb_left."""

    def test_absorb_right_rank_preserved(self, spinless_fermion_L4):
        """_absorb_right preserves the tensor's rank."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _build_rho_and_c(mpo, spc)
        iso = _compress_bond(rho[2], axis=0, alpha=2, itag='_test_bond_')
        A_near_reduced = _absorb_right(rho[1], iso, axis=1)
        assert len(A_near_reduced.indices) == len(rho[1].indices)

    def test_absorb_left_rank_preserved(self, spinless_fermion_L4):
        """_absorb_left preserves the tensor's rank."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _build_rho_and_c(mpo, spc)
        iso = _compress_bond(rho[2], axis=0, alpha=2, itag='_test_bond_')
        A_far_reduced = _absorb_left(iso, rho[2], axis=0)
        assert len(A_far_reduced.indices) == len(rho[2].indices)

    def test_absorb_right_reduced_bond_dim(self, spinless_fermion_L4):
        """_absorb_right's connector axis dimension equals alpha (or less)."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _build_rho_and_c(mpo, spc)
        alpha = 2
        iso = _compress_bond(rho[2], axis=0, alpha=alpha, itag='_test_bond_')
        A_near_reduced = _absorb_right(rho[1], iso, axis=1)
        assert A_near_reduced.indices[1].dim <= alpha

    def test_absorb_left_reduced_bond_dim(self, spinless_fermion_L4):
        """_absorb_left's connector axis dimension equals alpha (or less)."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _build_rho_and_c(mpo, spc)
        alpha = 2
        iso = _compress_bond(rho[2], axis=0, alpha=alpha, itag='_test_bond_')
        A_far_reduced = _absorb_left(iso, rho[2], axis=0)
        assert A_far_reduced.indices[0].dim <= alpha

    def test_absorb_preserves_other_dims(self, spinless_fermion_L4):
        """_absorb_right/_absorb_left leave non-connector axes unchanged."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _build_rho_and_c(mpo, spc)
        iso = _compress_bond(rho[2], axis=0, alpha=2, itag='_test_bond_')
        A_near_reduced = _absorb_right(rho[1], iso, axis=1)
        A_far_reduced = _absorb_left(iso, rho[2], axis=0)
        # A_near_reduced: axes (a, alpha, r, x) — axes 0,2,3 unchanged.
        assert A_near_reduced.indices[0].dim == rho[1].indices[0].dim
        assert A_near_reduced.indices[2].dim == rho[1].indices[2].dim
        assert A_near_reduced.indices[3].dim == rho[1].indices[3].dim
        # A_far_reduced: axes (alpha, e, u, y) — axes 1,2,3 unchanged.
        assert A_far_reduced.indices[1].dim == rho[2].indices[1].dim
        assert A_far_reduced.indices[2].dim == rho[2].indices[2].dim
        assert A_far_reduced.indices[3].dim == rho[2].indices[3].dim


class TestReduceOperands:
    """Tests for the _reduce_operands helper."""

    def test_alpha_none_returns_unchanged(self, spinless_fermion_L4):
        """alpha=None returns the operands unchanged (no compression)."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _build_rho_and_c(mpo, spc)
        out = _reduce_operands(rho[1], rho[2], rho[1], rho[2], alpha=None)
        assert out == (rho[1], rho[2], rho[1], rho[2])

    def test_returns_four_tensors(self, spinless_fermion_L4):
        """_reduce_operands returns four tensors when alpha is set."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _build_rho_and_c(mpo, spc)
        out = _reduce_operands(rho[1], rho[2], rho[1], rho[2], alpha=2)
        assert len(out) == 4
        for t in out:
            assert t is not None

    def test_reduced_bonds_match(self, spinless_fermion_L4):
        """Near and far reduced tensors share the same connector bond dimension."""
        mpo, spc, _ = spinless_fermion_L4
        rho, _ = _build_rho_and_c(mpo, spc)
        A_near_r, A_far_r, B_near_r, B_far_r = _reduce_operands(
            rho[1], rho[2], rho[1], rho[2], alpha=2,
        )
        assert A_near_r.indices[1].dim == A_far_r.indices[0].dim
        assert B_near_r.indices[1].dim == B_far_r.indices[0].dim


# ---------------------------------------------------------------------------
# left_partial / right_partial (reused from scheme_2s) with reduced operands
# ---------------------------------------------------------------------------

class TestHalfContractions:
    """Tests for left_partial/right_partial applied to reduced operands."""

    def _setup(self, mpo, spc, site_i=1):
        rho, mpo_c = _build_rho_and_c(mpo, spc)
        env_left, env_right = _build_envs_forward_upto(rho, mpo_c, site_i)
        A_i_r, A_j_r, B_i_r, B_j_r = _reduce_operands(
            rho[site_i], rho[site_i + 1], rho[site_i], rho[site_i + 1], alpha=2,
        )
        return env_left, env_right, A_i_r, A_j_r, B_i_r, B_j_r

    def test_left_half_rank(self, spinless_fermion_L4):
        """left_partial returns a rank-5 tensor (c, p, q, r, s)."""
        mpo, spc, _ = spinless_fermion_L4
        env_left, _, A_i_r, _, B_i_r, _ = self._setup(mpo, spc)
        lh = left_partial(env_left[1], A_i_r, B_i_r)
        assert len(lh.indices) == 5

    def test_right_half_rank(self, spinless_fermion_L4):
        """right_partial returns a rank-5 tensor (d, p, q, u, v)."""
        mpo, spc, _ = spinless_fermion_L4
        _, env_right, _, A_j_r, _, B_j_r = self._setup(mpo, spc)
        rh = right_partial(env_right[2], A_j_r, B_j_r)
        assert len(rh.indices) == 5


# ---------------------------------------------------------------------------
# _project_complement_left and _project_complement_right
# ---------------------------------------------------------------------------

class TestProjectComplement:
    """Tests for the discarded-space projection helpers."""

    def _setup(self, mpo, spc, site_i=1, alpha=2):
        rho, mpo_c = _build_rho_and_c(mpo, spc)
        env_left, env_right = _build_envs_forward_upto(rho, mpo_c, site_i)
        A_i_r, A_j_r, B_i_r, B_j_r = _reduce_operands(
            rho[site_i], rho[site_i + 1], rho[site_i], rho[site_i + 1], alpha,
        )
        lh = left_partial(env_left[site_i], A_i_r, B_i_r)
        rh = right_partial(env_right[site_i + 1], A_j_r, B_j_r)
        return lh, rh, mpo_c

    def test_left_projection_rank_preserved(self, spinless_fermion_L4):
        """_project_complement_left output has the same rank as the input."""
        mpo, spc, _ = spinless_fermion_L4
        lh, _, mpo_c = self._setup(mpo, spc)
        lh_disc = _project_complement_left(lh, mpo_c[1])
        assert len(lh_disc.indices) == len(lh.indices)

    def test_right_projection_rank_preserved(self, spinless_fermion_L4):
        """_project_complement_right output has the same rank as the input."""
        mpo, spc, _ = spinless_fermion_L4
        _, rh, mpo_c = self._setup(mpo, spc)
        rh_disc = _project_complement_right(rh, mpo_c[2])
        assert len(rh_disc.indices) == len(rh.indices)

    def test_left_discarded_orthogonal_to_isometric_projector(self, spinless_fermion_L4):
        """C_i^conj · left_half_disc ≈ 0 when C_i (left-isometric) is the projector.

        Builds envs up to bond `(0, 1)`, which canonicalizes `mpo_c[0]` into
        an exactly left-isometric tensor (via the `mpo_c.canonical(1)` step
        inside `_build_envs_forward_upto`), matching the projector's
        assumption.
        """
        mpo, spc, _ = spinless_fermion_L4
        rho, mpo_c = _build_rho_and_c(mpo, spc)
        env_left, _ = _build_envs_forward_upto(rho, mpo_c, site_i=1)
        A_i_r, A_j_r, B_i_r, B_j_r = _reduce_operands(
            rho[0], rho[1], rho[0], rho[1], alpha=None,
        )
        lh = left_partial(env_left[0], A_i_r, B_i_r)
        lh_disc = _project_complement_left(lh, mpo_c[0])
        # mpo_c[0] is now left-isometric (moved there by _build_envs_forward_upto).
        overlap = einsum('cdrs,cpqrs->dpq', mpo_c[0].conj(), lh_disc)
        norm = _tensor_norm(overlap)
        assert norm < 1e-8, f"Left discarded part not orthogonal: overlap norm = {norm}"

    def test_left_discarded_norm_bounded(self, spinless_fermion_L4):
        """The discarded left half has Frobenius norm <= that of the original."""
        mpo, spc, _ = spinless_fermion_L4
        lh, _, mpo_c = self._setup(mpo, spc)
        lh_disc = _project_complement_left(lh, mpo_c[1])
        norm_full = _tensor_norm(lh)
        norm_disc = _tensor_norm(lh_disc)
        assert norm_disc <= norm_full + 1e-9, (
            f"Discarded norm {norm_disc:.6f} exceeds original norm {norm_full:.6f}"
        )

    def test_right_discarded_norm_bounded(self, spinless_fermion_L4):
        """The discarded right half has Frobenius norm <= that of the original."""
        mpo, spc, _ = spinless_fermion_L4
        _, rh, mpo_c = self._setup(mpo, spc)
        rh_disc = _project_complement_right(rh, mpo_c[2])
        norm_full = _tensor_norm(rh)
        norm_disc = _tensor_norm(rh_disc)
        assert norm_disc <= norm_full + 1e-9, (
            f"Discarded norm {norm_disc:.6f} exceeds original norm {norm_full:.6f}"
        )


# ---------------------------------------------------------------------------
# _new_direction_right / _new_direction_left
# ---------------------------------------------------------------------------

class TestNewDirection:
    """Tests for the complement-direction extraction helpers."""

    def _setup(self, mpo, spc, site_i=1, alpha=2):
        rho, mpo_c = _build_rho_and_c(mpo, spc)
        env_left, env_right = _build_envs_forward_upto(rho, mpo_c, site_i)
        A_i_r, A_j_r, B_i_r, B_j_r = _reduce_operands(
            rho[site_i], rho[site_i + 1], rho[site_i], rho[site_i + 1], alpha,
        )
        lh = left_partial(env_left[site_i], A_i_r, B_i_r)
        rh = right_partial(env_right[site_i + 1], A_j_r, B_j_r)
        lh_disc = _project_complement_left(lh, mpo_c[site_i])
        rh_disc = _project_complement_right(rh, mpo_c[site_i + 1])
        return lh_disc, rh_disc

    def test_NR_rank(self, spinless_fermion_L4):
        """N_R is rank-4 (k, d, u, v)."""
        mpo, spc, _ = spinless_fermion_L4
        lh_disc, rh_disc = self._setup(mpo, spc)
        N_R = _new_direction_right(lh_disc, rh_disc, k_expand=2)
        assert len(N_R.indices) == 4

    def test_NL_rank(self, spinless_fermion_L4):
        """N_L is rank-4 (c, k, r, s)."""
        mpo, spc, _ = spinless_fermion_L4
        lh_disc, rh_disc = self._setup(mpo, spc)
        N_L = _new_direction_left(lh_disc, rh_disc, k_expand=2)
        assert len(N_L.indices) == 4

    def test_k_dimension_bounded(self, spinless_fermion_L4):
        """The complement bond dimension is <= k_expand."""
        mpo, spc, _ = spinless_fermion_L4
        lh_disc, rh_disc = self._setup(mpo, spc)
        k_expand = 2
        N_R = _new_direction_right(lh_disc, rh_disc, k_expand)
        N_L = _new_direction_left(lh_disc, rh_disc, k_expand)
        assert N_R.indices[0].dim <= k_expand
        assert N_L.indices[1].dim <= k_expand


# ---------------------------------------------------------------------------
# expand_forward / expand_backward
# ---------------------------------------------------------------------------

class TestExpandForward:
    """Tests for the expand_forward public function."""

    def _setup(self, mpo, spc, site_i=1, k_expand=2, alpha=2):
        rho, mpo_c = _build_rho_and_c(mpo, spc)
        env_left, env_right = _build_envs_forward_upto(rho, mpo_c, site_i)
        C_i_exp, C_j_exp, E_right_i_exp = expand_forward(
            rho[site_i], rho[site_i], mpo_c[site_i],
            rho[site_i + 1], rho[site_i + 1], mpo_c[site_i + 1],
            env_left[site_i], env_right[site_i + 1],
            k_expand, alpha,
        )
        return C_i_exp, C_j_exp, E_right_i_exp, rho, mpo_c

    def test_Ci_rank(self, spinless_fermion_L4):
        """C_i_exp is rank-4."""
        mpo, spc, _ = spinless_fermion_L4
        C_i_exp, _, _, _, _ = self._setup(mpo, spc)
        assert len(C_i_exp.indices) == 4

    def test_Ci_bond_expands(self, spinless_fermion_L4):
        """C_i_exp right bond dimension is >= C_i's original right bond dimension."""
        mpo, spc, _ = spinless_fermion_L4
        C_i_exp, _, _, _, mpo_c = self._setup(mpo, spc)
        original_dim = mpo_c[1].indices[1].dim
        assert C_i_exp.indices[1].dim >= original_dim

    def test_Cj_bond_expands(self, spinless_fermion_L4):
        """C_j_exp left bond dimension is >= C_j's original left bond dimension."""
        mpo, spc, _ = spinless_fermion_L4
        _, C_j_exp, _, _, mpo_c = self._setup(mpo, spc)
        original_dim = mpo_c[2].indices[0].dim
        assert C_j_exp.indices[0].dim >= original_dim

    def test_expanded_bond_dims_match(self, spinless_fermion_L4):
        """C_i_exp right bond dim equals C_j_exp left bond dim (shared expanded bond)."""
        mpo, spc, _ = spinless_fermion_L4
        C_i_exp, C_j_exp, _, _, _ = self._setup(mpo, spc)
        assert C_i_exp.indices[1].dim == C_j_exp.indices[0].dim

    def test_expanded_env_rank(self, spinless_fermion_L4):
        """The expanded right environment is rank-3."""
        mpo, spc, _ = spinless_fermion_L4
        _, _, E_right_i_exp, _, _ = self._setup(mpo, spc)
        assert len(E_right_i_exp.indices) == 3

    def test_expanded_env_bond_matches(self, spinless_fermion_L4):
        """The expanded right environment's C-bond matches C_i_exp's right bond."""
        mpo, spc, _ = spinless_fermion_L4
        C_i_exp, _, E_right_i_exp, _, _ = self._setup(mpo, spc)
        assert E_right_i_exp.indices[0].dim == C_i_exp.indices[1].dim

    def test_Ci_non_bond_dims_preserved(self, spinless_fermion_L4):
        """C_i_exp preserves the left bond (axis 0) and phys dimensions."""
        mpo, spc, _ = spinless_fermion_L4
        C_i_exp, _, _, _, mpo_c = self._setup(mpo, spc)
        assert C_i_exp.indices[0].dim == mpo_c[1].indices[0].dim
        assert C_i_exp.indices[2].dim == mpo_c[1].indices[2].dim
        assert C_i_exp.indices[3].dim == mpo_c[1].indices[3].dim

    def test_alpha_none_still_expands(self, spinless_fermion_L4):
        """expand_forward works with alpha=None (no cheap compression)."""
        mpo, spc, _ = spinless_fermion_L4
        C_i_exp, C_j_exp, E_right_i_exp, _, mpo_c = self._setup(mpo, spc, alpha=None)
        assert C_i_exp.indices[1].dim >= mpo_c[1].indices[1].dim


class TestExpandBackward:
    """Tests for the expand_backward public function."""

    def _setup(self, mpo, spc, site_i=2, k_expand=2, alpha=2):
        rho, mpo_c = _build_rho_and_c(mpo, spc)
        env_left, env_right = _build_envs_backward_upto(rho, mpo_c, site_i)
        C_i_exp, C_im1_exp, E_left_i_exp = expand_backward(
            rho[site_i], rho[site_i], mpo_c[site_i],
            rho[site_i - 1], rho[site_i - 1], mpo_c[site_i - 1],
            env_left[site_i - 1], env_right[site_i],
            k_expand, alpha,
        )
        return C_i_exp, C_im1_exp, E_left_i_exp, rho, mpo_c

    def test_Ci_rank(self, spinless_fermion_L4):
        """C_i_exp is rank-4."""
        mpo, spc, _ = spinless_fermion_L4
        C_i_exp, _, _, _, _ = self._setup(mpo, spc)
        assert len(C_i_exp.indices) == 4

    def test_Ci_bond_expands(self, spinless_fermion_L4):
        """C_i_exp left bond dimension is >= C_i's original left bond dimension."""
        mpo, spc, _ = spinless_fermion_L4
        C_i_exp, _, _, _, mpo_c = self._setup(mpo, spc)
        original_dim = mpo_c[2].indices[0].dim
        assert C_i_exp.indices[0].dim >= original_dim

    def test_Cim1_bond_expands(self, spinless_fermion_L4):
        """C_im1_exp right bond dimension is >= C_im1's original right bond dimension."""
        mpo, spc, _ = spinless_fermion_L4
        _, C_im1_exp, _, _, mpo_c = self._setup(mpo, spc)
        original_dim = mpo_c[1].indices[1].dim
        assert C_im1_exp.indices[1].dim >= original_dim

    def test_expanded_bond_dims_match(self, spinless_fermion_L4):
        """C_im1_exp right bond dim equals C_i_exp left bond dim (shared expanded bond)."""
        mpo, spc, _ = spinless_fermion_L4
        C_i_exp, C_im1_exp, _, _, _ = self._setup(mpo, spc)
        assert C_im1_exp.indices[1].dim == C_i_exp.indices[0].dim

    def test_expanded_env_rank(self, spinless_fermion_L4):
        """The expanded left environment is rank-3."""
        mpo, spc, _ = spinless_fermion_L4
        _, _, E_left_i_exp, _, _ = self._setup(mpo, spc)
        assert len(E_left_i_exp.indices) == 3

    def test_alpha_none_still_expands(self, spinless_fermion_L4):
        """expand_backward works with alpha=None (no cheap compression)."""
        mpo, spc, _ = spinless_fermion_L4
        C_i_exp, C_im1_exp, _, _, mpo_c = self._setup(mpo, spc, alpha=None)
        assert C_i_exp.indices[0].dim >= mpo_c[2].indices[0].dim
