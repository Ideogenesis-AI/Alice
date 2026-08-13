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


"""Tests for alice.algorithm.xtrg.scheme_1s."""

from __future__ import annotations

from nicole import allclose as tensors_allclose

from alice.network.thermal import thermal_mpo
from alice.algorithm.xtrg.environ import (
    Environment,
    build_right_envs,
    left_env_boundary,
    right_env_boundary,
    step_left_env,
    step_right_env,
)
from alice.algorithm.xtrg.scheme_1s import local_update_1s
from alice.algorithm.xtrg.xtrg import Options, _fit_mpo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_envs(rho, rho_a=None, rho_b=None):
    """Build initial left and right environments for a compression test.

    Parameters
    ----------
    rho:
        NormalMPO used as the compressed MPO C (must be right-canonical).
    rho_a:
        Factor MPO A. Defaults to `rho`.
    rho_b:
        Factor MPO B. Defaults to `rho`.

    Returns
    -------
    tuple
        `(env_left, env_right)` with all blocks populated.
    """
    if rho_a is None:
        rho_a = rho
    if rho_b is None:
        rho_b = rho

    L = rho.L
    env_left = Environment(L)
    env_right = Environment(L)
    env_left[0] = left_env_boundary(rho_a, rho_b, rho)
    build_right_envs(rho_a, rho_b, rho, env_right)
    return env_left, env_right


def _mixed_canonical_envs(mpo_c, mpo_a, mpo_b, center):
    """Build the environments surrounding `center` in `mpo_c`'s canonical frame.

    `mpo_c` must already be canonicalized at `center`, so that sites left of
    it are left-isometric and sites right of it are right-isometric. Both
    environments are then accumulated from their respective boundaries, which
    is what makes the 1-site update at `center` a true variational optimum.

    Parameters
    ----------
    mpo_c:
        Compressed MPO C, canonicalized at `center`.
    mpo_a:
        Factor MPO A.
    mpo_b:
        Factor MPO B.
    center:
        Site index of the orthogonality center.

    Returns
    -------
    tuple
        `(E_left, E_right)` at site `center`.
    """
    L = mpo_c.L
    E_left = left_env_boundary(mpo_a, mpo_b, mpo_c)
    for j in range(center):
        E_left = step_left_env(E_left, mpo_a[j], mpo_b[j], mpo_c[j])
    E_right = right_env_boundary(mpo_a, mpo_b, mpo_c)
    for j in range(L - 1, center, -1):
        E_right = step_right_env(E_right, mpo_a[j], mpo_b[j], mpo_c[j])
    return E_left, E_right


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLocalUpdate1s:
    """Tests for local_update_1s."""

    def test_output_rank(self, spinless_fermion_L4):
        """local_update_1s returns a rank-4 tensor."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)

        env_left, env_right = _setup_envs(rho)
        C0 = local_update_1s(env_left[0], rho[0], rho[0], env_right[0])
        assert len(C0.indices) == 4

    def test_output_matches_mpo_convention(self, spinless_fermion_L4):
        """Output C_i has the same index count as rho[i] (left, right, phys_in, phys_out)."""
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)
        rho.canonical(0)

        L = rho.L
        env_left, env_right = _setup_envs(rho)
        for i in range(L):
            C_i = local_update_1s(env_left[i], rho[i], rho[i], env_right[i])
            assert len(C_i.indices) == len(rho[i].indices) == 4
            # Build left env for next site.
            if i < L - 1:
                env_left[i + 1] = step_left_env(env_left[i], rho[i], rho[i], rho[i])

    def test_idempotency_when_converged(self, spinless_fermion_L4):
        """Re-applying local_update_1s to a converged fit reproduces its tensors.

        A converged variational fit is a fixed point of the 1-site update, so
        updating any site of `rho_sq` against its own environments must return
        that site's tensor back. The comparison is up to an overall positive
        factor because `_fit_mpo` finishes with `compact()`, which pulls the
        Frobenius norm out of the site tensors and into `log_scale`.
        """
        mpo, spc, _ = spinless_fermion_L4
        rho = thermal_mpo(mpo, 2 ** -12, 4, spc)

        opts = Options(scheme='1s', n_sweeps=4, max_bond=None)
        rho_sq, _ = _fit_mpo(rho, rho, opts)

        for i in range(rho_sq.L):
            rho_sq.canonical(i)
            E_left, E_right = _mixed_canonical_envs(rho_sq, rho, rho, i)
            C_new = local_update_1s(E_left, rho[i], rho[i], E_right)
            C_old = rho_sq[i]

            assert len(C_new.indices) == 4
            assert tensors_allclose(
                C_new / C_new.norm(), C_old / C_old.norm(), atol=1e-12
            ), f"1-site update is not idempotent at site {i}"
