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


"""Tests for alice.algorithm.tdvp.scheme_1s.

Beyond the usual algebraic checks, this module pins down the two properties
that make the local updates a tangent-space projection rather than an
arbitrary local operation:

- `matvec_1s` and `matvec_0s` reproduce `⟨δψ|H|ψ⟩` for a variation `δ` of the
  tensor they act on, so they really are `H` compressed onto the tangent
  space rather than a local approximation to it.
- the splits are exact, which is what keeps the surrounding tensors isometric
  and hence keeps that compression an *orthogonal* projection.
"""

from __future__ import annotations

import copy

import torch
from nicole import einsum

from alice.algorithm.tdvp.environ import (
    Environment,
    left_env_boundary,
    right_env_boundary,
    step_left_env,
    step_right_env,
)
from alice.algorithm.tdvp.krylov import inner_product
from alice.algorithm.tdvp.scheme_1s import (
    absorb_backward,
    absorb_forward,
    evolve_1site,
    evolve_bond,
    matvec_0s,
    matvec_1s,
    split_backward,
    split_forward,
)
from alice.network import observe

from .conftest import fresh_envs

KRYLOV = {'max_krylov': 20, 'tol': 1e-14}


def _randomize_like(t, seed):
    """Return a tensor shaped like `t` with fresh random complex blocks."""
    out = t.clone()
    gen = torch.Generator().manual_seed(seed)
    for key in out.data:
        shape = out.data[key].shape
        real = torch.randn(shape, generator=gen, dtype=torch.float64)
        imag = torch.randn(shape, generator=gen, dtype=torch.float64)
        out.data[key] = torch.complex(real, imag)
    return out


def _sandwich(bra, ket, mpo):
    """Compute `⟨bra|H|ket⟩` by an explicit full-network contraction."""
    E = left_env_boundary(ket, mpo)
    for i in range(ket.L):
        E = einsum('aob,acr,oprs,bds->cpd', E, bra[i].conj(), mpo[i], ket[i])
    key, val = next(iter(E.data.items()))
    weight = 1.0 if E.intw is None else complex(E.intw[key].weights[0, 0].item())
    return complex(val.item()) * weight


# ---------------------------------------------------------------------------
# 1-site effective Hamiltonian
# ---------------------------------------------------------------------------

class TestMatvec1s:
    """Tests for the 1-site effective Hamiltonian."""

    def test_energy_matches_observe(self, heisenberg_L2):
        """⟨M|H_eff|M⟩ equals the full-network energy."""
        mps, mpo = heisenberg_L2
        env_left, env_right = fresh_envs(mps, mpo)
        HM = matvec_1s(mps[0], mpo[0], env_left[0], env_right[0])
        assert abs(inner_product(mps[0], HM).real - observe(mps, mpo)) < 1e-10

    def test_hermitian(self, heisenberg_L2):
        """⟨a|H_eff|b⟩ = conj(⟨b|H_eff|a⟩)."""
        mps, mpo = heisenberg_L2
        env_left, env_right = fresh_envs(mps, mpo)
        a, b = mps[0], _randomize_like(mps[0], 11)
        lhs = inner_product(a, matvec_1s(b, mpo[0], env_left[0], env_right[0]))
        rhs = inner_product(b, matvec_1s(a, mpo[0], env_left[0], env_right[0]))
        assert abs(lhs - rhs.conjugate()) < 1e-12

    def test_preserves_index_structure(self, heisenberg_L2):
        """The output can be added to the input, as Lanczos requires."""
        mps, mpo = heisenberg_L2
        env_left, env_right = fresh_envs(mps, mpo)
        HM = matvec_1s(mps[0], mpo[0], env_left[0], env_right[0])
        assert HM.itags == mps[0].itags
        assert [i.direction for i in HM.indices] == [i.direction for i in mps[0].indices]

    def test_is_the_tangent_space_component(self, heisenberg_L6):
        """⟨δM|H_eff|M⟩ equals ⟨δψ|H|ψ⟩ for the state varied at that site.

        This is the statement that `matvec_1s` computes the one-site block of
        `P_{T_ψ} H |ψ⟩` and not merely a local operator.
        """
        mps, mpo = heisenberg_L6
        site = 2
        mps.canonical(site)

        env_left, env_right = Environment(mps.L), Environment(mps.L)
        env_left[0] = left_env_boundary(mps, mpo)
        env_right[mps.L - 1] = right_env_boundary(mps, mpo)
        for i in range(site):
            env_left[i + 1] = step_left_env(env_left[i], mps[i], mpo[i])
        for i in range(mps.L - 1, site, -1):
            env_right[i - 1] = step_right_env(env_right[i], mps[i], mpo[i])

        dM = _randomize_like(mps[site], 17)
        bra = copy.deepcopy(mps)
        bra[site] = dM

        lhs = _sandwich(bra, mps, mpo)
        rhs = inner_product(dM, matvec_1s(mps[site], mpo[site],
                                          env_left[site], env_right[site]))
        assert abs(lhs - rhs) < 1e-12


# ---------------------------------------------------------------------------
# Zero-site effective Hamiltonian
# ---------------------------------------------------------------------------

class TestMatvec0s:
    """Tests for the zero-site (bond) effective Hamiltonian."""

    def test_energy_matches_observe(self, heisenberg_L6):
        """⟨C|H_0|C⟩ equals the full-network energy after a forward split."""
        mps, mpo = heisenberg_L6
        env_left, env_right = fresh_envs(mps, mpo)

        A, C = split_forward(mps[0], mps._bond_itag(1))
        env_left[1] = step_left_env(env_left[0], A, mpo[0])
        H0C = matvec_0s(C, env_left[1], env_right[0])
        assert abs(inner_product(C, H0C).real - observe(mps, mpo)) < 1e-10

    def test_energy_matches_observe_backward(self, heisenberg_L6):
        """The same holds for the bond exposed by a backward split."""
        mps, mpo = heisenberg_L6
        L = mps.L
        mps.canonical(L - 1)

        env_left, env_right = Environment(L), Environment(L)
        env_left[0] = left_env_boundary(mps, mpo)
        env_right[L - 1] = right_env_boundary(mps, mpo)
        for i in range(L - 1):
            env_left[i + 1] = step_left_env(env_left[i], mps[i], mpo[i])

        C, B = split_backward(mps[L - 1], mps._bond_itag(L - 1))
        env_right[L - 2] = step_right_env(env_right[L - 1], B, mpo[L - 1])
        H0C = matvec_0s(C, env_left[L - 1], env_right[L - 2])
        assert abs(inner_product(C, H0C).real - observe(mps, mpo)) < 1e-10

    def test_hermitian(self, heisenberg_L6):
        """H_0 is Hermitian, as the Lanczos propagator assumes."""
        mps, mpo = heisenberg_L6
        env_left, env_right = fresh_envs(mps, mpo)
        A, C = split_forward(mps[0], mps._bond_itag(1))
        env_left[1] = step_left_env(env_left[0], A, mpo[0])

        other = _randomize_like(C, 23)
        lhs = inner_product(C, matvec_0s(other, env_left[1], env_right[0]))
        rhs = inner_product(other, matvec_0s(C, env_left[1], env_right[0]))
        assert abs(lhs - rhs.conjugate()) < 1e-12

    def test_preserves_index_structure(self, heisenberg_L6):
        """The output bond tensor is interchangeable with the input."""
        mps, mpo = heisenberg_L6
        env_left, env_right = fresh_envs(mps, mpo)
        A, C = split_forward(mps[0], mps._bond_itag(1))
        env_left[1] = step_left_env(env_left[0], A, mpo[0])
        out = matvec_0s(C, env_left[1], env_right[0])
        assert out.itags == C.itags
        assert [i.direction for i in out.indices] == [i.direction for i in C.indices]


# ---------------------------------------------------------------------------
# Splits and absorption
# ---------------------------------------------------------------------------

class TestSplits:
    """Tests for the QR splits and the absorb helpers."""

    def test_forward_roundtrip_is_exact(self, heisenberg_L6):
        """A · C absorbed into the next site reproduces the original pair."""
        mps, mpo = heisenberg_L6
        before = einsum('abr,bcs->acrs', mps[0], mps[1])

        A, C = split_forward(mps[0], mps._bond_itag(1))
        after = einsum('abr,bcs->acrs', A, absorb_forward(C, mps[1]))

        for key in before.data:
            assert torch.allclose(after.data[key], before.data[key], atol=1e-13)

    def test_backward_roundtrip_is_exact(self, heisenberg_L6):
        """The mirror image holds for the backward split."""
        mps, mpo = heisenberg_L6
        L = mps.L
        mps.canonical(L - 1)
        before = einsum('abr,bcs->acrs', mps[L - 2], mps[L - 1])

        C, B = split_backward(mps[L - 1], mps._bond_itag(L - 1))
        after = einsum('abr,bcs->acrs', absorb_backward(mps[L - 2], C), B)

        for key in before.data:
            assert torch.allclose(after.data[key], before.data[key], atol=1e-13)

    def test_forward_split_never_grows_the_bond(self, heisenberg_L6):
        """The split reports the attainable rank, which never exceeds the bond.

        Near a chain edge a nominal bond can exceed `dim_left · d`, so the
        exact QR trims it. That costs nothing — those states were null — but
        it means the 1-site scheme pins the rank only after the first trim,
        not from the initial tensor shapes.
        """
        mps, _ = heisenberg_L6
        before = mps[0].indices[1].dim
        A, _ = split_forward(mps[0], mps._bond_itag(1))
        assert A.indices[1].dim <= before

    def test_forward_split_is_a_fixed_point(self, heisenberg_L6):
        """Re-splitting an already-split tensor leaves the bond alone.

        This is the fixed-rank property the 1-site scheme relies on: once the
        bond sits at the attainable rank, no later step moves it.
        """
        mps, _ = heisenberg_L6
        A, C = split_forward(mps[0], mps._bond_itag(1))
        A2, _ = split_forward(absorb_backward(A, C), mps._bond_itag(1))
        assert A2.indices[1].dim == A.indices[1].dim

    def test_forward_split_is_left_isometric(self, heisenberg_L6):
        """A†A = 1, which is what makes the environments projectors."""
        mps, _ = heisenberg_L6
        A, _ = split_forward(mps[0], mps._bond_itag(1))
        gram = einsum('abr,acr->bc', A.conj(), A)
        for key, block in gram.data.items():
            assert torch.allclose(
                block, torch.eye(block.shape[0], dtype=block.dtype), atol=1e-12
            )

    def test_backward_split_is_right_isometric(self, heisenberg_L6):
        """BB† = 1 for the backward split."""
        mps, _ = heisenberg_L6
        mps.canonical(mps.L - 1)
        _, B = split_backward(mps[mps.L - 1], mps._bond_itag(mps.L - 1))
        gram = einsum('abr,cbr->ac', B.conj(), B)
        for key, block in gram.data.items():
            assert torch.allclose(
                block, torch.eye(block.shape[0], dtype=block.dtype), atol=1e-12
            )


# ---------------------------------------------------------------------------
# Local evolutions
# ---------------------------------------------------------------------------

class TestEvolve:
    """Tests for `evolve_1site` and `evolve_bond`."""

    def test_evolve_1site_is_unitary(self, heisenberg_L2):
        mps, mpo = heisenberg_L2
        env_left, env_right = fresh_envs(mps, mpo)
        out, _ = evolve_1site(mps[0], mpo[0], env_left[0], env_right[0], 0.3, KRYLOV)
        assert abs(out.norm() - mps[0].norm()) < 1e-12

    def test_evolve_1site_conserves_the_local_energy(self, heisenberg_L2):
        """A unitary generated by H_eff commutes with H_eff."""
        mps, mpo = heisenberg_L2
        env_left, env_right = fresh_envs(mps, mpo)
        M = mps[0]
        before = inner_product(M, matvec_1s(M, mpo[0], env_left[0], env_right[0])).real
        out, _ = evolve_1site(M, mpo[0], env_left[0], env_right[0], 0.3, KRYLOV)
        after = inner_product(out, matvec_1s(out, mpo[0], env_left[0], env_right[0])).real
        assert abs(after - before) < 1e-11

    def test_evolve_1site_is_reversible(self, heisenberg_L2):
        mps, mpo = heisenberg_L2
        env_left, env_right = fresh_envs(mps, mpo)
        fwd, _ = evolve_1site(mps[0], mpo[0], env_left[0], env_right[0], 0.3, KRYLOV)
        back, _ = evolve_1site(fwd, mpo[0], env_left[0], env_right[0], -0.3, KRYLOV)
        for key in mps[0].data:
            assert torch.allclose(back.data[key], mps[0].data[key], atol=1e-11)

    def test_evolve_bond_is_unitary(self, heisenberg_L6):
        mps, mpo = heisenberg_L6
        env_left, env_right = fresh_envs(mps, mpo)
        A, C = split_forward(mps[0], mps._bond_itag(1))
        env_left[1] = step_left_env(env_left[0], A, mpo[0])
        out, _ = evolve_bond(C, env_left[1], env_right[0], -0.2, KRYLOV)
        assert abs(out.norm() - C.norm()) < 1e-12

    def test_evolve_bond_is_reversible(self, heisenberg_L6):
        mps, mpo = heisenberg_L6
        env_left, env_right = fresh_envs(mps, mpo)
        A, C = split_forward(mps[0], mps._bond_itag(1))
        env_left[1] = step_left_env(env_left[0], A, mpo[0])
        fwd, _ = evolve_bond(C, env_left[1], env_right[0], -0.2, KRYLOV)
        back, _ = evolve_bond(fwd, env_left[1], env_right[0], 0.2, KRYLOV)
        for key in C.data:
            assert torch.allclose(back.data[key], C.data[key], atol=1e-11)

    def test_evolve_1site_does_not_mutate_its_inputs(self, heisenberg_L2):
        """The local updates are pure, so they can be dispatched remotely."""
        mps, mpo = heisenberg_L2
        env_left, env_right = fresh_envs(mps, mpo)
        M_before = {k: v.clone() for k, v in mps[0].data.items()}
        W_before = {k: v.clone() for k, v in mpo[0].data.items()}
        E_before = {k: v.clone() for k, v in env_left[0].data.items()}

        evolve_1site(mps[0], mpo[0], env_left[0], env_right[0], 0.3, KRYLOV)

        for k, v in M_before.items():
            assert torch.equal(mps[0].data[k], v)
        for k, v in W_before.items():
            assert torch.equal(mpo[0].data[k], v)
        for k, v in E_before.items():
            assert torch.equal(env_left[0].data[k], v)

    def test_evolve_bond_does_not_mutate_its_inputs(self, heisenberg_L6):
        mps, mpo = heisenberg_L6
        env_left, env_right = fresh_envs(mps, mpo)
        A, C = split_forward(mps[0], mps._bond_itag(1))
        env_left[1] = step_left_env(env_left[0], A, mpo[0])
        C_before = {k: v.clone() for k, v in C.data.items()}

        evolve_bond(C, env_left[1], env_right[0], -0.2, KRYLOV)

        for k, v in C_before.items():
            assert torch.equal(C.data[k], v)
