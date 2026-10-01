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


"""Tests for alice.algorithm.tdvp.scheme_2s.

The 2-site scheme differs from the 1-site one mainly in that its splits are
*not* exact: they truncate, and that truncation is the only place where the
2-site scheme can lose norm. These tests therefore separate the algebra of
the effective Hamiltonian, which should be exact, from the splits, where the
interesting question is whether the reported discarded weight is honest.
"""

from __future__ import annotations


import torch
from nicole import einsum

from alice.algorithm.tdvp.krylov import inner_product
from alice.algorithm.tdvp.scheme_2s import (
    build_bulk,
    discarded_weight,
    evolve_2site,
    matvec_2s,
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
        out.data[key] = torch.complex(
            torch.randn(shape, generator=gen, dtype=torch.float64),
            torch.randn(shape, generator=gen, dtype=torch.float64),
        )
    return out


# ---------------------------------------------------------------------------
# Two-site bulk tensor
# ---------------------------------------------------------------------------

class TestBuildBulk:
    """Tests for the 2-site bulk tensor."""

    def test_index_order(self, heisenberg_L6):
        """The bulk carries (left, right, phys_left, phys_right)."""
        mps, _ = heisenberg_L6
        bulk = build_bulk(mps[0], mps[1])
        assert bulk.itags == (mps[0].itags[0], mps[1].itags[1],
                              mps[0].itags[2], mps[1].itags[2])

    def test_norm_matches_the_pair(self, heisenberg_L6):
        """Merging two tensors is a reshape, so the norm is untouched."""
        mps, _ = heisenberg_L6
        bulk = build_bulk(mps[0], mps[1])
        pair = einsum('abr,bcs->acrs', mps[0], mps[1])
        assert abs(bulk.norm() - pair.norm()) < 1e-13


# ---------------------------------------------------------------------------
# 2-site effective Hamiltonian
# ---------------------------------------------------------------------------

class TestMatvec2s:
    """Tests for the 2-site effective Hamiltonian."""

    def test_energy_matches_observe(self, heisenberg_L6):
        """⟨B|H_eff|B⟩ equals the full-network energy."""
        mps, mpo = heisenberg_L6
        env_left, env_right = fresh_envs(mps, mpo)
        bulk = build_bulk(mps[0], mps[1])
        HB = matvec_2s(bulk, mpo[0], mpo[1], env_left[0], env_right[1])
        assert abs(inner_product(bulk, HB).real - observe(mps, mpo)) < 1e-10

    def test_hermitian(self, heisenberg_L6):
        mps, mpo = heisenberg_L6
        env_left, env_right = fresh_envs(mps, mpo)
        bulk = build_bulk(mps[0], mps[1])
        other = _randomize_like(bulk, 31)
        lhs = inner_product(bulk, matvec_2s(other, mpo[0], mpo[1],
                                            env_left[0], env_right[1]))
        rhs = inner_product(other, matvec_2s(bulk, mpo[0], mpo[1],
                                             env_left[0], env_right[1]))
        assert abs(lhs - rhs.conjugate()) < 1e-12

    def test_preserves_index_structure(self, heisenberg_L6):
        mps, mpo = heisenberg_L6
        env_left, env_right = fresh_envs(mps, mpo)
        bulk = build_bulk(mps[0], mps[1])
        out = matvec_2s(bulk, mpo[0], mpo[1], env_left[0], env_right[1])
        assert out.itags == bulk.itags
        assert [i.direction for i in out.indices] == [i.direction for i in bulk.indices]

    def test_agrees_with_the_1site_matvec_on_a_shared_site(self, heisenberg_L6):
        """Compressing the 2-site matvec onto one site reproduces the 1-site one.

        With site 1 right-isometric, contracting it out of the 2-site result
        must give exactly the 1-site effective Hamiltonian at site 0, since
        the 2-site tangent space contains the 1-site one.
        """
        mps, mpo = heisenberg_L6
        from alice.algorithm.tdvp.scheme_1s import matvec_1s

        env_left, env_right = fresh_envs(mps, mpo)
        bulk = build_bulk(mps[0], mps[1])
        HB = matvec_2s(bulk, mpo[0], mpo[1], env_left[0], env_right[1])

        # Project the 2-site output back onto site 0 through B₁†.
        reduced = einsum('acru,bcu->abr', HB, mps[1].conj())
        direct = matvec_1s(mps[0], mpo[0], env_left[0], env_right[0])
        for key in direct.data:
            assert torch.allclose(reduced.data[key], direct.data[key], atol=1e-11)


# ---------------------------------------------------------------------------
# Splits and truncation
# ---------------------------------------------------------------------------

class TestSplits:
    """Tests for the truncating SVD splits."""

    def test_forward_roundtrip_is_exact_without_truncation(self, heisenberg_L6):
        """At full rank the split reproduces Θ block for block."""
        mps, _ = heisenberg_L6
        bulk = build_bulk(mps[2], mps[3])
        A, R = split_forward(bulk, mps._bond_itag(3), None)
        back = einsum('abr,bcs->acrs', A, R)
        for key in bulk.data:
            assert torch.allclose(back.data[key], bulk.data[key], atol=1e-12)

    def test_backward_roundtrip_is_exact_without_truncation(self, heisenberg_L6):
        mps, _ = heisenberg_L6
        bulk = build_bulk(mps[2], mps[3])
        L, V = split_backward(bulk, mps._bond_itag(3), None)
        back = einsum('abr,bcs->acrs', L, V)
        for key in bulk.data:
            assert torch.allclose(back.data[key], bulk.data[key], atol=1e-12)

    def test_forward_split_is_left_isometric(self, heisenberg_L6):
        mps, _ = heisenberg_L6
        bulk = build_bulk(mps[2], mps[3])
        A, _ = split_forward(bulk, mps._bond_itag(3), None)
        gram = einsum('abr,acr->bc', A.conj(), A)
        for block in gram.data.values():
            assert torch.allclose(
                block, torch.eye(block.shape[0], dtype=block.dtype), atol=1e-12
            )

    def test_backward_split_is_right_isometric(self, heisenberg_L6):
        mps, _ = heisenberg_L6
        bulk = build_bulk(mps[2], mps[3])
        _, V = split_backward(bulk, mps._bond_itag(3), None)
        gram = einsum('abr,cbr->ac', V.conj(), V)
        for block in gram.data.values():
            assert torch.allclose(
                block, torch.eye(block.shape[0], dtype=block.dtype), atol=1e-12
            )

    def test_nkeep_is_respected_per_sector(self, heisenberg_L6):
        """`nkeep` caps each charge sector, so the bond cannot run away."""
        mps, _ = heisenberg_L6
        bulk = build_bulk(mps[2], mps[3])
        A, _ = split_forward(bulk, mps._bond_itag(3), {'nkeep': 1})
        for sector in A.indices[1].sectors:
            assert sector.dim <= 1

    def test_truncation_shrinks_the_bond(self, heisenberg_L6):
        mps, _ = heisenberg_L6
        bulk = build_bulk(mps[2], mps[3])
        full, _ = split_forward(bulk, mps._bond_itag(3), None)
        cut, _ = split_forward(bulk, mps._bond_itag(3), {'nkeep': 1})
        assert cut.indices[1].dim < full.indices[1].dim


class TestDiscardedWeight:
    """Tests for the singular-value bookkeeping helper."""

    def test_zero_without_truncation(self, heisenberg_L6):
        mps, _ = heisenberg_L6
        bulk = build_bulk(mps[2], mps[3])
        assert discarded_weight(bulk, None) < 1e-20

    def test_positive_when_the_bond_is_cut(self, heisenberg_L6):
        """Cutting the bond hard must be admitted, not hidden."""
        mps, _ = heisenberg_L6
        bulk = build_bulk(mps[2], mps[3])
        assert discarded_weight(bulk, {'nkeep': 1}) > 0.0

    def test_matches_the_norm_actually_lost(self, heisenberg_L6):
        """The reported weight equals `1 - ‖kept‖²` for a normalized Θ.

        This checks the bookkeeping against the split output, rather than
        trusting the two independent SVDs to agree by construction.
        """
        mps, _ = heisenberg_L6
        bulk = build_bulk(mps[2], mps[3])
        bulk = bulk / bulk.norm()
        for trunc in ({'nkeep': 1}, {'nkeep': 2}, {'thresh': 1e-2}):
            A, R = split_forward(bulk, mps._bond_itag(3), trunc)
            kept = einsum('abr,bcs->acrs', A, R).norm()
            assert abs(discarded_weight(bulk, trunc) - (1.0 - kept ** 2)) < 1e-10

    def test_grows_as_the_cut_tightens(self, heisenberg_L6):
        mps, _ = heisenberg_L6
        bulk = build_bulk(mps[2], mps[3])
        hard = discarded_weight(bulk, {'nkeep': 1})
        soft = discarded_weight(bulk, {'nkeep': 2})
        assert hard > soft >= 0.0


# ---------------------------------------------------------------------------
# Local evolution
# ---------------------------------------------------------------------------

class TestEvolve2site:
    """Tests for `evolve_2site`."""

    def test_is_unitary(self, heisenberg_L6):
        mps, mpo = heisenberg_L6
        env_left, env_right = fresh_envs(mps, mpo)
        bulk = build_bulk(mps[0], mps[1])
        out, _ = evolve_2site(mps[0], mps[1], mpo[0], mpo[1],
                              env_left[0], env_right[1], 0.2, KRYLOV)
        assert abs(out.norm() - bulk.norm()) < 1e-11

    def test_is_reversible(self, heisenberg_L6):
        """Running the local propagator back recovers Θ.

        The backward leg has to be driven from the split tensors, since
        `evolve_2site` consumes a site pair rather than a Θ; splitting the
        forward output without truncation keeps the comparison exact.
        """
        mps, mpo = heisenberg_L6
        env_left, env_right = fresh_envs(mps, mpo)
        bulk = build_bulk(mps[0], mps[1])

        fwd, _ = evolve_2site(mps[0], mps[1], mpo[0], mpo[1],
                              env_left[0], env_right[1], 0.2, KRYLOV)
        A, R = split_forward(fwd, mps._bond_itag(1), None)
        back, _ = evolve_2site(A, R, mpo[0], mpo[1],
                               env_left[0], env_right[1], -0.2, KRYLOV)

        for key in bulk.data:
            assert torch.allclose(back.data[key], bulk.data[key], atol=1e-10)

    def test_conserves_the_local_energy(self, heisenberg_L6):
        mps, mpo = heisenberg_L6
        env_left, env_right = fresh_envs(mps, mpo)
        bulk = build_bulk(mps[0], mps[1])
        args = (mpo[0], mpo[1], env_left[0], env_right[1])
        before = inner_product(bulk, matvec_2s(bulk, *args)).real
        out, _ = evolve_2site(mps[0], mps[1], *args, 0.2, KRYLOV)
        after = inner_product(out, matvec_2s(out, *args)).real
        assert abs(after - before) < 1e-10

    def test_does_not_mutate_its_inputs(self, heisenberg_L6):
        mps, mpo = heisenberg_L6
        env_left, env_right = fresh_envs(mps, mpo)
        m_before = {k: v.clone() for k, v in mps[0].data.items()}
        w_before = {k: v.clone() for k, v in mpo[0].data.items()}
        e_before = {k: v.clone() for k, v in env_left[0].data.items()}

        evolve_2site(mps[0], mps[1], mpo[0], mpo[1],
                     env_left[0], env_right[1], 0.2, KRYLOV)

        for k, v in m_before.items():
            assert torch.equal(mps[0].data[k], v)
        for k, v in w_before.items():
            assert torch.equal(mpo[0].data[k], v)
        for k, v in e_before.items():
            assert torch.equal(env_left[0].data[k], v)
