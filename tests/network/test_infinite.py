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


"""Tests for the infinite network containers: iNetwork, iMPS, iMPO."""

from __future__ import annotations

import pytest
import torch
from nicole import Direction, Tensor
from nicole.index import Index, Sector

from alice.network import MPS, iMPO, iMPS, iNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bond(group, sectors, direction=Direction.IN, dim=2):
    """Build a bond index with `dim` states in each of `sectors`."""
    return Index(
        direction=direction,
        group=group,
        sectors=tuple(Sector(charge=q, dim=dim) for q in sectors),
    )


def _imps(spin_space, N, sectors=(-1, 0, 1), dim=2, seed=0):
    """Build a random iMPS unit cell of `N` sites with a uniform bond index.

    Every bond carries the same index, which is the simplest way to satisfy
    the ring-closure requirement, including the wrap-around bond.
    """
    Spc, _ = spin_space
    bulk = _bond(Spc.group, sectors, dim=dim)
    tensors = [
        Tensor.random(
            [bulk, bulk.flip(), Spc],
            seed=seed + i,
            itags=[f'A{i % N:02d}', f'A{(i + 1) % N:02d}', f's{i:02d}'],
        )
        for i in range(N)
    ]
    return iMPS(tensors)


def _impo(spin_space, N, sectors=(0,), dim=3, seed=100):
    """Build a random iMPO unit cell of `N` sites with a uniform bond index."""
    Spc, _ = spin_space
    bulk = _bond(Spc.group, sectors, dim=dim)
    tensors = [
        Tensor.random(
            [bulk, bulk.flip(), Spc, Spc.flip()],
            seed=seed + i,
            itags=[f'W{i % N:02d}', f'W{(i + 1) % N:02d}', f's{i:02d}', f's{i:02d}'],
        )
        for i in range(N)
    ]
    return iMPO(tensors)


# ---------------------------------------------------------------------------
# Ring closure
# ---------------------------------------------------------------------------

class TestRingClosure:
    """The unit cell must close on itself, wrap-around bond included."""

    @pytest.mark.parametrize('N', [1, 2, 3, 4, 7])
    def test_accepts_closed_cell(self, spin_space, N):
        imps = _imps(spin_space, N)
        assert imps.N == N
        assert imps.L == N

    def test_n1_cell_shares_one_bond(self, spin_space):
        # With a single site the left and right bonds are the same bond, so
        # both axes carry itag 'A00'. That is the translation-invariance
        # requirement, not a bug.
        imps = _imps(spin_space, 1)
        assert imps[0].itags[0] == imps[0].itags[1] == 'A00'
        assert imps[0].indices[0].direction != imps[0].indices[1].direction

    def test_rejects_open_wrap_bond(self, spin_space):
        # A cell whose last right bond is dimension 1 (an open end) does not
        # close; the error must name the wrap-around bond specifically.
        Spc, _ = spin_space
        bulk = _bond(Spc.group, (-1, 0, 1))
        trivial = _bond(Spc.group, (0,), dim=1)
        tensors = [
            Tensor.random([bulk, bulk.flip(), Spc], seed=1, itags=['A00', 'A01', 's00']),
            Tensor.random([bulk, trivial.flip(), Spc], seed=2, itags=['A01', 'A00', 's01']),
        ]
        with pytest.raises(ValueError, match='wrap-around'):
            iMPS(tensors)

    def test_rejects_wrap_itag_mismatch(self, spin_space):
        Spc, _ = spin_space
        bulk = _bond(Spc.group, (-1, 0, 1))
        tensors = [
            Tensor.random([bulk, bulk.flip(), Spc], seed=1, itags=['A00', 'A01', 's00']),
            # Right bond tagged 'A02' instead of wrapping back to 'A00'.
            Tensor.random([bulk, bulk.flip(), Spc], seed=2, itags=['A01', 'A02', 's01']),
        ]
        with pytest.raises(ValueError, match='wrap-around'):
            iMPS(tensors)

    def test_rejects_wrap_direction_clash(self, spin_space):
        Spc, _ = spin_space
        bulk = _bond(Spc.group, (-1, 0, 1))
        tensors = [
            Tensor.random([bulk, bulk.flip(), Spc], seed=1, itags=['A00', 'A01', 's00']),
            # Right bond left as IN, same direction as site 0's left bond.
            Tensor.random([bulk, bulk, Spc], seed=2, itags=['A01', 'A00', 's01']),
        ]
        with pytest.raises(ValueError, match='opposite directions'):
            iMPS(tensors)

    def test_rejects_empty_cell(self):
        with pytest.raises(ValueError, match='non-empty'):
            iNetwork([])


# ---------------------------------------------------------------------------
# Bond bookkeeping
# ---------------------------------------------------------------------------

class TestBondBookkeeping:
    """A cell of N sites has N bonds, not N-1."""

    @pytest.mark.parametrize('N', [1, 2, 5])
    def test_bond_dims_length_equals_cell_size(self, spin_space, N):
        imps = _imps(spin_space, N)
        assert len(imps.bond_dims) == N
        assert len(imps.bond_states) == N

    def test_bond_dims_values(self, spin_space):
        # Three sectors of dimension 2 each.
        imps = _imps(spin_space, 3, sectors=(-1, 0, 1), dim=2)
        assert imps.bond_dims == [6, 6, 6]

    def test_phys_dims(self, spin_space):
        imps = _imps(spin_space, 3)
        assert imps.phys_dims == [2, 2, 2]

    def test_bond_itag_wraps(self, spin_space):
        imps = _imps(spin_space, 4)
        assert imps._bond_itag(0) == 'A00'
        assert imps._bond_itag(3) == 'A03'
        # Bond 4 is bond 0: this identification is what closes the ring.
        assert imps._bond_itag(4) == 'A00'
        assert imps._bond_itag(5) == 'A01'

    def test_bond_itag_digit_width_grows(self, spin_space):
        imps = _imps(spin_space, 12)
        assert imps._bond_itag(0) == 'A00'
        assert imps._bond_itag(11) == 'A11'


# ---------------------------------------------------------------------------
# Per-type validation
# ---------------------------------------------------------------------------

class TestValidation:
    """iMPS and iMPO each enforce their own axis and itag conventions."""

    def test_imps_rejects_four_axes(self, spin_space):
        impo = _impo(spin_space, 2)
        with pytest.raises(ValueError, match='must have 3 axes'):
            iMPS(list(impo))

    def test_imps_rejects_wrong_phys_itag(self, spin_space):
        imps = _imps(spin_space, 2)
        t = imps[1].clone()
        t.retag(2, 's99')
        with pytest.raises(ValueError, match="itag 's01'"):
            iMPS([imps[0], t])

    def test_impo_rejects_three_axes(self, spin_space):
        imps = _imps(spin_space, 2)
        with pytest.raises(ValueError, match='must have 4 axes'):
            iMPO(list(imps))

    def test_impo_rejects_parallel_phys_axes(self, spin_space):
        Spc, _ = spin_space
        bulk = _bond(Spc.group, (0,), dim=3)
        # Both physical axes IN — they must be opposite for an operator.
        tensors = [
            Tensor.random(
                [bulk, bulk.flip(), Spc, Spc],
                seed=1, itags=['W00', 'W00', 's00', 's00'],
            ),
        ]
        with pytest.raises(ValueError, match='opposite directions'):
            iMPO(tensors)


# ---------------------------------------------------------------------------
# Separation from the finite family
# ---------------------------------------------------------------------------

class TestSeparateFromFinite:
    """An infinite network must not pass for a finite one."""

    def test_imps_is_not_an_mps(self, spin_space):
        imps = _imps(spin_space, 2)
        assert not isinstance(imps, MPS)

    def test_dmrg_rejects_infinite_input(self, spin_space):
        from alice import dmrg

        imps = _imps(spin_space, 2)
        impo = _impo(spin_space, 2)
        with pytest.raises(TypeError, match='idmrg.run'):
            dmrg.run(imps, impo)


# ---------------------------------------------------------------------------
# Container protocol, device, serialization
# ---------------------------------------------------------------------------

class TestContainerAndIO:

    def test_len_and_indexing(self, spin_space):
        imps = _imps(spin_space, 3)
        assert len(imps) == 3
        t = imps[1]
        imps[1] = t
        assert imps[1] is t

    def test_to_dtype(self, spin_space):
        imps = _imps(spin_space, 2)
        imps.to(dtype=torch.complex128)
        assert all(t.dtype == torch.complex128 for t in imps)

    @pytest.mark.parametrize('N', [1, 3])
    def test_serialize_roundtrip(self, spin_space, N):
        imps = _imps(spin_space, N)
        restored = iNetwork.deserialize(imps.serialize())
        assert isinstance(restored, iMPS)
        assert restored.N == imps.N
        assert restored.bond_dims == imps.bond_dims
        assert [t.itags for t in restored] == [t.itags for t in imps]

    def test_serialize_roundtrip_impo(self, spin_space):
        impo = _impo(spin_space, 2)
        restored = iNetwork.deserialize(impo.serialize())
        assert isinstance(restored, iMPO)
        assert restored.bond_dims == impo.bond_dims

    def test_deserialize_rejects_finite_payload(self, spin_space):
        # The two registries are separate, so a finite payload cannot be
        # revived as an infinite network.
        Spc, _ = spin_space
        trivial = _bond(Spc.group, (0,), dim=1)
        bulk = _bond(Spc.group, (-1, 1), dim=1)
        mps = MPS([
            Tensor.random([trivial, bulk.flip(), Spc], seed=1, itags=['A00', 'A01', 's00']),
            Tensor.random([bulk, trivial.flip(), Spc], seed=2, itags=['A01', 'A02', 's01']),
        ])
        with pytest.raises(ValueError, match='Unknown infinite network class'):
            iNetwork.deserialize(mps.serialize())

    def test_deserialize_rejects_bad_version(self, spin_space):
        payload = _imps(spin_space, 2).serialize()
        payload['version'] = 2
        with pytest.raises(ValueError, match='Unsupported serialization version'):
            iNetwork.deserialize(payload)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRepr:

    def test_imps_repr_marks_repetition(self, spin_space):
        text = repr(_imps(spin_space, 3))
        assert 'Infinite Matrix Product State (iMPS)' in text
        assert '···' in text
        assert 'cell:   3' in text

    def test_impo_repr(self, spin_space):
        text = repr(_impo(spin_space, 2))
        assert 'Infinite Matrix Product Operator (iMPO)' in text

    def test_inetwork_repr_is_compact(self, spin_space):
        # The bare base class has no diagram renderer; it falls back to the
        # one-line form that reports all N bonds.
        imps = _imps(spin_space, 3)
        text = repr(iNetwork(list(imps), itag_prefix='A'))
        assert text.startswith('iNetwork(N=3,')
        assert text.count('-') == 2
