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

import math

import pytest
import torch
from nicole import Direction, Tensor, einsum
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


def _gaugeable_imps(spin_space, N, width=2, dim=2, seed=1):
    """Build a random iMPS whose unit cell can actually be gauge fixed.

    `_imps` gives every bond the same charges, which for a spin-1/2 physical
    space makes the transfer operator bipartite: the physical charges are
    odd, so a bond charge reachable at even positions is not reachable at
    odd ones. Here bond `k` carries only charges of parity `k`, which is
    what a physical state looks like and what lets the power iteration
    converge. `N` must be even for the cell to close.
    """
    Spc, _ = spin_space
    def bond(k):
        charges = tuple(
            q for q in range(-2 * width, 2 * width + 1) if (q - k) % 2 == 0
        )
        return _bond(Spc.group, charges, dim=dim)

    return iMPS([
        Tensor.random(
            [bond(i), bond(i + 1).flip(), Spc],
            seed=seed * 10 + i,
            itags=[f'A{i % N:02d}', f'A{(i + 1) % N:02d}', f's{i:02d}'],
        )
        for i in range(N)
    ])


def _dimer_imps(spin_space):
    """Build the exact Majumdar-Ghosh dimer state as a 2-site iMPS.

    Sites 0 and 1 share a singlet and the pattern repeats, so bond 0 cuts
    no singlet (one Schmidt value) while bond 1 cuts one (two equal Schmidt
    values). That makes the Schmidt spectrum known in closed form, which is
    what turns this into a real check rather than a self-consistency one.
    """
    Spc, _ = spin_space
    trivial = _bond(Spc.group, (0,), dim=1)
    half = _bond(Spc.group, (-1, 1), dim=1)

    opener = Tensor.zeros(
        [trivial, half.flip(), Spc], itags=['A00', 'A01', 's00'],
    )
    # The open singlet records which spin was left behind on site 0.
    opener.data[(0, 1, 1)][:] = 1.0
    opener.data[(0, -1, -1)][:] = 1.0

    closer = Tensor.zeros(
        [half, trivial.flip(), Spc], itags=['A01', 'A00', 's01'],
    )
    # ... and site 1 takes the opposite spin, antisymmetrically.
    closer.data[(1, 0, -1)][:] = 1.0 / math.sqrt(2.0)
    closer.data[(-1, 0, 1)][:] = -1.0 / math.sqrt(2.0)

    return iMPS([opener, closer])


def _neel_imps(spin_space, N):
    """Build the Neel product state as an `N`-site iMPS, `N` even.

    Every bond is one-dimensional, so the state is already canonical and
    every bond matrix is a number. A polarized product state would be the
    simpler choice but cannot close a ring under U(1): each site would
    push the bond charge up by one and nothing would bring it back.
    """
    Spc, _ = spin_space
    # The bond charge records the spin the previous site put there, so it
    # alternates with the pattern rather than staying at zero.
    bonds = [_bond(Spc.group, (k % 2,), dim=1) for k in range(N + 1)]
    tensors = []
    for i in range(N):
        spin = 1 if i % 2 == 0 else -1
        t = Tensor.zeros(
            [bonds[i], bonds[i + 1].flip(), Spc],
            itags=[f'A{i % N:02d}', f'A{(i + 1) % N:02d}', f's{i:02d}'],
        )
        t.data[(i % 2, (i + 1) % 2, spin)][:] = 1.0
        tensors.append(t)
    return iMPS(tensors)


def _deviation_from_identity(gram):
    """Return the largest entry by which `gram` misses the identity."""
    worst = 0.0
    for blk in gram.data.values():
        mat = blk.reshape(blk.shape[0], blk.shape[1], -1)[:, :, 0]
        eye = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
        worst = max(worst, float((mat - eye).abs().max()))
    return worst


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
# Canonical form
# ---------------------------------------------------------------------------

class TestCanonicalForm:
    """`canonical` must leave the three defining relations exactly satisfied."""

    @pytest.mark.parametrize('N', [2, 4, 6])
    def test_isometries_are_orthonormal(self, spin_space, N):
        imps = _gaugeable_imps(spin_space, N)
        imps.canonical()
        for i in range(N):
            left = imps.left_isometries()[i]
            right = imps.right_isometries()[i]
            assert _deviation_from_identity(
                einsum('las,lbs->ab', left.conj(), left)
            ) < 1e-10
            assert _deviation_from_identity(
                einsum('als,bls->ab', right.conj(), right)
            ) < 1e-10

    @pytest.mark.parametrize('N', [2, 4, 6])
    def test_gauge_relation_holds(self, spin_space, N):
        # AL[i] C[i+1] == C[i] AR[i] is what makes AL, AR and C describe one
        # state rather than three unrelated ones.
        imps = _gaugeable_imps(spin_space, N)
        imps.canonical()
        for i in range(N):
            through_left = einsum(
                'lms,mb->lbs', imps.left_isometries()[i], imps.bond_matrix(i + 1),
            )
            assert float((through_left - imps.center_tensor(i)).norm()) < 1e-10

    @pytest.mark.parametrize('N', [2, 4])
    def test_schmidt_weights_are_normalized(self, spin_space, N):
        imps = _gaugeable_imps(spin_space, N)
        imps.canonical()
        for bond in range(N):
            values = imps.schmidt_values(bond)
            assert values == sorted(values, reverse=True)
            assert sum(v * v for v in values) == pytest.approx(1.0)

    def test_bond_matrix_wraps(self, spin_space):
        imps = _gaugeable_imps(spin_space, 2)
        imps.canonical()
        assert imps.bond_matrix(2) is imps.bond_matrix(0)
        assert imps.bond_matrix(-1) is imps.bond_matrix(1)

    def test_center_defaults_to_target(self, spin_space):
        imps = _gaugeable_imps(spin_space, 4)
        imps.canonical(target=2)
        assert imps.center == 2
        expected = einsum(
            'ab,brs->ars', imps.bond_matrix(2), imps.right_isometries()[2],
        )
        assert float((imps.center_tensor() - expected).norm()) < 1e-12

    def test_negative_target_counts_from_the_end(self, spin_space):
        imps = _gaugeable_imps(spin_space, 4)
        imps.canonical(target=-1)
        assert imps.center == 3

    def test_bond_dimensions_are_untouched(self, spin_space):
        # QR keeps every bond exactly as wide as it was; truncation is the
        # sweep's job, not the gauge's.
        imps = _gaugeable_imps(spin_space, 4)
        before = imps.bond_dims
        imps.canonical()
        assert imps.bond_dims == before

    def test_second_call_is_a_fixed_point(self, spin_space):
        imps = _gaugeable_imps(spin_space, 4)
        imps.canonical()
        first = [imps.schmidt_values(k) for k in range(imps.N)]
        imps.canonical()
        for before, after in zip(first, (imps.schmidt_values(k) for k in range(imps.N))):
            assert after == pytest.approx(before, abs=1e-10)


class TestCanonicalKnownStates:
    """States whose canonical form is known in closed form."""

    def test_dimer_schmidt_spectrum(self, spin_space):
        # Bond 0 lies inside no singlet and bond 1 cuts exactly one, so the
        # spectra are [1] and [1/sqrt(2), 1/sqrt(2)] with nothing in between.
        imps = _dimer_imps(spin_space)
        imps.canonical()
        assert imps.schmidt_values(0) == pytest.approx([1.0])
        assert imps.schmidt_values(1) == pytest.approx(
            [1.0 / math.sqrt(2.0)] * 2
        )

    @pytest.mark.parametrize('N', [2, 4])
    def test_product_state_is_unentangled(self, spin_space, N):
        imps = _neel_imps(spin_space, N)
        imps.canonical()
        for bond in range(N):
            assert imps.schmidt_values(bond) == pytest.approx([1.0])


class TestCanonicalFailures:
    """Gauge fixing must refuse rather than hand back an inconsistent state."""

    def test_two_cycle_is_reported(self, spin_space):
        # A uniform bond index mixes both charge parities, which the odd
        # physical charges then swap, so the iteration alternates forever.
        with pytest.raises(ValueError, match='two-cycle'):
            _imps(spin_space, 3).canonical()

    def test_zero_amplitude_cell_is_reported(self, spin_space):
        # Even bond charges are unreachable from each other through odd
        # physical ones, so every tensor in this cell is identically zero.
        with pytest.raises(ValueError, match='no amplitude'):
            _imps(spin_space, 2, sectors=(-2, 0, 2)).canonical()

    def test_iteration_cap_is_reported(self, spin_space):
        with pytest.raises(ValueError, match='did not converge in 2 sweeps'):
            _gaugeable_imps(spin_space, 4).canonical(maxiter=2)

    @pytest.mark.parametrize('target', [4, -5])
    def test_target_out_of_range(self, spin_space, target):
        with pytest.raises(IndexError, match='out of range'):
            _gaugeable_imps(spin_space, 4).canonical(target=target)

    @pytest.mark.parametrize(
        'call',
        [
            lambda m: m.bond_matrix(0),
            lambda m: m.left_isometries(),
            lambda m: m.right_isometries(),
            lambda m: m.center_tensor(0),
            lambda m: m.schmidt_values(0),
        ],
    )
    def test_accessors_require_canonical(self, spin_space, call):
        with pytest.raises(ValueError, match='not gauge fixed'):
            call(_gaugeable_imps(spin_space, 2))


class TestCanonicalLifetime:
    """The gauge must not outlive the tensors it describes."""

    def test_setitem_discards_the_gauge(self, spin_space):
        imps = _gaugeable_imps(spin_space, 2)
        imps.canonical()
        imps[0] = imps[0]
        assert imps.center is None
        with pytest.raises(ValueError, match='not gauge fixed'):
            imps.bond_matrix(0)

    def test_serialize_roundtrip_keeps_the_gauge(self, spin_space):
        imps = _gaugeable_imps(spin_space, 2)
        imps.canonical(target=1)
        restored = iNetwork.deserialize(imps.serialize())
        assert restored.center == 1
        for bond in range(imps.N):
            assert restored.schmidt_values(bond) == pytest.approx(
                imps.schmidt_values(bond)
            )
        for before, after in zip(imps.left_isometries(), restored.left_isometries()):
            assert float((before - after).norm()) < 1e-12

    def test_serialize_roundtrip_without_a_gauge(self, spin_space):
        restored = iNetwork.deserialize(_gaugeable_imps(spin_space, 2).serialize())
        with pytest.raises(ValueError, match='not gauge fixed'):
            restored.bond_matrix(0)

    def test_to_dtype_carries_the_gauge(self, spin_space):
        imps = _gaugeable_imps(spin_space, 2)
        imps.canonical()
        imps.to(dtype=torch.complex128)
        assert imps.bond_matrix(0).dtype == torch.complex128
        assert all(t.dtype == torch.complex128 for t in imps.left_isometries())


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
