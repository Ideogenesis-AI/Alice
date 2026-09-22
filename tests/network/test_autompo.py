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


"""Tests for AutoMPO, including InteractionNSite four-operator terms."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from nicole import Direction, Tensor, einsum, identity

from alice.algorithm.dmrg import dmrg
from alice.network import (
    Interaction2Site,
    InteractionNSite,
    build_hamiltonian,
    init_mps,
    observe,
)
from alice.physics.system import build_fermionic


# ---------------------------------------------------------------------------
# Local 4-order helpers (same construction as system.py / plugins)
# ---------------------------------------------------------------------------

def _make_leading4(op3: Tensor) -> Tensor:
    op4 = op3.clone()
    op4.insert_index(0, direction=Direction.IN, itag='_aux_')
    return op4.permute([0, 3, 1, 2])


def _make_terminal4(op3: Tensor) -> Tensor:
    op4 = op3.clone()
    op4.insert_index(3, direction=Direction.OUT, itag='_aux_')
    return op4.permute([2, 3, 0, 1])


def _make_intermid4(string_op: Tensor, leading_tnsr: Tensor) -> Tensor:
    op_idx_out = leading_tnsr.indices[1]
    op_id = identity(op_idx_out.flip())
    return einsum('op,rs->oprs', op_id, string_op)


def _channel_templates(ops):
    """Build single-channel fermionic MPO templates from build_fermionic ops."""
    C4 = _make_leading4(ops['C'])
    C4dag = _make_terminal4(ops['Cd'])
    F4 = _make_leading4(ops['F'])
    F4dag = _make_terminal4(ops['Fd'])
    C4mid = _make_intermid4(ops['Z'], C4)
    F4mid = _make_intermid4(ops['Z'], F4)
    return {
        'C4': C4, 'C4dag': C4dag, 'C4mid': C4mid,
        'F4': F4, 'F4dag': F4dag, 'F4mid': F4mid,
        'I4': ops['I4'], 'N4': ops['N4'],
    }


def _onebody_window(i: int, j: int, ch: dict) -> list:
    """Window for the one-body channel connected to c†_j c_i.

    Channel convention matching `oplus(G, Gdag)` pairing with cpl=+1:

    - create-left / annihilate-right → C4 / C4dag → c†_left c_right
    - annihilate-left / create-right → F4 / F4dag → c†_right c_left

    Alice's Jordan-Wigner strings are already absorbed into these templates,
    so both channels use positive coupling when combined into Hermitian hopping.
    """
    if i == j:
        return [ch['N4'].clone()]
    left, right = (i, j) if i < j else (j, i)
    span = right - left + 1
    if j > i:
        leading, mid, terminal = ch['F4'], ch['F4mid'], ch['F4dag']
    else:
        leading, mid, terminal = ch['C4'], ch['C4mid'], ch['C4dag']
    window = [leading.clone()]
    for _ in range(span - 2):
        window.append(mid.clone())
    window.append(terminal.clone())
    return window


def _four_op_window(m: int, n: int, k: int, l: int, ch: dict) -> list:
    """Concatenate two disjoint one-body windows for (c†_m c_l)(c†_n c_k)."""
    a_hi = max(m, l)
    b_lo = min(n, k)
    assert a_hi < b_lo, (
        f"Factors must be disjoint and ordered: "
        f"[{min(m, l)},{a_hi}] then [{b_lo},{max(n, k)}]"
    )
    win_a = _onebody_window(l, m, ch)
    win_b = _onebody_window(k, n, ch)
    gap = [ch['I4'].clone() for _ in range(b_lo - a_hi - 1)]
    return win_a + gap + win_b


def _hc_window_and_sites(m, n, k, l, ch):
    """Build Hermitian-conjugate window for (c†_m c†_n c_k c_l)†."""
    # Alice's JW-dressed F/C channels already match G4 with cpl=+1 for both,
    # so no extra fermion sign is applied on either the term or its conjugate.
    if max(l, m) < min(k, n):
        sites = [l, k, n, m]
        return _four_op_window(l, k, n, m, ch), sites
    if max(k, n) < min(l, m):
        sites = [k, l, m, n]
        return _four_op_window(k, l, m, n, ch), sites
    raise ValueError(f"Cannot order HC factors for (m,n,k,l)=({m},{n},{k},{l})")


# ---------------------------------------------------------------------------
# Numpy Jordan-Wigner reference
# ---------------------------------------------------------------------------

def _jw_ops(L: int):
    eye2 = np.eye(2, dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    sp = np.array([[0, 0], [1, 0]], dtype=complex)
    c_list, cd_list = [], []
    for j in range(L):
        mats_c, mats_cd = [], []
        for s in range(L):
            if s < j:
                mats_c.append(z)
                mats_cd.append(z)
            elif s == j:
                mats_c.append(sm)
                mats_cd.append(sp)
            else:
                mats_c.append(eye2)
                mats_cd.append(eye2)
        c, cd = mats_c[0], mats_cd[0]
        for s in range(1, L):
            c = np.kron(c, mats_c[s])
            cd = np.kron(cd, mats_cd[s])
        c_list.append(c)
        cd_list.append(cd)
    return c_list, cd_list


def _four_op_dense(m: int, n: int, k: int, l: int, L: int) -> np.ndarray:
    c, cd = _jw_ops(L)
    return cd[m] @ cd[n] @ c[k] @ c[l]


def _config_to_bitstring(config: list[int]) -> int:
    idx = 0
    for occ in config:
        idx = (idx << 1) | int(occ)
    return idx


def _all_configs(L: int, n_particles: int):
    for occupied in itertools.combinations(range(L), n_particles):
        cfg = [0] * L
        for i in occupied:
            cfg[i] = 1
        yield cfg


def _safe_observe(mps, mpo) -> float:
    """Like `observe`, but return 0.0 when charge conservation yields an empty tensor."""
    try:
        return observe(mps, mpo)
    except StopIteration:
        return 0.0


def _sector_eigs(H: np.ndarray, L: int, n_particles: int) -> np.ndarray:
    cfgs = list(_all_configs(L, n_particles))
    idx = [_config_to_bitstring(c) for c in cfgs]
    H2 = H[np.ix_(idx, idx)]
    return np.linalg.eigvalsh(0.5 * (H2 + H2.conj().T).real)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def ferm_u1():
    return build_fermionic('U1')


class TestInteractionNSiteValidation:
    """Validation errors for malformed InteractionNSite objects."""

    def test_tnsrs_none_raises(self, ferm_u1):
        spc, _ = ferm_u1
        intr = InteractionNSite(cpl=1.0, sites=[0, 1, 2, 3], tnsrs=None)
        with pytest.raises(ValueError, match='tnsrs=None'):
            build_hamiltonian([intr], 4, spc)

    def test_wrong_window_length_raises(self, ferm_u1):
        spc, ops = ferm_u1
        intr = InteractionNSite(
            cpl=1.0, sites=[0, 1, 2, 3], tnsrs=[ops['I4'].clone()] * 2,
        )
        with pytest.raises(ValueError, match='len\\(tnsrs\\)'):
            build_hamiltonian([intr], 4, spc)


class TestFourOperatorMPO:
    """Pin channel signs for concatenated four-fermion InteractionNSite terms."""

    @pytest.mark.parametrize(
        'L,m,n,k,l',
        [
            # Full-span windows on L=4 (DMRG-stable for this sparse MPO).
            # p>0: l < m < n < k
            (4, 1, 2, 3, 0),
            # p<0: m < l < k < n
            (4, 0, 3, 2, 1),
        ],
    )
    def test_dmrg_energy_matches_jw(self, ferm_u1, L, m, n, k, l):
        spc, ops = ferm_u1
        ch = _channel_templates(ops)
        window = _four_op_window(m, n, k, l, ch)
        intr = InteractionNSite(
            cpl=1.0, label=['TEST', 'PAIR'], sites=[m, n, k, l], tnsrs=window,
        )
        window_hc, sites_hc = _hc_window_and_sites(m, n, k, l, ch)
        intr_hc = InteractionNSite(
            cpl=1.0, label=['TEST', 'HC'], sites=sites_hc, tnsrs=window_hc,
        )
        mpo = build_hamiltonian([intr, intr_hc], L, spc, compact_every=0)

        H_ref = _four_op_dense(m, n, k, l, L)
        H_herm = H_ref + H_ref.conj().T
        eigs = _sector_eigs(H_herm, L, n_particles=2)
        e_ref = float(eigs[0])

        n_part = 2
        target_qn = 2 * n_part - L
        mps = init_mps(L, spc, ops, bond_dim=8, seed=0, target_qn=target_qn)
        opts = dmrg.Options(
            scheme='1sp', n_sweeps=12, e_tol=1e-12,
            trunc_thresh=1e-14, max_bond=32,
        )
        summary = dmrg.run(mps, mpo, opts)
        assert abs(summary.energy - e_ref) < 1e-8, (
            f"DMRG {summary.energy} != JW {e_ref} for (m,n,k,l)=({m},{n},{k},{l})"
        )

    def test_gapped_window_structure(self, ferm_u1):
        """Gapped factors insert I4 between the two one-body channels."""
        _, ops = ferm_u1
        ch = _channel_templates(ops)
        # l=0,m=1,n=3,k=4 → factors [0,1] and [3,4] with one I4 gap
        window = _four_op_window(1, 3, 4, 0, ch)
        assert len(window) == 5
        # Middle tensor is identity on the physical space (gap site 2).
        assert window[2].indices[0].dim == 1
        assert window[2].indices[1].dim == 1


    def test_density_density_p0(self, ferm_u1):
        """p=0 term is n_m n_n; window is N4, I4…, N4."""
        L = 4
        m, n = 0, 2
        spc, ops = ferm_u1
        ch = _channel_templates(ops)
        window = [
            ch['N4'].clone(),
            ch['I4'].clone(),
            ch['N4'].clone(),
        ]
        intr = InteractionNSite(
            cpl=1.0, label=['TEST', 'DENS'], sites=[m, n, n, m], tnsrs=window,
        )
        mpo = build_hamiltonian([intr], L, spc, compact_every=0)

        for cfg in _all_configs(L, 2):
            mps = init_mps(L, spc, ops, bond_dim=1, config=cfg)
            val = _safe_observe(mps, mpo)
            ref = float(cfg[m] * cfg[n])
            assert abs(val - ref) < 1e-10, f"cfg={cfg}: {val} != {ref}"

    def test_single_hopping_channel_matches_g4(self, ferm_u1):
        """C and F channels together reproduce the G4/G4dag Hermitian hopping."""
        L = 4
        spc, ops = ferm_u1
        ch = _channel_templates(ops)

        hop = Interaction2Site(
            cpl=1.0,
            label=['NN'],
            leading_site=0,
            terminal_site=1,
            leading_tnsr=ops['G4'].clone(),
            terminal_tnsr=ops['G4dag'].clone(),
        )
        mpo_g = build_hamiltonian([hop], L, spc, compact_every=0)

        # Both channels use cpl=+1: Alice's JW strings make F…Fdag match
        # the G4 annihilation/creation pairing without an extra minus.
        win_c = _onebody_window(1, 0, ch)  # c†_0 c_1
        win_f = _onebody_window(0, 1, ch)  # c†_1 c_0
        intr_c = InteractionNSite(
            cpl=1.0, sites=[0, 1], tnsrs=[t.clone() for t in win_c],
        )
        intr_f = InteractionNSite(
            cpl=1.0, sites=[0, 1], tnsrs=[t.clone() for t in win_f],
        )
        mpo_ch = build_hamiltonian([intr_c, intr_f], L, spc, compact_every=0)

        for seed in range(3):
            mps = init_mps(L, spc, ops, bond_dim=12, seed=seed)
            assert abs(observe(mps, mpo_g) - observe(mps, mpo_ch)) < 1e-8
