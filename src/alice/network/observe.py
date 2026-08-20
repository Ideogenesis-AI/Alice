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


"""Expectation-value computation for MPS and thermal MPO states."""

from __future__ import annotations

import math
from typing import Sequence, Union

from nicole import Direction, Tensor, einsum, identity

from .network import MPS, MPO


def observe(
    state: Union[MPS, MPO, Sequence[Tensor]],
    observable: Union[MPO, Sequence[Tensor]],
) -> float:
    """Compute the expectation value of an observable for a given state.

    Dispatches to the appropriate contraction routine based on the type of
    `state`:

    - `MPS` (or a plain sequence of tensors): evaluates ⟨ψ|O|ψ⟩ via a
      left-to-right MPS-MPO-MPS transfer-matrix sweep.
    - `NormalMPO` (thermal density matrix): evaluates
      `Tr[ρ O] / Tr[ρ]` via a left-to-right transfer-matrix sweep that
      accumulates a small `(ρ_bond, O_bond)` environment, analogous to the
      MPS case.

    Parameters
    ----------
    state:
        The state to evaluate. Either an `MPS` object, a plain sequence of
        MPS site tensors, or a `NormalMPO` thermal density matrix.
    observable:
        The observable encoded as an `MPO` object, or a plain sequence of
        MPO site tensors, of the same length as `state`.

    Returns
    -------
    float
        The expectation value of the observable.

    Raises
    ------
    TypeError
        If `state` is not an `MPS`, `NormalMPO`, or a sequence of tensors.
    NotImplementedError
        If `state` is a plain `MPO` (use a `NormalMPO` for thermal states).
    """
    # Import here to avoid a top-level circular dependency
    # (thermal.py imports from network.py, which is fine, but we cannot
    # import NormalMPO at module load time without potential issues).
    from .thermal import NormalMPO

    if isinstance(state, NormalMPO):
        return _observe_thermal(state, observable)
    if isinstance(state, MPO):
        raise NotImplementedError(
            "observe() does not support a plain MPO as the state. "
            "Wrap the density matrix in a NormalMPO first."
        )
    if isinstance(state, (MPS, Sequence)):
        return _observe_mps(state, observable)
    raise TypeError(
        f"state must be an MPS, NormalMPO, or a sequence of tensors, "
        f"got {type(state).__name__!r}"
    )


def _observe_thermal(rho, observable: Union[MPO, Sequence[Tensor]]) -> float:
    """Compute the thermal expectation value `Tr[ρ O] / Tr[ρ]`.

    Evaluates the numerator via a left-to-right transfer-matrix sweep that
    accumulates a 2nd-order environment `E[ρ_bond, O_bond]`, analogous to
    `_observe_mps`'s bra-mpo-ket sweep. At each site, `ρ`'s phys_out is
    contracted against `O`'s phys_in (matrix product) and `ρ`'s phys_in
    against `O`'s phys_out (closing the physical trace loop) in a single
    `einsum` call.

    The ratio is computed via log-space combination (rather than
    `trace() / trace()`), so it remains correct even when either trace
    would overflow float64 (e.g. deep into an XTRG run at low
    temperature), even though the ratio itself is a well-behaved O(1)
    number. The denominator uses `rho.log_trace()` directly (a single MPO
    trace sweep — already cheap). The numerator's raw magnitude comes
    from the environment sweep on `rho` and `O_norm`'s internal
    (unit-normed) tensors, so it is combined with `rho.log_scale +
    O_norm.log_scale` to recover the full log-magnitude.

    Parameters
    ----------
    rho:
        Thermal density matrix as a `NormalMPO`.
    observable:
        The observable as an `MPO` (or plain sequence of MPO site tensors).

    Returns
    -------
    float
        The thermal expectation value `Tr[ρ O] / Tr[ρ]`.

    Raises
    ------
    ValueError
        If `rho` and `observable` have different lengths.
    ZeroDivisionError
        If `Tr[ρ]` is exactly zero.

    Notes
    -----
    Letters used in the per-site `einsum`:

    - `a`, `c` — ρ's left and right bond
    - `b`, `d` — O's left and right bond
    - `s` — ρ's phys_out paired with O's phys_in (matrix-product contraction)
    - `r` — ρ's phys_in paired with O's phys_out (closes the trace loop)

    `a`, `b` are contracted against `E`; `c`, `d` become the updated `E`.
    """
    from .thermal import NormalMPO

    if not isinstance(observable, MPO):
        observable = MPO(list(observable))
    if len(observable) != rho.L:
        raise ValueError(
            f"rho and observable must have the same length, got {rho.L} and {len(observable)}"
        )

    O_norm = NormalMPO.from_mpo(observable)

    log_den, sign_den = rho.log_trace()
    if sign_den == 0.0:
        raise ZeroDivisionError("Tr[ρ] is numerically zero; cannot compute expectation value")

    # Left boundary: both rho[0] and O_norm[0] have their left bond in the
    # IN direction (standard MPO convention), so E needs two independent
    # OUT axes to contract against them. Build it via identity() (which
    # gives one IN, one OUT axis on rho[0]'s trivial left bond), insert a
    # second OUT axis for O_norm's left bond, then discard the leftover
    # (unused) IN axis with squeeze().
    E = identity(rho[0].indices[0])
    E.retag([0, 1], ['_thermal_env_', rho[0].itags[0]])
    E.insert_index(2, direction=Direction.OUT, itag=O_norm[0].itags[0])
    E.squeeze(0)
    # E axes: (rho_left=a, O_left=b), both OUT and dim-1.

    for i in range(rho.L):
        # absorb rho and O into E; see Notes for letter definitions
        E = einsum('ab,acrs,bdsr->cd', E, rho[i], O_norm[i])

    # E is now a 1×1 tensor at the right boundary. Extract the scalar,
    # accounting for the Bridge normalization weight in non-Abelian groups.
    key, val = next(iter(E.data.items()))
    weight = 1.0 if E.intw is None else float(E.intw[key].weights[0, 0])
    raw_num = float(val.item().real) * weight

    if raw_num == 0.0:
        return 0.0
    log_num = math.log(abs(raw_num)) + rho.log_scale + O_norm.log_scale
    sign_num = math.copysign(1.0, raw_num)

    return (sign_num * sign_den) * math.exp(log_num - log_den)


def _observe_mps(
    mps: Union[MPS, Sequence[Tensor]],
    mpo: Union[MPO, Sequence[Tensor]],
) -> float:
    """Compute ⟨ψ|O|ψ⟩ by a left-to-right MPS-MPO-MPS contraction.

    Performs a transfer-matrix sweep from site 0 to site L−1, accumulating a
    3rd-order environment `E[bra_bond, mpo_bond, ket_bond]` at each step.

    The MPO tensors must follow the axis layout:

        axis 0 — left bond  (IN direction)
        axis 1 — right bond (OUT direction)
        axis 2 — phys_bra   (IN direction,  contracts with conj MPS physical)
        axis 3 — phys_ket   (OUT direction, contracts with MPS physical)

    The MPS tensors must follow the standard layout:

        axis 0 — left bond  (IN direction)
        axis 1 — right bond (OUT direction)
        axis 2 — physical   (IN direction)

    Parameters
    ----------
    mps:
        Sequence of MPS site tensors (or an `MPS` object).
    mpo:
        Sequence of MPO site tensors (or an `MPO` object) of the same length.

    Returns
    -------
    float
        The expectation value ⟨ψ|O|ψ⟩.

    Notes
    -----
    The left boundary environment is initialized as an identity on the dim-1
    left bond of `mps[0]`, extended with a dim-1 MPO bond index. At each
    site the environment is updated via `einsum('aob,acr,oprs,bds->cpd', ...)`,
    where the letters denote:

    - a, c — bra (conj MPS) left and right bonds
    - b, d — ket (MPS) left and right bonds
    - o, p — MPO left and right bonds
    - r — physical bra index (shared between bra and MPO axis 2)
    - s — physical ket index (shared between MPO axis 3 and ket)

    a, o, b are contracted against E; c, p, d become the updated E.

    After the right boundary E is a 1×1×1 tensor. The scalar is read from
    its single data block, multiplied by the Bridge weight for non-Abelian
    symmetry groups.
    """
    L = len(mps)
    if len(mpo) != L:
        raise ValueError(
            f"mps and mpo must have the same length, got {L} and {len(mpo)}"
        )

    # Left boundary: identity on the dim-1 left bond of mps[0], then insert
    # a dim-1 MPO bond index so E has shape (bra_left, mpo_left, ket_left).
    E = identity(mps[0].indices[0])
    E.retag([0, 1], [mps[0].itags[0], mps[0].itags[0]])
    E.insert_index(1, direction=Direction.OUT, itag=mpo[0].itags[0])
    # E axes: (bra_left, mpo_left, ket_left)

    for i in range(L):
        # absorb bra, MPO, and ket into E; see Notes for letter definitions
        E = einsum('aob,acr,oprs,bds->cpd', E, mps[i].conj(), mpo[i], mps[i])

    # E is now a 1×1×1 tensor at the right boundary. Extract the scalar,
    # accounting for the Bridge normalization weight in non-Abelian groups.
    k, v = next(iter(E.data.items()))
    weight = 1.0 if E.intw is None else float(E.intw[k].weights[0, 0])

    return float(v.item()) * weight
