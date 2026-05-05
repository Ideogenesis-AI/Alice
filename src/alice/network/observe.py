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
    - `MPO` (thermal density matrix): not yet implemented.

    Parameters
    ----------
    state:
        The state to evaluate.  Either an `MPS` object, or a plain sequence
        of MPS site tensors.
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
        If `state` is not an `MPS` or a sequence of tensors.
    NotImplementedError
        If `state` is an `MPO` (thermal density matrix support is pending).
    """
    if isinstance(state, MPO):
        raise NotImplementedError(
            "observe for thermal states (MPO density matrix) is not yet implemented."
        )
    if isinstance(state, (MPS, Sequence)):
        return _observe_mps(state, observable)
    raise TypeError(
        f"state must be an MPS or a sequence of tensors, got {type(state).__name__!r}"
    )


def _observe_mps(
    mps: Union[MPS, Sequence[Tensor]],
    mpo: Union[MPO, Sequence[Tensor]],
) -> float:
    """Compute ⟨ψ|O|ψ⟩ by a left-to-right MPS-MPO-MPS contraction.

    Performs a transfer-matrix sweep from site 0 to site L−1, accumulating a
    three-legged environment `E[bra_bond, mpo_bond, ket_bond]` at each step.

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
    The left boundary environment is initialised as an identity on the dim-1
    left bond of `mps[0]`, extended with a dim-1 MPO bond index. At each
    site the environment is updated via `einsum('ace,abg,cdgh,efh->bdf', ...)`,
    where the letters denote:

    - a, b — bra (conj MPS) left and right bonds
    - c, d — MPO left and right bonds
    - e, f — ket (MPS) left and right bonds
    - g — physical bra index (shared between bra and MPO axis 2)
    - h — physical ket index (shared between MPO axis 3 and ket)

    a, c, e are contracted against E; b, d, f become the updated E.

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
        E = einsum('ace,abg,cdgh,efh->bdf', E, mps[i].conj(), mpo[i], mps[i])

    # E is now a 1×1×1 tensor at the right boundary. Extract the scalar,
    # accounting for the Bridge normalization weight in non-Abelian groups.
    k, v = next(iter(E.data.items()))
    weight = 1.0 if E.intw is None else float(E.intw[k].weights[0, 0])

    return float(v.item()) * weight
