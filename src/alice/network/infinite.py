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


"""iMPS, iMPO, and base tensor network class for infinite 1D chains.

An infinite network stores one translation-invariant unit cell of `N` site
tensors. Unlike the finite `Network` family the unit cell closes on itself:
the right bond of site `N-1` is the *same* bond as the left bond of site `0`,
so the cell has `N` bonds rather than `N-1` and both "ends" carry a full
virtual index instead of a trivial dimension-1 boundary.

That closure is why this family does not derive from `Network`. Several of
the finite methods are not merely inconvenient on a ring, they are wrong:
`bond_dims` counts `L-1` bonds, `canonical` is an open-boundary sweep that
pins the center at site 0, and `norm` / `MPO.compact` contract against
boundary vectors that do not exist here. Keeping the two families unrelated
also means `isinstance(x, MPS)` is `False` for an infinite state, so the
finite entry points reject one instead of silently producing nonsense.

Axis layout (shared with the finite family):

- Axis 0: left bond
- Axis 1: right bond
- Axis 2: physical index (ket / phys_in)
- Axis 3 (iMPO only): conjugate physical index (bra / phys_out)

Bond itag convention: the left bond of site *i* carries itag
`{prefix}{i % N:0M}` and the right bond carries `{prefix}{(i+1) % N:0M}`,
where `prefix` is `itag_prefix` (`'A'` for `iMPS`, `'W'` for `iMPO`) and *M*
is `max(2, len(str(N)))`. The modulo is what closes the ring: at `i = N-1`
the right bond is tagged `{prefix}00`, matching the left bond of site `0`.

Physical itag convention: site *i* physical axis carries itag `s{i:02d}`,
exactly as in the finite family.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from nicole import Tensor, decomp
from nicole import serialize as _serialize_tensor, deserialize as _deserialize_tensor

# Default itag prefix for the bare `iNetwork` class. Subclasses override it.
_DEFAULT_ITAG_PREFIX = '_init_'

# Guards the division that pulls a phase out of a diagonal entry.
_TINY = 1e-300


def _scale_axis(blk: torch.Tensor, scale: torch.Tensor, axis: int) -> None:
    """Multiply `blk` along `axis` by the per-index factors in `scale`."""
    shape = [1] * blk.dim()
    shape[axis] = -1
    blk.mul_(scale.reshape(shape))


def _qr_positive(t: Tensor, axes: List[int], itag: str) -> Tuple[Tensor, Tensor]:
    """QR decomposition whose triangular factor has a non-negative diagonal.

    `decomp(..., mode='QR')` inherits LAPACK's sign freedom: for any diagonal
    sign matrix `D`, `(Q D, D R)` factorizes the same tensor just as well.
    That is harmless once, but fatal inside a fixed-point iteration, where it
    lets the gauge alternate between `+R` and `-R` and never meet its
    tolerance. Pinning the phase of the diagonal makes the factorization
    unique, which is what the Majumdar-Ghosh dimer state needs to converge
    at all.

    Parameters
    ----------
    t:
        Tensor to decompose.
    axes:
        Axes forming the isometric factor. The new bond is appended to them.
    itag:
        itag for the new bond.

    Returns
    -------
    tuple
        `(Q, R)` with `Q` isometric over `axes` and the diagonal of `R` real
        and non-negative.
    """
    q, r = decomp(t, axes=axes, mode='QR', itag=itag)

    # The bond sits on axis 0 of R and on the last axis of Q, and a block key
    # carries one charge per axis, so the two factors match up by charge.
    phases = {}
    for key, blk in r.data.items():
        # R has two indices and therefore a single intertwiner, so its
        # leading slice holds the whole matrix for a generic group too.
        diag = torch.diagonal(blk.reshape(blk.shape[0], blk.shape[1], -1)[:, :, 0])
        mag = diag.abs()
        phase = torch.where(
            mag > 0, diag / mag.clamp_min(_TINY), torch.ones_like(diag)
        )
        # A rectangular R has rows past the diagonal; those keep phase 1.
        full = torch.ones(blk.shape[0], dtype=blk.dtype, device=blk.device)
        full[:phase.numel()] = phase
        phases[key[0]] = full
        _scale_axis(blk, full.conj(), 0)

    bond_axis = len(q.indices) - 1
    for key, blk in q.data.items():
        phase = phases.get(key[-1])
        if phase is not None:
            _scale_axis(blk, phase, bond_axis)
    return q, r


class iNetwork:  # noqa: N801 — iMPS/iMPO naming follows the physics convention
    """Base class for 1D tensor network unit cells with infinite boundaries.

    Stores the `N` site tensors of one translation-invariant unit cell and
    provides uniform iteration, bond bookkeeping, and axis conventions.
    Subclasses add the per-type validation (`iMPS`, `iMPO`) and, for states,
    the gauge-fixing machinery.
    """

    def __init__(
        self,
        tensors: List[Tensor],
        center: Optional[int] = None,
        itag_prefix: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        tensors:
            Non-empty list of the unit cell's site tensors.
        center:
            Orthogonality center site index, or `None` if the canonical form
            is unspecified.
        itag_prefix:
            Bond itag prefix string. Defaults to `'_init_'` when `None`.
            Subclasses override this default (`'A'` for `iMPS`, `'W'` for
            `iMPO`), so supply this argument only when a non-default prefix
            is explicitly required.

        Raises
        ------
        ValueError
            If `tensors` is empty, or if the unit cell does not close into a
            consistent ring.
        """
        if not tensors:
            raise ValueError("tensors must be a non-empty list")
        self._tensors: List[Tensor] = list(tensors)
        # Fixed for the whole family; kept as an attribute so that code which
        # inspects `bc` can treat finite and infinite networks uniformly.
        self.bc: str = 'IBC'
        self._center: Optional[int] = center
        self._itag_prefix: str = itag_prefix if itag_prefix is not None else _DEFAULT_ITAG_PREFIX
        self._validate()

    def _validate(self) -> None:
        """Check that the unit cell closes into a consistent ring.

        For each of the `N` bonds — including the wrap-around bond between
        site `N-1` and site `0` — verifies that the right index of the left
        site and the left index of the right site:

        - share the same itag,
        - have opposite directions (so they can be contracted), and
        - agree on the dimension of every charge sector present in both.

        For `N == 1` the single bond is the tensor's own left/right pair,
        which is exactly the translation-invariance requirement.

        Subclasses should override this method to add further checks (axis
        count, physical itags) and call `super()._validate()`.

        Raises
        ------
        ValueError
            If any bond consistency check fails.
        """
        N = self.N
        for i in range(N):
            j = (i + 1) % N
            r = self._tensors[i].indices[1]
            l = self._tensors[j].indices[0]
            r_itag = self._tensors[i].itags[1]
            l_itag = self._tensors[j].itags[0]
            # Name the wrap-around bond explicitly; a mismatch there is the
            # most likely failure and the least obvious one to diagnose.
            where = f"between sites {i} and {j}" + (" (wrap-around)" if j < i else "")

            if r_itag != l_itag:
                raise ValueError(
                    f"Bond {where}: itag mismatch "
                    f"('{r_itag}' on site {i}'s right bond vs "
                    f"'{l_itag}' on site {j}'s left bond)"
                )
            if r.direction == l.direction:
                raise ValueError(
                    f"Bond {where}: right index of site {i} and left index "
                    f"of site {j} must have opposite directions"
                )
            r_dims = {s.charge: s.dim for s in r.sectors}
            l_dims = {s.charge: s.dim for s in l.sectors}
            for charge, dim in r_dims.items():
                if charge in l_dims and l_dims[charge] != dim:
                    raise ValueError(
                        f"Bond {where}: sector charge {charge} has dimension "
                        f"{dim} on site {i}'s right bond but {l_dims[charge]} "
                        f"on site {j}'s left bond"
                    )

    # ------------------------------------------------------------------
    # itag_prefix
    # ------------------------------------------------------------------

    @property
    def itag_prefix(self) -> str:
        """Prefix string for bond itags produced by the gauge-fixing sweeps.

        The left bond of site *i* is tagged `{prefix}{i % N:0M}` and the
        right bond is tagged `{prefix}{(i+1) % N:0M}`, where *M* is
        `max(2, len(str(N)))`. Changing the prefix only affects future
        canonicalization calls; existing itags are not retroactively renamed.
        """
        return self._itag_prefix

    @itag_prefix.setter
    def itag_prefix(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("itag_prefix must be a non-empty string")
        self._itag_prefix = value

    def _bond_itag(self, k: int) -> str:
        """Return the itag for bond index *k*, wrapped into `[0, N)`.

        The left bond of site *i* has index *i* and the right bond has index
        *i+1*; the modulo is what identifies bond `N` with bond `0` and so
        closes the ring.
        """
        N = self.N
        ndigits = max(2, len(str(N)))
        return f'{self._itag_prefix}{k % N:0{ndigits}d}'

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tensors)

    def __getitem__(self, i: int) -> Tensor:
        return self._tensors[i]

    def __setitem__(self, i: int, t: Tensor) -> None:
        self._tensors[i] = t

    def __repr__(self) -> str:
        dims = '-'.join(str(d) for d in self.bond_dims)
        return (
            f"{type(self).__name__}(N={self.N}, "
            f"center={self.center}, bonds=[{dims}])"
        )

    @property
    def center(self) -> Optional[int]:
        """Orthogonality center site index, or `None` if unspecified.

        Read-only. Set internally by the gauge-fixing routines.
        """
        return self._center

    @property
    def N(self) -> int:
        """Number of sites in the unit cell."""
        return len(self._tensors)

    @property
    def L(self) -> int:
        """Alias for `N`, for symmetry with the finite `Network` family."""
        return self.N

    @property
    def bond_dims(self) -> List[int]:
        """Bond dimension at each bond of the unit cell (length `N`).

        Entry *i* is the dimension of the right bond of site *i*. Entry
        `N-1` is the wrap-around bond, which is also the left bond of site
        `0` — note the length is `N`, not `N-1` as in the finite case.
        """
        return [t.indices[1].dim for t in self._tensors]

    @property
    def bond_states(self) -> List[int]:
        """Number of physical states at each bond of the unit cell (length `N`).

        Entry *i* is the state count of the right bond of site *i*. For
        Abelian symmetry groups this is identical to `bond_dims`; for generic
        groups (e.g. SU(2)) it is larger due to multiplet degeneracy: each
        multiplet of spin *j* contributes *2j+1* states.
        """
        return [t.indices[1].num_states for t in self._tensors]

    @property
    def phys_dims(self) -> List[int]:
        """Physical dimension at each site of the unit cell (axis 2, the ket index)."""
        return [t.indices[2].dim for t in self._tensors]

    # ------------------------------------------------------------------
    # Device and dtype
    # ------------------------------------------------------------------

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Move every site tensor to `device` and cast it to `dtype`, in place.

        Both arguments are optional and an omitted one leaves that aspect of
        the tensors alone. Casting the blocks also casts the Bridge
        intertwiner weights, which is what a generic symmetry group needs:
        the two must agree for the tensor to contract.

        Parameters
        ----------
        device:
            Target device. If `None`, the current device is kept.
        dtype:
            Target block dtype. If `None`, the current dtype is kept.
        """
        for i, t in enumerate(self._tensors):
            self._tensors[i] = t.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> dict:
        """Convert the unit cell to a plain dict compatible with `torch.save`.

        The returned dict contains only Python primitives and `torch.Tensor`
        values, making it directly usable with
        `torch.save` / `torch.load(..., weights_only=True)`.

        Returns
        -------
        dict
            Serialized representation with keys `"version"`, `"class"`,
            `"bc"`, `"center"`, `"itag_prefix"`, and `"tensors"`.
        """
        return {
            "version": 1,
            "class": type(self).__name__,
            # Always 'IBC'; recorded so a payload is self-describing when
            # inspected alongside a finite one.
            "bc": self.bc,
            "center": self._center,
            "itag_prefix": self._itag_prefix,
            "tensors": [_serialize_tensor(t) for t in self._tensors],
        }

    @staticmethod
    def deserialize(data: dict, device: str = "cpu") -> iNetwork:
        """Reconstruct an `iNetwork` (or subclass) from a dict from `serialize`.

        Dispatches to the correct subclass (`iNetwork`, `iMPS`, or `iMPO`)
        based on the `"class"` key in `data`. The registry is separate from
        the finite one, so a finite payload cannot be loaded as an infinite
        network or the other way around.

        Parameters
        ----------
        data:
            Dict previously produced by `serialize`.
        device:
            Device to place all tensor blocks on. Defaults to `"cpu"`.

        Returns
        -------
        iNetwork
            Reconstructed network on *device*. The concrete type matches what
            was serialized.

        Raises
        ------
        ValueError
            If `data["version"]` is not `1`, or if `data["class"]` names an
            unknown class.
        """
        version = data.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported serialization version: {version!r}")

        cls_name = data["class"]
        # Registry is built inline to avoid a forward-reference problem — iMPS
        # and iMPO are defined after iNetwork in this module.
        _registry = {"iNetwork": iNetwork, "iMPS": iMPS, "iMPO": iMPO}
        if cls_name not in _registry:
            raise ValueError(
                f"Unknown infinite network class in serialized data: {cls_name!r}"
            )
        cls = _registry[cls_name]

        tensors = [_deserialize_tensor(d, device=device) for d in data["tensors"]]
        return cls(tensors, center=data["center"], itag_prefix=data["itag_prefix"])


class iMPS(iNetwork):  # noqa: N801 — physics convention
    """Infinite matrix product state.

    Each site tensor has axes `(left_bond, right_bond, physical)` where the
    physical axis at site *i* carries itag `s{i:02d}`. The right bond of the
    last site is the left bond of the first, so the state describes an
    infinite chain built by repeating the unit cell.
    """

    def __init__(
        self,
        tensors: List[Tensor],
        center: Optional[int] = None,
        itag_prefix: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        tensors:
            Unit cell site tensors, each with exactly 3 axes
            `(left_bond, right_bond, physical)`. The physical axis at site
            *i* must carry itag `s{i:02d}`.
        center:
            Orthogonality center site index, or `None` if unspecified.
        itag_prefix:
            Bond itag prefix string. Defaults to `'A'` when `None`.

        Raises
        ------
        ValueError
            If any tensor has the wrong number of axes, an incorrect physical
            itag, or the unit cell does not close into a consistent ring.
        """
        super().__init__(tensors, center, itag_prefix if itag_prefix is not None else 'A')

    def _validate(self) -> None:
        """Extend the ring validation with iMPS-specific axis and itag checks."""
        for i, t in enumerate(self._tensors):
            if len(t.indices) != 3:
                raise ValueError(
                    f"iMPS tensor at site {i} must have 3 axes, "
                    f"got {len(t.indices)}"
                )
            expected = f's{i:02d}'
            if t.itags[2] != expected:
                raise ValueError(
                    f"iMPS tensor at site {i}: physical axis (2) must have "
                    f"itag '{expected}', got '{t.itags[2]}'"
                )
        super()._validate()

    def __repr__(self) -> str:
        from alice.network.display import infinite_summary
        return infinite_summary(self)


class iMPO(iNetwork):  # noqa: N801 — physics convention
    """Infinite matrix product operator.

    Each site tensor has axes `(left_bond, right_bond, phys_in, phys_out)`
    where `phys_in` (ket, axis 2) and `phys_out` (bra, axis 3) both carry
    itag `s{i:02d}` at site *i*, differentiated by opposite directions.

    A Hamiltonian iMPO built by `build_ihamiltonian` additionally carries
    the upper-triangular automaton structure: bond sector `0` is the "start"
    channel and the last bond sector is the "finish" channel, both acting as
    the identity on the diagonal. The environment solver relies on that
    structure, which is why an iMPO is never canonicalized or compressed.
    """

    def __init__(
        self,
        tensors: List[Tensor],
        center: Optional[int] = None,
        itag_prefix: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        tensors:
            Unit cell site tensors, each with exactly 4 axes
            `(left_bond, right_bond, phys_in, phys_out)`. Both physical axes
            at site *i* must carry itag `s{i:02d}` with opposite directions.
        center:
            Orthogonality center site index, or `None` if unspecified.
        itag_prefix:
            Bond itag prefix string. Defaults to `'W'` when `None`.

        Raises
        ------
        ValueError
            If any tensor has the wrong axis count, incorrect physical itags,
            physical axes with the same direction, or the unit cell does not
            close into a consistent ring.
        """
        super().__init__(tensors, center, itag_prefix if itag_prefix is not None else 'W')

    def _validate(self) -> None:
        """Extend the ring validation with iMPO-specific axis and itag checks."""
        for i, t in enumerate(self._tensors):
            if len(t.indices) != 4:
                raise ValueError(
                    f"iMPO tensor at site {i} must have 4 axes, "
                    f"got {len(t.indices)}"
                )
            expected = f's{i:02d}'
            if t.itags[2] != expected or t.itags[3] != expected:
                raise ValueError(
                    f"iMPO tensor at site {i}: both physical axes (2, 3) must "
                    f"carry itag '{expected}', "
                    f"got ('{t.itags[2]}', '{t.itags[3]}')"
                )
            if t.indices[2].direction == t.indices[3].direction:
                raise ValueError(
                    f"iMPO tensor at site {i}: phys_in (axis 2) and phys_out "
                    f"(axis 3) must have opposite directions"
                )
        super()._validate()

    def __repr__(self) -> str:
        from alice.network.display import infinite_summary
        return infinite_summary(self)
