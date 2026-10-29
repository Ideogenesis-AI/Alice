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
tensors, which tiles the chain. Site `N` is site `0` of the next copy, so
the right bond of site `N-1` is the *same* index as the left bond of site
`0`: the cell has `N` bonds, and both of its ends carry a full virtual
index rather than a trivial dimension-1 boundary.

This is a tiling, not a closed loop. The chain is genuinely infinite, and
the index identification above says that translating by `N` sites changes
nothing. It does not bond two distant sites to each other, which is what
periodic boundary conditions do; the finite closed chain is `Network` with
`bc='PBC'`, and it is a different physical system solved by a different
algorithm.

That tiling is why this family stands apart from `Network`. An infinite
chain has no boundary vector to contract against and no open end to start
a sweep from, so counting bonds, fixing a gauge, taking a norm and
compacting an operator all need their own definitions here. Keeping the
two families unrelated also makes `isinstance(x, MPS)` `False` for an
infinite state, so the finite entry points reject one rather than silently
producing nonsense.

Axis layout (shared with the finite family):

- Axis 0: left bond
- Axis 1: right bond
- Axis 2: physical index (ket / phys_in)
- Axis 3 (iMPO only): conjugate physical index (bra / phys_out)

Bond itag convention: the left bond of site *i* carries itag
`{prefix}{i % N:0M}` and the right bond carries `{prefix}{(i+1) % N:0M}`,
where `prefix` is `itag_prefix` (`'A'` for `iMPS`, `'W'` for `iMPO`) and *M*
is `max(2, len(str(N)))`. The modulo is what makes the cell tile: at
`i = N-1` the right bond is tagged `{prefix}00`, matching the left bond of
site `0` in the next copy.

Physical itag convention: site *i* physical axis carries itag `s{i:02d}`,
exactly as in the finite family.

Canonical form
--------------

An infinite state has no ends to sweep from, so its canonical form is not
reached in one pass but is a fixed point of the gauge iteration: sweep the
cell with QR, carry the triangular factor on into the next copy, repeat
until it stops moving. `iMPS.canonical` does this from both sides and leaves the
unit cell uniformly right-orthonormal with one bond matrix per bond.
Writing `AR` for the stored tensors and `AL` for the left-orthonormal ones,
the three satisfy

    AL[i] C[i+1] = C[i] AR[i]

Each `C` is diagonal: the gauge iteration leaves it a general matrix, and a
final rotation of both canonical forms into the Schmidt basis turns it into
the Schmidt values themselves. That is worth the extra decomposition,
because it makes the stored form unique. A general `C` is only one of many
gauges describing the same state, so two states could not be compared bond
by bond and the entanglement spectrum could not be read off without a
further decomposition. Diagonalizing also gives `set_canonical` a normal form to put
its input into, so a state that has been through an update looks exactly
like one fresh from `canonical`.

All three are kept. `AL` cannot be recovered from `AR` and `C` by a fresh
QR sweep, because QR fixes its own basis on each bond and the result, while
left-orthonormal, no longer pairs with the stored `C`. Reconstructing it
through `C` instead would be least accurate exactly where the Schmidt
values are small, so keeping one extra tensor per site is cheaper.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import torch
from nicole import Tensor, decomp, einsum, identity
from nicole import serialize as _serialize_tensor, deserialize as _deserialize_tensor

logger = logging.getLogger(__name__)

# Default itag prefix for the bare `iNetwork` class. Subclasses override it.
_DEFAULT_ITAG_PREFIX = '_init_'

# itag worn by a bond between the QR step that creates it and the retag that
# gives it its unit-cell name.
_GAUGE_ITAG = '_gauge_'

# itags worn by the two halves of a bond while it is being rotated into the
# Schmidt basis. They differ so that the rotations cannot be contracted
# against each other by accident.
_SCHMIDT_ITAGS = ('_schmidt_L_', '_schmidt_R_')

# Guards the division that pulls a phase out of a diagonal entry.
_TINY = 1e-300


def _scale_axis(blk: torch.Tensor, scale: torch.Tensor, axis: int) -> None:
    """Multiply `blk` along `axis` by the per-index factors in `scale`."""
    shape = [1] * blk.dim()
    shape[axis] = -1
    blk.mul_(scale.reshape(shape))


def _qr_positive(
    t: Tensor, axes: List[int], itag: str, flow: str = '>>',
) -> Tuple[Tensor, Tensor]:
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
    flow:
        Direction convention for the new bond, forwarded to `decomp`.
        `'>>'` makes it incoming on `Q`, `'<<'` outgoing.

    Returns
    -------
    tuple
        `(Q, R)` with `Q` isometric over `axes` and the diagonal of `R` real
        and non-negative.
    """
    q, r = decomp(t, axes=axes, mode='QR', itag=itag, flow=flow)

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
            If `tensors` is empty, or if the unit cell does not tile
            consistently.
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
        """Check that the unit cell tiles consistently.

        For each of the `N` bonds — including the one joining site `N-1` to
        site `0` of the next copy — verifies that the right index of the
        left site and the left index of the right site:

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
            # Name the bond onto the next cell explicitly; a mismatch there
            # is the most likely failure and the least obvious to diagnose.
            where = f"between sites {i} and {j}" + (" (cell boundary)" if j < i else "")

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
        *i+1*; the modulo is what identifies bond `N` with bond `0`, which
        is what makes the cell tile.
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
        `N-1` is the bond onto the next copy of the cell, which is the same
        bond as the left bond of site `0` — note the length is `N`, not
        `N-1` as in the finite case.
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
        net = cls(tensors, center=data["center"], itag_prefix=data["itag_prefix"])
        # Only an iMPS writes these keys, and only once it has been gauge
        # fixed; restoring them saves a checkpointed run from re-converging.
        matrices = data.get("bond_matrices")
        if matrices is not None:
            net._bond_matrices = [
                _deserialize_tensor(d, device=device) for d in matrices
            ]
            net._left_isos = [
                _deserialize_tensor(d, device=device)
                for d in data["left_isometries"]
            ]
        return net


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
            itag, or the unit cell does not tile consistently.
        """
        self._bond_matrices: Optional[List[Tensor]] = None
        self._left_isos: Optional[List[Tensor]] = None
        super().__init__(tensors, center, itag_prefix if itag_prefix is not None else 'A')

    def __setitem__(self, i: int, t: Tensor) -> None:
        """Replace a site tensor, dropping the gauge it invalidates.

        The gauge goes with it, `center` included. A gauge is not
        bookkeeping: it carries the Schmidt spectrum, and keeping a stale
        one would let every later measurement quietly describe a state that
        no longer exists.
        """
        self._tensors[i] = t
        self._bond_matrices = None
        self._left_isos = None
        self._center = None

    def _validate(self) -> None:
        """Extend the cell validation with iMPS-specific axis and itag checks."""
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

    # ------------------------------------------------------------------
    # Gauge fixing
    # ------------------------------------------------------------------

    def _left_sweep(
        self, tensors: List[Tensor], gauge: Tensor,
    ) -> Tuple[List[Tensor], Tensor]:
        """Carry `gauge` left to right through the cell with QR at each site.

        Returns the left-orthonormal tensors and the triangular factor that
        falls off the right end, which the tiling puts back on bond 0.

        The `'<<'` flow sends the new bond out of `AL` rather than into it,
        which is what makes `AL` carry the same index directions as `AR` and
        the carried gauge an ordinary `(IN, OUT)` matrix. Without it the two
        canonical forms end up in dual bases, and `AC` can no longer be
        dropped into the chain in place of `AR`.
        """
        isometries: List[Tensor] = []
        for i, t in enumerate(tensors):
            m = einsum('al,lrs->ars', gauge, t)
            q, gauge = _qr_positive(m, [0, 2], _GAUGE_ITAG, flow='<<')
            # QR appends the new bond, giving (left, phys, bond); the MPS
            # layout wants it on axis 1.
            q.permute([0, 2, 1], in_place=True)
            q.retag([0, 1], [self._bond_itag(i), self._bond_itag(i + 1)])
            isometries.append(q)
            tag = self._bond_itag(i + 1)
            gauge.retag([0, 1], [tag, tag])
        return isometries, gauge

    def _right_sweep(
        self, tensors: List[Tensor], gauge: Tensor,
    ) -> Tuple[List[Tensor], Tensor]:
        """Carry `gauge` right to left through the cell with QR at each site."""
        isometries: List[Optional[Tensor]] = [None] * len(tensors)
        for i in range(len(tensors) - 1, -1, -1):
            m = einsum('lrs,br->lbs', tensors[i], gauge)
            q, gauge = _qr_positive(m, [1, 2], _GAUGE_ITAG)
            # Here the appended bond is the site's *left* bond, so it moves
            # to the front rather than to the middle.
            q.permute([2, 0, 1], in_place=True)
            q.retag([0, 1], [self._bond_itag(i), self._bond_itag(i + 1)])
            isometries[i] = q
            tag = self._bond_itag(i)
            gauge.retag([0, 1], [tag, tag])
        return isometries, gauge

    def _diagonalize_gauge(
        self,
        left_isos: List[Tensor],
        right_isos: List[Tensor],
        matrices: List[Tensor],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Rotate both canonical forms so that every bond matrix is diagonal.

        `C` cannot be built diagonal in the first place. It is the change
        of basis between the two canonical forms, `C[0] = L R^dag` from the
        two gauge fixed points, and those are found by independent
        iterations that know nothing of each other: `L` comes back upper
        triangular and `R^dag` lower, so the product is a general matrix
        whose off-diagonal entries are of order its diagonal ones. `C` is
        diagonal exactly when `AL` and `AR` are already Schmidt-aligned,
        which is what this rotation establishes rather than assumes.
        Writing `C[k] = U[k] S[k] Vh[k]` and setting

            AL'[i] = U[i]^dag AL[i] U[i+1]
            AR'[i] = Vh[i] AR[i] Vh[i+1]^dag
            C'[k]  = S[k]

        leaves `AL' C' = C' AR'` intact, since the rotations cancel in
        pairs, and keeps both forms orthonormal, since `U` and `Vh` are
        unitary. The bond matrices are then the Schmidt values themselves.

        Parameters
        ----------
        left_isos:
            Left-orthonormal tensors from the gauge iteration.
        right_isos:
            Right-orthonormal tensors from the gauge iteration.
        matrices:
            One bond matrix per bond, length `N`.

        Returns
        -------
        tuple
            `(left_isos, right_isos, matrices)` in the Schmidt basis.
        """
        N = self.N
        rot_l: List[Tensor] = []
        rot_r: List[Tensor] = []
        diagonals: List[Tensor] = []
        for k, c in enumerate(matrices):
            u, s, vh = decomp(c, axes=0, mode='SVD', flow='<<', itag=_SCHMIDT_ITAGS)
            tag = self._bond_itag(k)
            s.retag([0, 1], [tag, tag])
            rot_l.append(u)
            rot_r.append(vh)
            diagonals.append(s)

        rotated_l: List[Tensor] = []
        rotated_r: List[Tensor] = []
        for i in range(N):
            j = (i + 1) % N
            al = einsum('am,ars,rn->mns', rot_l[i].conj(), left_isos[i], rot_l[j])
            ar = einsum('nb,bcs,mc->nms', rot_r[i], right_isos[i], rot_r[j].conj())
            tags = [self._bond_itag(i), self._bond_itag(i + 1)]
            al.retag([0, 1], tags)
            ar.retag([0, 1], tags)
            rotated_l.append(al)
            rotated_r.append(ar)
        return rotated_l, rotated_r, diagonals

    def _gauge_fixed_point(
        self, to_left: bool, tol: float, maxiter: int,
    ) -> Tuple[List[Tensor], Tensor]:
        """Iterate one-directional sweeps until the gauge stops moving.

        Each sweep multiplies the gauge by the unit-cell transfer operator
        and re-orthogonalizes, so this is power iteration and converges as
        `|lambda_2 / lambda_1|` per sweep.

        Parameters
        ----------
        to_left:
            Sweep right to left (producing `AR`) rather than left to right.
        tol:
            Convergence threshold on the change in the gauge matrix.
        maxiter:
            Sweep budget before giving up.

        Returns
        -------
        tuple
            `(isometries, gauge)` at the fixed point.

        Raises
        ------
        ValueError
            If the iteration two-cycles or exhausts `maxiter`.
        """
        sweep = self._right_sweep if to_left else self._left_sweep
        seed = (
            self._tensors[-1].indices[1] if to_left
            else self._tensors[0].indices[0]
        )
        gauge = identity(seed)
        tag = self._bond_itag(0)
        gauge.retag([0, 1], [tag, tag])

        side = 'right' if to_left else 'left'
        isometries: List[Tensor] = []
        previous = older = None
        for step in range(1, maxiter + 1):
            isometries, gauge = sweep(self._tensors, gauge)
            norm = float(gauge.norm())
            # Sweeping can land the gauge in charge sectors the bond indices
            # do not carry, leaving nothing behind. The cell then describes
            # the zero state and there is no gauge to find.
            if norm == 0.0:
                raise ValueError(
                    f"{side} gauge fixing collapsed to zero on sweep {step}: "
                    f"the unit cell has no amplitude. Its bond charges "
                    f"cannot be reached from each other by the physical "
                    f"charges over N={self.N} sites."
                )
            gauge = gauge * (1.0 / norm)

            if previous is not None and float((gauge - previous).norm()) < tol:
                logger.debug("%s gauge converged after %d sweeps", side, step)
                return isometries, gauge
            # A two-cycle is not slow convergence, it is a mismatch between
            # the cell and the state, so say so rather than spin to maxiter.
            if older is not None and float((gauge - older).norm()) < tol:
                raise ValueError(
                    f"{side} gauge fixing fell into a two-cycle after {step} "
                    f"sweeps: the unit-cell transfer operator has two "
                    f"dominant eigenvalues of opposite sign. That happens "
                    f"when the physical charges split the bond charges into "
                    f"two classes that N={self.N} sites map into each other "
                    f"rather than onto themselves, so the cell cannot host a "
                    f"translation-invariant state. Double the unit cell, or "
                    f"give each bond only the charges reachable at that "
                    f"position."
                )
            older, previous = previous, gauge

        residual = float((previous - older).norm())
        raise ValueError(
            f"{side} gauge fixing did not converge in {maxiter} sweeps "
            f"(residual {residual:.3e}, tolerance {tol:.3e}). Power "
            f"iteration converges as the ratio of the two largest transfer "
            f"operator eigenvalues, so raise maxiter, loosen tol, or start "
            f"from a state closer to canonical."
        )

    def canonical(
        self, target: int = 0, tol: float = 1e-12, maxiter: int = 200,
    ) -> None:
        """Fix the gauge of the unit cell, in place.

        Leaves every stored tensor right-orthonormal (`AR`) and attaches one
        diagonal bond matrix `C` to each of the `N` bonds, with
        `AL[i] C[i+1] = C[i] AR[i]`. The state is normalized on the way: the
        transfer operator of the stored tensors has dominant eigenvalue `1`
        and each bond matrix has unit norm, so the Schmidt weights on every
        bond sum to one.

        The cell has no open end to start from, so one sweep is not enough:
        the gauge is found by iterating sweeps until the triangular factor
        carried into the next copy reproduces itself, after which both forms
        are rotated into the Schmidt basis. No truncation happens here, as
        the sweeps use QR, which keeps every bond exactly as wide as it was.

        Parameters
        ----------
        target:
            Site to record as the orthogonality center. Bond matrices are
            computed on every bond regardless, so this only chooses which
            site `center_tensor` defaults to describing. Negative values
            count from the end.
        tol:
            Convergence threshold on the change in the gauge matrix between
            successive sweeps.
        maxiter:
            Sweep budget for each of the two directions.

        Raises
        ------
        IndexError
            If `target` is out of range `[0, N)`.
        ValueError
            If either gauge iteration fails to converge, or if the two
            converged gauges disagree across the cell.
        """
        if target < 0:
            target += self.N
        if not (0 <= target < self.N):
            raise IndexError(f"target {target} out of range for N={self.N}")

        left_isos, left_gauge = self._gauge_fixed_point(False, tol, maxiter)
        right_isos, right_gauge = self._gauge_fixed_point(True, tol, maxiter)

        # The two gauges overlap on bond 0 to give C there; the defining
        # relation AL C = C AR then propagates it to the remaining bonds.
        bond = einsum('ao,bo->ab', left_gauge, right_gauge)
        matrices = [bond * (1.0 / float(bond.norm()))]
        for i in range(self.N):
            centered = einsum('ab,brs->ars', matrices[i], right_isos[i])
            matrices.append(einsum('lms,lbs->mb', left_isos[i].conj(), centered))

        # Propagating through a whole cell must return the bond matrix it
        # started from, since that is what translation invariance means. If
        # it does not, the two iterations converged to gauges that describe
        # different states and nothing below is valid.
        drift = float((matrices[self.N] - matrices[0]).norm())
        if drift > max(1e-8, 1e3 * tol):
            raise ValueError(
                f"gauge fixing is not translation invariant: propagating the "
                f"bond matrix through all {self.N} sites moved it by "
                f"{drift:.3e}. Tighten tol or raise maxiter."
            )

        left_isos, right_isos, matrices = self._diagonalize_gauge(
            left_isos, right_isos, matrices[:self.N],
        )

        self._tensors = right_isos
        self._left_isos = left_isos
        self._bond_matrices = matrices
        self._center = target
        self._validate()

    def set_canonical(
        self,
        left_isos: List[Tensor],
        right_isos: List[Tensor],
        bond_matrices: List[Tensor],
        target: int = 0,
    ) -> None:
        """Install a canonical form computed elsewhere, in place.

        The counterpart of `canonical` for callers that already know the
        answer. A variational update on the uniform manifold produces `AL`,
        `AR` and `C` directly, and re-deriving them with the gauge iteration
        would be both wasteful and fragile: the iteration assumes a single
        dominant transfer-operator eigenvalue, which a state in mid-update
        need not have.

        The three sets are not required to satisfy `AL[i] C[i+1] = C[i]
        AR[i]`. Away from a variational fixed point they do not, and the
        size of the mismatch is exactly the convergence measure such an
        update reports. They are rotated into the Schmidt basis on the way
        in, so the stored bond matrices are diagonal as they are after
        `canonical`.

        Parameters
        ----------
        left_isos:
            Left-orthonormal tensors, one per site.
        right_isos:
            Right-orthonormal tensors, one per site. These become the
            stored site tensors.
        bond_matrices:
            Bond matrices, one per bond, in the `(AL, AR)` basis pair.
        target:
            Site to record as the orthogonality center. Negative values
            count from the end.

        Raises
        ------
        IndexError
            If `target` is out of range `[0, N)`.
        ValueError
            If any of the three lists has the wrong length, or if the
            resulting unit cell does not tile consistently.
        """
        N = self.N
        if target < 0:
            target += N
        if not (0 <= target < N):
            raise IndexError(f"target {target} out of range for N={N}")
        for name, seq in (
            ('left_isos', left_isos),
            ('right_isos', right_isos),
            ('bond_matrices', bond_matrices),
        ):
            if len(seq) != N:
                raise ValueError(f"{name} must have {N} entries, got {len(seq)}")

        left_isos, right_isos, matrices = self._diagonalize_gauge(
            list(left_isos), list(right_isos), list(bond_matrices),
        )
        self._tensors = right_isos
        self._left_isos = left_isos
        self._bond_matrices = matrices
        self._center = target
        self._validate()

    # ------------------------------------------------------------------
    # Canonical-form accessors
    # ------------------------------------------------------------------

    def _require_canonical(self) -> None:
        """Raise unless `canonical` has run and nothing has invalidated it."""
        if self._bond_matrices is None:
            raise ValueError(
                "the unit cell is not gauge fixed; call canonical() first "
                "(replacing a site tensor discards the gauge)"
            )

    def bond_matrix(self, bond: int) -> Tensor:
        """Return the bond matrix `C` on `bond`, taken modulo `N`.

        Parameters
        ----------
        bond:
            Bond position. Bond `k` is the left bond of site `k`.

        Returns
        -------
        Tensor
            Diagonal 2-index tensor holding the Schmidt values, with its
            left axis in the `AL` basis and its right axis in the `AR`
            basis.
        """
        self._require_canonical()
        return self._bond_matrices[bond % self.N]

    def right_isometries(self) -> List[Tensor]:
        """Return the right-orthonormal tensors `AR`, one per site.

        These are the stored tensors, so this is a convenience alias that
        says which canonical form they are in.
        """
        self._require_canonical()
        return list(self._tensors)

    def left_isometries(self) -> List[Tensor]:
        """Return the left-orthonormal tensors `AL`, one per site.

        These come from the left gauge iteration and pair with the stored
        bond matrices as `AL[i] C[i+1] = C[i] AR[i]`. Their bonds carry the
        same itags as `AR`, but the two sets of bases are different.
        """
        self._require_canonical()
        return list(self._left_isos)

    def center_tensor(self, site: Optional[int] = None) -> Tensor:
        """Return the center tensor `AC[site] = C[site] AR[site]`.

        Parameters
        ----------
        site:
            Site index, taken modulo `N`. Defaults to `center`.

        Returns
        -------
        Tensor
            3-index tensor with the usual `(left, right, physical)` layout.
        """
        self._require_canonical()
        if site is None:
            site = self._center or 0
        site %= self.N
        return einsum(
            'ab,brs->ars', self._bond_matrices[site], self._tensors[site],
        )

    def schmidt_values(self, bond: int) -> List[float]:
        """Return the Schmidt values on `bond`, largest first.

        These are the diagonal of the bond matrix, so they are normalized
        to `sum(s**2) == 1`. For a generic symmetry group they
        are multiplet-resolved: a value in a spin-*j* sector stands for
        *2j+1* degenerate Schmidt states.

        Parameters
        ----------
        bond:
            Bond position, taken modulo `N`.

        Returns
        -------
        List[float]
            Singular values in descending order.
        """
        values = [
            float(v)
            for blk in self.bond_matrix(bond).data.values()
            for v in torch.diagonal(
                blk.reshape(blk.shape[0], blk.shape[1], -1)[:, :, 0]
            )
        ]
        return sorted(values, reverse=True)

    # ------------------------------------------------------------------
    # Device, dtype and serialization
    # ------------------------------------------------------------------

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Move the unit cell and its gauge, in place.

        Extends `iNetwork.to` so that a gauge-fixed state keeps `AL` and the
        bond matrices on the same device as the tensors they multiply.
        """
        super().to(device=device, dtype=dtype)
        if self._bond_matrices is not None:
            self._bond_matrices = [
                c.to(device=device, dtype=dtype) for c in self._bond_matrices
            ]
            self._left_isos = [
                a.to(device=device, dtype=dtype) for a in self._left_isos
            ]

    def serialize(self) -> dict:
        """Extend the base payload with the gauge, when the state has one."""
        data = super().serialize()
        if self._bond_matrices is None:
            data["bond_matrices"] = None
            data["left_isometries"] = None
        else:
            data["bond_matrices"] = [
                _serialize_tensor(c) for c in self._bond_matrices
            ]
            data["left_isometries"] = [
                _serialize_tensor(a) for a in self._left_isos
            ]
        return data


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
            tile consistently.
        """
        super().__init__(tensors, center, itag_prefix if itag_prefix is not None else 'W')

    def _validate(self) -> None:
        """Extend the cell validation with iMPO-specific axis and itag checks."""
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
