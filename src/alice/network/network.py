# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Alice library.
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


"""MPS, MPO, and base tensor network class for 1D chains."""

from __future__ import annotations

from typing import List, Optional

from nicole import Tensor, contract
from nicole.decomp import decomp

# Sentinel used to distinguish "caller passed nothing" from "caller passed None".
# canonical() uses this so that trunc=None retains its natural decomp meaning
# (no truncation), while an omitted argument triggers the max-phys-dim default.
_DEFAULT_TRUNC = object()


class Network:
    """Base class for 1D tensor network chains.

    Stores a list of site tensors and provides uniform iteration and axis
    conventions.  Subclasses must define `_row_axes` and typically add a
    `center` attribute together with `canonical()` and `norm()`.

    Axis layout assumed by this base class (shared by MPS and MPO):

    - Axis 0: left bond
    - Axis 1: right bond
    - Axis 2: physical index (ket / phys_in)
    - Axis 3 (MPO only): conjugate physical index (bra / phys_out)

    Bond itag convention (set by `canonical()`):

    - `R{i:02d}`: bond between sites *i* and *i+1* produced by a left-to-right
      QR sweep (sites 0 … *i* are left-canonical).
    - `L{i:02d}`: same bond position produced by a right-to-left LQ sweep
      (sites *i* … L-1 are right-canonical).

    Physical itag convention: site *i* physical axis carries itag ``s{i:02d}``.
    """

    def __init__(
        self,
        tensors: List[Tensor],
        bc: str = 'OBC',
        center: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        tensors:
            Non-empty list of site tensors.
        bc:
            Boundary condition: ``'OBC'`` (default) or ``'PBC'``.
        center:
            Orthogonality center site index, or `None` if the canonical form
            is unspecified.

        Raises
        ------
        ValueError
            If `bc` is not ``'OBC'`` or ``'PBC'``, or if `tensors` is empty.
        """
        bc = bc.upper()
        if bc not in ('OBC', 'PBC'):
            raise ValueError(f"bc must be 'OBC' or 'PBC', got '{bc}'")
        if not tensors:
            raise ValueError("tensors must be a non-empty list")
        self._tensors: List[Tensor] = list(tensors)
        self.bc: str = bc
        self.center: Optional[int] = center

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
        dims = '-'.join(str(d) for d in self.bond_dims) if self.L > 1 else '(none)'
        return (
            f"{type(self).__name__}(L={self.L}, bc={self.bc!r}, "
            f"center={self.center}, bonds=[{dims}])"
        )

    @property
    def L(self) -> int:
        """Number of sites."""
        return len(self)

    @property
    def bond_dims(self) -> List[int]:
        """Bond dimension at each internal bond (length ``L - 1``).

        Entry *i* is the dimension of the right bond of site *i*, which equals
        the left bond of site *i + 1*.
        """
        return [self._tensors[i].indices[1].dim for i in range(self.L - 1)]

    @property
    def phys_dims(self) -> List[int]:
        """Physical dimension at each site (axis 2, the ket index)."""
        return [self._tensors[i].indices[2].dim for i in range(self.L)]

    # ------------------------------------------------------------------
    # Canonicalization internals — shared by MPS and MPO
    # ------------------------------------------------------------------

    @property
    def _row_axes(self) -> List[int]:
        """Axes that form the row block in the QR/LQ decomposition.

        Must be overridden by subclasses.  For MPS this is ``[0, 2]``
        (left bond + physical); for MPO it is ``[0, 2, 3]``
        (left bond + phys_in + phys_out).

        Raises
        ------
        NotImplementedError
            When called directly on `Network`.
        """
        raise NotImplementedError(
            "_row_axes must be implemented by a concrete subclass (MPS or MPO)"
        )

    def _restore_perm(self) -> List[int]:
        """Permutation that moves the new decomp bond to axis 1.

        After merging ``_row_axes`` and decomposing, the unmerged *U* tensor
        (for QR) or the contracted previous-site tensor (for LQ) has shape
        ``(row[0], row[1], …, row[n-1], bond_new)``.  This permutation
        rearranges it to ``(row[0]=left, bond_new, row[1], …, row[n-1])``.
        """
        n = len(self._row_axes)
        return [0, n] + list(range(1, n))

    def _left_canon_site(self, i: int, trunc: Optional[dict] = None) -> None:
        """Left-canonicalize site *i*: QR decomp; absorb R into site *i+1*."""
        T = self._tensors[i]

        # rows = (left, phys…), col = right  →  QR produces isometry Q and R = S·Vh
        Q, R = decomp(T, axes=self._row_axes, flow='>>', mode='UR', trunc=trunc)

        # Q has axes (left, phys…, '_bond_L') after unmerging.
        # Move the new bond from the last position to axis 1.
        Q = Q.permute(self._restore_perm())
        Q.retag(1, f'R{i:02d}')
        self._tensors[i] = Q

        # R has axes ('_bond_L', old_right_of_i).
        # Contract into site i+1: axis 1 of R matches axis 0 of A[i+1] (same itag,
        # opposite directions).
        A_next = contract(R, self._tensors[i + 1], axes=(1, 0))
        # Result: ('_bond_L', right_i+1, phys_i+1, …) — correct axis order already.
        A_next.retag(0, f'R{i:02d}')
        self._tensors[i + 1] = A_next

    def _right_canon_site(self, i: int, trunc: Optional[dict] = None) -> None:
        """Right-canonicalize site *i*: LQ decomp; absorb L into site *i-1*."""
        T = self._tensors[i]

        # row = left, cols = (right, phys…)  →  LQ produces L = U·S and isometry V
        L_fac, V = decomp(T, axes=0, flow='<<', mode='LV', trunc=trunc)

        # V already has axes ('_bond_R', right, phys…) — correct order.
        V.retag(0, f'L{i:02d}')
        self._tensors[i] = V

        # L_fac has axes (old_left_of_i, '_bond_R').
        # Contract into site i-1: axis 1 of A[i-1] matches axis 0 of L_fac.
        A_prev = contract(self._tensors[i - 1], L_fac, axes=(1, 0))
        # Result: (left_i-1, phys_i-1, …, '_bond_R') — bond at the last axis.
        # Move bond to axis 1.
        A_prev = A_prev.permute(self._restore_perm())
        A_prev.retag(1, f'L{i:02d}')
        self._tensors[i - 1] = A_prev

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    def canonical(self, target: int, trunc=_DEFAULT_TRUNC) -> None:
        """Move the orthogonality center to site `target`.

        Uses QR for left-to-right steps and LQ for right-to-left steps.  Bond
        itags are updated according to the ``R{i:02d}`` / ``L{i:02d}``
        convention: ``R`` prefix for bonds at or to the left of the center,
        ``L`` prefix for bonds to the right.

        If `center` is `None`, a full right-canonicalization sweep
        (sites ``L-1`` → 1) is performed first to establish ``center = 0``,
        and then the center is moved to `target`.

        Parameters
        ----------
        target:
            Destination site for the orthogonality center.
        trunc:
            Truncation options forwarded to `decomp` at each QR/LQ step.
            When omitted, defaults to ``{'nkeep': max(self.bond_dims)}`` so
            that the sweep preserves the current maximum bond dimension.
            Pass ``None`` to disable truncation entirely (consistent with the
            `decomp` convention).

        Raises
        ------
        IndexError
            If `target` is out of range ``[0, L)``.
        """
        if trunc is _DEFAULT_TRUNC:
            trunc = {'nkeep': max(self.bond_dims)} if self.bond_dims else None

        if target < 0:
            target += self.L
        if not (0 <= target < self.L):
            raise IndexError(f"target {target} out of range for L={self.L}")

        start = self.center
        if start is None:
            # Establish canonical form: right-canonicalize sites L-1 down to 1.
            for i in range(self.L - 1, 0, -1):
                self._right_canon_site(i, trunc=trunc)
            self.center = 0
            start = 0

        if start < target:
            # Sweep left → right: left-canonicalize each site up to target-1.
            for i in range(start, target):
                self._left_canon_site(i, trunc=trunc)
        elif start > target:
            # Sweep right → left: right-canonicalize each site down to target+1.
            for i in range(start, target, -1):
                self._right_canon_site(i, trunc=trunc)

        self.center = target

    def norm(self) -> float:
        """Compute the network norm.

        When `center` is set, the network is in mixed canonical form and the
        norm equals the Frobenius norm of the center tensor, which is returned
        directly.  If `center` is `None`, `canonical(0)` is called first
        (modifying the network in place).

        Returns
        -------
        float
            The norm of the network.
        """
        if self.center is None:
            self.canonical(0, trunc=None)
        return self._tensors[self.center].norm()


class MPS(Network):
    """Matrix product state.

    Each site tensor has axes ``(left_bond, right_bond, physical)`` where
    the physical axis at site *i* carries itag ``s{i:02d}``.
    """

    def __init__(
        self,
        tensors: List[Tensor],
        bc: str = 'OBC',
        center: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        tensors:
            List of site tensors, each with exactly 3 axes
            ``(left_bond, right_bond, physical)``.  The physical axis at site
            *i* must carry itag ``s{i:02d}``.
        bc:
            Boundary condition: ``'OBC'`` (default) or ``'PBC'``.
        center:
            Orthogonality center site index, or `None` if unspecified.

        Raises
        ------
        ValueError
            If any tensor has the wrong number of axes or an incorrect physical
            itag.
        """
        for i, t in enumerate(tensors):
            if len(t.indices) != 3:
                raise ValueError(
                    f"MPS tensor at site {i} must have 3 axes, "
                    f"got {len(t.indices)}"
                )
            expected = f's{i:02d}'
            if t.itags[2] != expected:
                raise ValueError(
                    f"MPS tensor at site {i}: physical axis (2) must have "
                    f"itag '{expected}', got '{t.itags[2]}'"
                )
        super().__init__(tensors, bc, center)

    @property
    def _row_axes(self) -> List[int]:
        # Left bond (0) and physical (2) are grouped as rows for QR.
        return [0, 2]


class MPO(Network):
    """Matrix product operator.

    Each site tensor has axes
    ``(left_bond, right_bond, phys_in, phys_out)`` where:

    - ``phys_in`` (ket, axis 2) and ``phys_out`` (bra, axis 3) both carry
      itag ``s{i:02d}`` at site *i*, differentiated by opposite directions.
    - Contracting an MPS ket (``s{i:02d}``, IN) against this MPO is
      automatic: the MPO bra (``s{i:02d}``, OUT) pairs with the MPS ket.
    """

    def __init__(
        self,
        tensors: List[Tensor],
        bc: str = 'OBC',
        center: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        tensors:
            List of site tensors, each with exactly 4 axes
            ``(left_bond, right_bond, phys_in, phys_out)``.  Both physical
            axes at site *i* must carry itag ``s{i:02d}`` with opposite
            directions.
        bc:
            Boundary condition: ``'OBC'`` (default) or ``'PBC'``.
        center:
            Orthogonality center site index, or `None` if unspecified.

        Raises
        ------
        ValueError
            If any tensor has the wrong axis count, incorrect physical itags,
            or physical axes with the same direction.
        """
        for i, t in enumerate(tensors):
            if len(t.indices) != 4:
                raise ValueError(
                    f"MPO tensor at site {i} must have 4 axes, "
                    f"got {len(t.indices)}"
                )
            expected = f's{i:02d}'
            if t.itags[2] != expected or t.itags[3] != expected:
                raise ValueError(
                    f"MPO tensor at site {i}: both physical axes (2, 3) must "
                    f"carry itag '{expected}', "
                    f"got ('{t.itags[2]}', '{t.itags[3]}')"
                )
            if t.indices[2].direction == t.indices[3].direction:
                raise ValueError(
                    f"MPO tensor at site {i}: phys_in (axis 2) and phys_out "
                    f"(axis 3) must have opposite directions"
                )
        super().__init__(tensors, bc, center)

    @property
    def _row_axes(self) -> List[int]:
        # Left bond (0), phys_in (2), and phys_out (3) are grouped as rows for QR.
        return [0, 2, 3]
