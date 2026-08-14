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


"""MPS, MPO, and base tensor network class for 1D chains."""

from __future__ import annotations

from typing import List, Optional

import math

from nicole import Tensor, contract, einsum, decomp
from nicole import serialize as _serialize_tensor, deserialize as _deserialize_tensor

# Sentinel used to distinguish "caller passed nothing" from "caller passed None".
# canonical() uses this so that trunc=None retains its natural decomp meaning
# (no truncation), while an omitted argument triggers the max-bond-dim default.
_DEFAULT_TRUNC = object()
# Default itag prefix for `Network` class.
_DEFAULT_ITAG_PREFIX = '_init_'


class Network:
    """Base class for 1D tensor network chains.

    Stores a list of site tensors and provides uniform iteration and axis
    conventions. Subclasses typically add a `center` attribute together
    with `canonical()` and `norm()`.

    Axis layout assumed by this base class (shared by MPS and MPO):

    - Axis 0: left bond
    - Axis 1: right bond
    - Axis 2: physical index (ket / phys_in)
    - Axis 3 (MPO only): conjugate physical index (bra / phys_out)

    Bond itag convention (set by `canonical()`):

    The left bond of site *i* carries itag `{prefix}{i:0N}` and the right bond
    carries `{prefix}{i+1:0N}`, where `prefix` is `itag_prefix` (default
    `'_init_'` for `Network`, `'A'` for `MPS`, `'W'` for `MPO`) and *N* is
    `max(2, len(str(L)))` — at least 2 digits, growing automatically for large
    chains. The bond between sites *i* and *i+1* is therefore
    `{prefix}{i+1:0N}`.

    Physical itag convention: site *i* physical axis carries itag `s{i:02d}`.
    """

    def __init__(
        self,
        tensors: List[Tensor],
        bc: str = 'OBC',
        center: Optional[int] = None,
        itag_prefix: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        tensors:
            Non-empty list of site tensors.
        bc:
            Boundary condition: `'OBC'` (default) or `'PBC'`.
        center:
            Orthogonality center site index, or `None` if the canonical form
            is unspecified.
        itag_prefix:
            Bond itag prefix string. Defaults to `'_init_'` when `None`.
            Subclasses override this default (`'A'` for `MPS`, `'W'` for
            `MPO`), so supply this argument only when a non-default prefix is
            explicitly required.

        Raises
        ------
        ValueError
            If `bc` is not `'OBC'` or `'PBC'`, or if `tensors` is empty.
        """
        bc = bc.upper()
        if bc not in ('OBC', 'PBC'):
            raise ValueError(f"bc must be 'OBC' or 'PBC', got '{bc}'")
        if not tensors:
            raise ValueError("tensors must be a non-empty list")
        self._tensors: List[Tensor] = list(tensors)
        self.bc: str = bc
        self._center: Optional[int] = center
        self._itag_prefix: str = itag_prefix if itag_prefix is not None else _DEFAULT_ITAG_PREFIX
        self._validate()

    def _validate(self) -> None:
        """Check that adjacent bond indices are consistent.

        For each pair of neighbouring sites, verifies that the right bond of
        site *i* and the left bond of site *i+1*:

        - share the same itag,
        - have opposite directions (so they can be contracted), and
        - agree on the dimension of every charge sector present in both.

        Subclasses should override this method to add further checks (e.g.
        axis count, physical itags) and call `super()._validate()`.

        Raises
        ------
        ValueError
            If any bond consistency check fails.
        """
        for i in range(self.L - 1):
            r = self._tensors[i].indices[1]
            l = self._tensors[i + 1].indices[0]
            r_itag = self._tensors[i].itags[1]
            l_itag = self._tensors[i + 1].itags[0]

            if r_itag != l_itag:
                raise ValueError(
                    f"Bond between sites {i} and {i + 1}: itag mismatch "
                    f"('{r_itag}' on site {i}'s right bond vs "
                    f"'{l_itag}' on site {i + 1}'s left bond)"
                )
            if r.direction == l.direction:
                raise ValueError(
                    f"Bond between sites {i} and {i + 1}: "
                    f"right index of site {i} and left index of site {i + 1} "
                    f"must have opposite directions"
                )
            r_dims = {s.charge: s.dim for s in r.sectors}
            l_dims = {s.charge: s.dim for s in l.sectors}
            for charge, dim in r_dims.items():
                if charge in l_dims and l_dims[charge] != dim:
                    raise ValueError(
                        f"Bond between sites {i} and {i + 1}: "
                        f"sector charge {charge} has dimension {dim} on site "
                        f"{i}'s right bond but {l_dims[charge]} on site "
                        f"{i + 1}'s left bond"
                    )

    # ------------------------------------------------------------------
    # itag_prefix
    # ------------------------------------------------------------------

    @property
    def itag_prefix(self) -> str:
        """Prefix string for bond itags produced by `canonical()`.

        The left bond of site *i* is tagged `{prefix}{i:0N}` and the right
        bond is tagged `{prefix}{i+1:0N}`, where *N* is
        `max(2, len(str(L)))`. Changing the prefix only affects future
        canonicalization calls; existing itags on the stored tensors are not
        retroactively renamed.
        """
        return self._itag_prefix

    @itag_prefix.setter
    def itag_prefix(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("itag_prefix must be a non-empty string")
        self._itag_prefix = value

    def _bond_itag(self, k: int) -> str:
        """Return the itag for bond index *k*.

        The left bond of site *i* has index *i* and the right bond has index
        *i+1*. The number of digits is `max(2, len(str(L)))` so that itags
        sort correctly for any chain length.
        """
        ndigits = max(2, len(str(self.L)))
        return f'{self._itag_prefix}{k:0{ndigits}d}'

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
    def center(self) -> Optional[int]:
        """Orthogonality center site index, or `None` if unspecified.

        Read-only. Set internally by `canonical()` and `redistribute_norm()`.
        """
        return self._center

    @property
    def L(self) -> int:
        """Number of sites."""
        return len(self)

    @property
    def bond_dims(self) -> List[int]:
        """Bond dimension at each internal bond (length `L - 1`).

        Entry *i* is the dimension of the right bond of site *i*, which equals
        the left bond of site *i+1*.
        """
        return [self._tensors[i].indices[1].dim for i in range(self.L - 1)]

    @property
    def bond_states(self) -> List[int]:
        """Number of physical states at each internal bond (length `L - 1`).

        Entry *i* is the state count of the right bond of site *i*, which equals
        the left bond of site *i+1*. For Abelian symmetry groups this is identical
        to `bond_dims`; for non-Abelian groups (e.g. SU(2)) it is larger due to
        multiplet degeneracy: each multiplet of spin *j* contributes *2j+1* states.
        """
        return [self._tensors[i].indices[1].num_states for i in range(self.L - 1)]

    @property
    def phys_dims(self) -> List[int]:
        """Physical dimension at each site (axis 2, the ket index)."""
        return [self._tensors[i].indices[2].dim for i in range(self.L)]

    # ------------------------------------------------------------------
    #  Canonicalization internals — shared by MPS and MPO
    # ------------------------------------------------------------------

    def _left_canon_site(self, i: int, trunc: Optional[dict] = None) -> None:
        """Left-canonicalize site *i*: LV decomp; absorb L into site *i+1*."""
        # col = right bond only → LV gives isometry A_this with axes (_bond_R, left, phys…)
        L, A_this = decomp(self._tensors[i], axes=1, flow='<<', mode='LV', trunc=trunc)

        # Swap new bond to axis 1: (_bond_R, left, phys…) → (left, _bond_R, phys…)
        perm = [1, 0] + list(range(2, len(A_this.indices)))
        A_this.permute(perm, in_place=True)
        # Right bond of site i has index i+1.
        tag = self._bond_itag(i + 1)
        A_this.retag(1, tag)
        self._tensors[i] = A_this

        # L has axes (old_right_of_i, '_bond_R').
        # Contract into site i+1 over old_right (axis 0 of L, axis 0 of A[i+1]).
        A_next = contract(L, self._tensors[i + 1], axes=(0, 0))
        # Result: ('_bond_R', right_i+1, phys_i+1, …) — correct axis order already.
        A_next.retag(0, tag)
        self._tensors[i + 1] = A_next

    def _right_canon_site(self, i: int, trunc: Optional[dict] = None) -> None:
        """Right-canonicalize site *i*: LQ decomp; absorb L into site *i-1*."""
        # row = left, cols = (right, phys…)  →  LQ produces L = U·S and isometry A_this
        L_fac, A_this = decomp(self._tensors[i], axes=0, flow='<<', mode='LV', trunc=trunc)

        # A_this already has axes ('_bond_R', right, phys…) — correct order.
        # Left bond of site i has index i.
        tag = self._bond_itag(i)
        A_this.retag(0, tag)
        self._tensors[i] = A_this

        # L_fac has axes (old_left_of_i, '_bond_R').
        # Contract into site i-1: axis 1 of A[i-1] matches axis 0 of L_fac.
        A_prev = contract(self._tensors[i - 1], L_fac, axes=(1, 0))
        # Result: (left_i-1, phys_i-1, …, '_bond_R') — bond at the last axis.
        # Move bond from the last position to axis 1.
        perm = [0, len(A_prev.indices) - 1] + list(range(1, len(A_prev.indices) - 1))
        A_prev.permute(perm, in_place=True)
        A_prev.retag(1, tag)
        self._tensors[i - 1] = A_prev

    # ------------------------------------------------------------------
    #  Public operations
    # ------------------------------------------------------------------

    def canonical(self, target: int, trunc=_DEFAULT_TRUNC) -> None:
        """Move the orthogonality center to site `target`.

        Uses QR for left-to-right steps and LQ for right-to-left steps. Bond
        itags are updated according to `itag_prefix`: the bond between sites
        *i* and *i+1* is always tagged `{itag_prefix}{i+1:0N}` regardless of
        sweep direction.

        If `center` is `None`, a full right-canonicalization sweep
        (sites `L-1` → 1) is performed first to establish `center = 0`,
        and then the center is moved to `target`.

        Parameters
        ----------
        target:
            Destination site for the orthogonality center.
        trunc:
            Truncation options forwarded to `decomp` at each QR/LQ step.
            When omitted, defaults to `{'nkeep': max(self.bond_dims)}` so
            that the sweep preserves the current maximum bond dimension.
            Pass `None` to disable truncation entirely (consistent with the
            `decomp` convention).

        Raises
        ------
        IndexError
            If `target` is out of range `[0, L)`.
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
            self._center = 0
            start = 0

        if start < target:
            # Sweep left → right: left-canonicalize each site up to target-1.
            for i in range(start, target):
                self._left_canon_site(i, trunc=trunc)
        elif start > target:
            # Sweep right → left: right-canonicalize each site down to target+1.
            for i in range(start, target, -1):
                self._right_canon_site(i, trunc=trunc)

        self._center = target

    def norm(self) -> float:
        """Compute the network norm without modifying the network.

        When `center` is set, the network is in mixed canonical form and the
        norm equals the Frobenius norm of the center tensor, which is returned
        directly. If `center` is `None`, the norm is computed by contracting
        ⟨ψ|ψ⟩ site by site from left to right.

        Returns
        -------
        float
            The norm of the network.
        """
        if self.center is not None:
            return self._tensors[self.center].norm()

        # Compute ||ψ|| via a left-to-right transfer-matrix contraction of ⟨ψ|ψ⟩.
        # One einsum letter per physical axis, starting at 'e'.
        phys = ''.join(chr(ord('e') + k) for k in range(len(self._tensors[0].indices) - 2))

        # First site: contract over shared left bond (a) and physical axes.
        # env[c, d] = Σ_{a, s...} t*[a, c, s...] · t[a, d, s...]
        t0 = self._tensors[0]
        env = einsum(f'ac{phys},ad{phys}->cd', t0.conj(), t0)

        # Remaining sites: absorb env and contract over left bonds (a, b) and physical axes.
        # env_new[c, d] = Σ_{a, b, s...} env[a, b] · t*[a, c, s...] · t[b, d, s...]
        for t in self._tensors[1:]:
            env = einsum(f'ab,ac{phys},bd{phys}->cd', env, t.conj(), t)

        # For OBC, env is a 1×1 tensor containing ⟨ψ|ψ⟩ = ||ψ||².
        # env.norm() = ⟨ψ|ψ⟩, so ||ψ|| = sqrt(env.norm()).
        return math.sqrt(env.norm())

    def normalize(self) -> None:
        """Normalize the network in-place by dividing the center tensor by its norm.

        The network must be in canonical form (i.e. `center` must not be
        `None`) so that the full norm is concentrated in a single tensor.
        After this call the center tensor has unit Frobenius norm.

        Raises
        ------
        ValueError
            If `center` is `None`. Call `canonical()` first to bring the
            network into mixed-canonical form before normalizing.
        ValueError
            If the norm is numerically zero (cannot divide by zero).
        """
        if self.center is None:
            raise ValueError(
                "normalize() requires a canonical form; call canonical() first"
            )
        # Divide the tensor at orthogonality center by its norm
        n = self._tensors[self.center].norm()
        if math.isclose(n, 0.0, abs_tol=1e-15):
            raise ValueError("cannot normalize: network norm is numerically zero")
        self._tensors[self.center] = self._tensors[self.center] * (1.0 / n)

    # ------------------------------------------------------------------
    #  Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> dict:
        """Convert the network to a plain dict compatible with `torch.save`.

        The returned dict contains only Python primitives and `torch.Tensor`
        values, making it directly usable with
        `torch.save` / `torch.load(..., weights_only=True)`.

        Each site tensor is serialized using `nicole.serialize`, preserving
        symmetry structure, index metadata, and block data.

        Returns
        -------
        dict
            Serialized representation with keys `"version"`, `"class"`,
            `"bc"`, `"center"`, `"itag_prefix"`, and `"tensors"`.

        Examples
        --------
        >>> payload = mps.serialize()
        >>> torch.save(payload, "state.mps")
        """
        return {
            "version": 1,
            "class": type(self).__name__,
            "bc": self.bc,
            "center": self._center,
            "itag_prefix": self._itag_prefix,
            "tensors": [_serialize_tensor(t) for t in self._tensors],
        }

    @staticmethod
    def deserialize(data: dict, device: str = "cpu") -> Network:
        """Reconstruct a `Network` (or subclass) from a dict produced by `serialize`.

        Dispatches to the correct subclass (`Network`, `MPS`, or `MPO`) based
        on the `"class"` key in `data`, then restores all metadata including
        `itag_prefix`.

        Parameters
        ----------
        data:
            Dict previously produced by `serialize`.
        device:
            Device to place all tensor blocks on. Defaults to `"cpu"`.

        Returns
        -------
        Network
            Reconstructed network on *device*. The concrete type matches what
            was serialized (`Network`, `MPS`, or `MPO`).

        Raises
        ------
        ValueError
            If `data["version"]` is not `1`, or if `data["class"]` names an
            unknown class.

        Examples
        --------
        >>> payload = torch.load("state.mps", weights_only=True)
        >>> mps = Network.deserialize(payload, device="cpu")
        """
        version = data.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported serialization version: {version!r}")

        cls_name = data["class"]
        # Registry is built inline to avoid a forward-reference problem — MPS
        # and MPO are defined after Network in this module.
        _registry = {"Network": Network, "MPS": MPS, "MPO": MPO}
        if cls_name not in _registry:
            raise ValueError(f"Unknown network class in serialized data: {cls_name!r}")
        cls = _registry[cls_name]

        tensors = [_deserialize_tensor(d, device=device) for d in data["tensors"]]
        return cls(tensors, bc=data["bc"], center=data["center"], itag_prefix=data["itag_prefix"])



class MPS(Network):
    """Matrix product state.

    Each site tensor has axes `(left_bond, right_bond, physical)` where
    the physical axis at site *i* carries itag `s{i:02d}`.
    """

    def __init__(
        self,
        tensors: List[Tensor],
        bc: str = 'OBC',
        center: Optional[int] = None,
        itag_prefix: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        tensors:
            List of site tensors, each with exactly 3 axes
            `(left_bond, right_bond, physical)`. The physical axis at site
            *i* must carry itag `s{i:02d}`.
        bc:
            Boundary condition: `'OBC'` (default) or `'PBC'`.
        center:
            Orthogonality center site index, or `None` if unspecified.
        itag_prefix:
            Bond itag prefix string. Defaults to `'A'` when `None`.

        Raises
        ------
        ValueError
            If any tensor has the wrong number of axes, an incorrect physical
            itag, or adjacent bond indices are inconsistent.
        """
        super().__init__(tensors, bc, center)
        self._itag_prefix = itag_prefix if itag_prefix is not None else 'A'

    def _validate(self) -> None:
        """Extend base validation with MPS-specific axis and itag checks."""
        for i, t in enumerate(self._tensors):
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
        super()._validate()

    def __repr__(self) -> str:
        from alice.network.display import network_summary
        return network_summary(self)


class MPO(Network):
    """Matrix product operator.

    Each site tensor has axes
    `(left_bond, right_bond, phys_in, phys_out)` where:

    - `phys_in` (ket, axis 2) and `phys_out` (bra, axis 3) both carry
      itag `s{i:02d}` at site *i*, differentiated by opposite directions.
    - Contracting an MPS ket (`s{i:02d}`, IN) against this MPO is
      automatic: the MPO bra (`s{i:02d}`, OUT) pairs with the MPS ket.
    """

    def __init__(
        self,
        tensors: List[Tensor],
        bc: str = 'OBC',
        center: Optional[int] = None,
        itag_prefix: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        tensors:
            List of site tensors, each with exactly 4 axes
            `(left_bond, right_bond, phys_in, phys_out)`. Both physical
            axes at site *i* must carry itag `s{i:02d}` with opposite
            directions.
        bc:
            Boundary condition: `'OBC'` (default) or `'PBC'`.
        center:
            Orthogonality center site index, or `None` if unspecified.
        itag_prefix:
            Bond itag prefix string. Defaults to `'W'` when `None`.

        Raises
        ------
        ValueError
            If any tensor has the wrong axis count, incorrect physical itags,
            physical axes with the same direction, or adjacent bond indices
            are inconsistent.
        """
        super().__init__(tensors, bc, center)
        self._itag_prefix = itag_prefix if itag_prefix is not None else 'W'

    def _validate(self) -> None:
        """Extend base validation with MPO-specific axis and itag checks."""
        for i, t in enumerate(self._tensors):
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
        super()._validate()

    def __repr__(self) -> str:
        from alice.network.display import network_summary
        return network_summary(self)

    def compact(self, trunc: Optional[dict] = None) -> None:
        """Compress the MPO bond dimensions in-place with norm preservation.

        Performs a two-sweep canonicalization:

        1. **Left-to-right** sweep without truncation, moving the orthogonality
           center to the rightmost site. This concentrates the full operator
           norm into the center tensor.
        2. **Normalize** the center tensor to unit Frobenius norm, temporarily
           factoring out the overall scale.
        3. **Right-to-left** sweep with SVD truncation (controlled by `trunc`),
           moving the center to site 0. Because the environment is unit-normed,
           the truncation threshold is applied on a consistent scale.
        4. **Restore** the overall scale into the new center tensor, then call
           `redistribute_norm()` to spread it evenly across all sites.

        Parameters
        ----------
        trunc:
            Truncation options forwarded to `canonical()` during the
            right-to-left compression sweep. Defaults to
            ``{'thresh': 1e-14}`` when `None`.
        """
        if trunc is None:
            trunc = {'thresh': 1e-14}

        # Left sweep — no truncation; center moves to L-1.
        self.canonical(self.L - 1, trunc=None)

        # Factor out the norm so the SVD threshold is on a unit scale.
        n = self.norm()
        self.normalize()

        # Right sweep — SVD truncation; center moves to 0.
        self.canonical(0, trunc=trunc)

        # Restore the overall scale then distribute it evenly.
        self._tensors[self.center] = self._tensors[self.center] * n
        self.redistribute_norm()

    def redistribute_norm(self) -> None:
        """Redistribute the MPO norm equally across all site tensors.

        Computes the total Frobenius norm N, multiplies every site tensor by
        `factor = N^(1/L)`, then divides the center tensor by N to remove the
        excess. The product of all scale factors is `factor^L / N = 1`, so
        the operator and its total norm are preserved. `center` is reset to
        `None` because the per-site rescaling breaks any prior canonical form.

        Raises
        ------
        ValueError
            If `center` is `None` (the MPO must be in canonical form so the
            norm is concentrated in a well-defined center tensor).
        ValueError
            If the MPO norm is numerically zero (e.g. due to norm decay).
        """
        if self.center is None:
            raise ValueError("redistribute_norm requires a canonical form")
        # Compute the total norm
        N = self.norm()
        if math.isclose(N, 0.0, abs_tol=1e-15):
            raise ValueError("cannot redistribute norm: MPO norm is numerically zero")

        factor = N ** (1.0 / self.L)
        # Scale each tensor by the factor
        for i in range(self.L):
            self._tensors[i] = self._tensors[i] * factor
        # Remove the excess factor^L = N from the center tensor.
        self._tensors[self.center] = self._tensors[self.center] * (1.0 / N)
        self._center = None
