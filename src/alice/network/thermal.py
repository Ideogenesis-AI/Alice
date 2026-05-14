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


"""Thermal density matrix via Taylor expansion for 1D MPO chains.

This module provides `NormalMPO`, an `MPO` subclass that maintains its
internal tensors at unit Frobenius norm while tracking the true physical
scale as `_scale`, and `thermal_mpo`, which approximates
``ρ(β) = exp(−βH)`` via a truncated Taylor series.

The one-directional import chain is: `thermal.py` → `network.py`. No
circular dependency is introduced.
"""

from __future__ import annotations

import math
from math import factorial
from typing import List, Optional

from nicole import Direction, Index, Tensor
from nicole import capcup, einsum, identity, merge_axes, oplus, trace

from .network import MPO


# ---------------------------------------------------------------------------
# Identity MPO helper
# ---------------------------------------------------------------------------

def _identity_mpo(spc: Index, L: int) -> MPO:
    """Build an L-site identity MPO for physical space `spc`.

    Each site carries the identity operator on `spc` with trivial (dim-1)
    left and right bond indices, following the same itag convention as
    `build_hamiltonian`.

    Parameters
    ----------
    spc:
        Physical space index (direction IN, shape matches the local Hilbert
        space).
    L:
        Number of sites.

    Returns
    -------
    MPO
        L-site identity MPO with `center=None`.
    """
    ndigits = max(2, len(str(L)))

    # identity(spc) is a 2-leg tensor (phys_in=spc_IN, phys_out=spc_OUT).
    I = identity(spc)
    I4 = I.clone()
    # Insert trivial bond axes: left bond at position 0, right at position 1.
    I4.insert_index(0, direction=Direction.IN, itag='L')
    I4.insert_index(1, direction=Direction.OUT, itag='R')
    # I4 axes: (L_IN, R_OUT, phys_in, phys_out)

    tensors: List[Tensor] = []
    for i in range(L):
        t = I4.clone()
        t.retag(
            [0, 1, 2, 3],
            [f'W{i:0{ndigits}d}', f'W{i+1:0{ndigits}d}', f's{i:02d}', f's{i:02d}'],
        )
        tensors.append(t)

    return MPO(tensors, center=None)


# ---------------------------------------------------------------------------
# NormalMPO
# ---------------------------------------------------------------------------

class NormalMPO(MPO):
    """MPO with a separately tracked overall scale factor.

    Represents the physical operator as ``_scale × mpo``, where the internal
    MPO satisfies ``mpo.norm() ≈ 1`` after `compact()` or `from_mpo()`. All
    arithmetic operations (`+`, `@`, `*`) preserve this representation,
    updating `_scale` analytically without altering the unit-norm convention
    until `compact()` is called explicitly.

    Parameters
    ----------
    tensors:
        Non-empty list of site tensors, each with 4 axes
        ``(left_bond, right_bond, phys_in, phys_out)``.
    scale:
        Overall scale factor. Defaults to ``1.0``.
    bc:
        Boundary condition: ``'OBC'`` (default) or ``'PBC'``.
    center:
        Orthogonality center site index, or ``None`` if unspecified.
    """

    def __init__(
        self,
        tensors: List[Tensor],
        scale: float = 1.0,
        bc: str = 'OBC',
        center: Optional[int] = None,
    ) -> None:
        super().__init__(tensors, bc=bc, center=center)
        self._scale: float = float(scale)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_mpo(cls, mpo: MPO) -> NormalMPO:
        """Create a `NormalMPO` from a plain `MPO`.

        The input MPO is not modified. The returned object has
        ``mpo.norm() ≈ 1`` and ``scale == original_frobenius_norm``.

        Parameters
        ----------
        mpo:
            Source MPO. May have any canonical form.

        Returns
        -------
        NormalMPO
            Normalized copy with ``center=0``.

        Raises
        ------
        ValueError
            If the source MPO has numerically zero norm.
        """
        # Clone to avoid modifying the caller's MPO.
        tensors = [t.clone() for t in mpo]
        copy = MPO(tensors, bc=mpo.bc, center=mpo.center)
        copy.canonical(0)
        n = copy.norm()
        if math.isclose(n, 0.0, abs_tol=1e-15):
            raise ValueError("cannot create NormalMPO from a zero-norm MPO")
        copy.normalize()
        return cls(copy._tensors, scale=n, bc=mpo.bc, center=0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def scale(self) -> float:
        """Overall scale factor (read-only)."""
        return self._scale

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compact(self, trunc: Optional[dict] = None) -> None:
        """Compress bond dimensions in-place, keeping `_scale` consistent.

        Performs the same two-sweep canonicalization as `MPO.compact()`, but
        folds the extracted norm into `_scale` instead of redistributing it
        across the site tensors. After the call, the internal MPO is
        approximately unit-normed and `_scale` absorbs the physical
        magnitude.

        Bond arrows may be reoriented by the QR/SVD sweeps; this method
        restores the standard IN/OUT convention via `capcup` after the
        sweeps, so that subsequent `trace()`, `__add__`, and `__matmul__`
        calls work correctly.

        Parameters
        ----------
        trunc:
            Truncation options forwarded to `canonical()` during the
            right-to-left compression sweep. Defaults to
            ``{'thresh': 1e-14}``.
        """
        if trunc is None:
            trunc = {'thresh': 1e-14}

        # Left sweep — no truncation; center → L-1.
        self.canonical(self.L - 1, trunc=None)

        # Extract and remove overall scale so the SVD threshold is on a
        # unit scale.
        n = self.norm()
        self.normalize()

        # Right sweep — SVD truncation; center → 0.
        self.canonical(0, trunc=trunc)

        # Fold the extracted norm into _scale; tensors remain unit-normed.
        self._scale *= n

        # Restore the standard IN/OUT bond arrow convention. Canonical sweeps
        # (QR/LQ) may reorient bond arrows; capcup undoes this so that
        # subsequent oplus (in __add__) and contract (in trace) work correctly.
        for k in range(self.L - 1):
            capcup(self[k], 1, self[k + 1], 0)

        # The canonical form is no longer valid after capcup.
        self._center = None

    # ------------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------------

    def trace(self) -> float:
        r"""Compute the MPO trace :math:`\operatorname{Tr}[\rho]`.

        Performs a left-to-right transfer-matrix sweep. At each site the
        physical indices (phys_in and phys_out, sharing itag ``s{i:02d}``
        with opposite directions) are traced using Nicole's `trace` function,
        yielding a 2-leg bond tensor.         Adjacent bond tensors are chained with
        `einsum`.

        Returns
        -------
        float
            ``_scale × Tr[internal_mpo]``.
        """
        L = self.L
        # Site 0: partial trace over physical axes → 2-leg bond tensor.
        env = trace(self._tensors[0], axes=[(2, 3)])

        # Sites 1..L-1: absorb each site's partial trace into the environment.
        for i in range(1, L):
            w_tr = trace(self._tensors[i], axes=[(2, 3)])
            env = einsum('ab,bc->ac', env, w_tr)

        # env is a 2-leg 1×1 tensor (both bonds are the vacuum at OBC
        # boundaries). Extract the scalar, accounting for the Bridge weight
        # in non-Abelian groups.
        key, val = next(iter(env.data.items()))
        weight = 1.0 if env.intw is None else float(env.intw[key].weights[0, 0])
        return float(val.item().real) * weight * self._scale

    # ------------------------------------------------------------------
    # MPO-MPO product
    # ------------------------------------------------------------------

    def __matmul__(self, other: NormalMPO) -> NormalMPO:
        """Return the MPO product ``self @ other`` without canonicalization.

        For each site ``i``:

        1. Retag the bond axes of ``self[i]`` to ``'L1'``/``'R1'`` and
           ``other[i]`` to ``'L2'``/``'R2'`` (isolates bond structure from
           shared physical itags).
        2. Contract over the shared physical axis (``self``'s ``phys_out``
           × ``other``'s ``phys_in``) via
           ``einsum('abrs,cdsu->acbdru', W1, W2)``, producing a 6-axis
           tensor ``(L1, L2, R1, R2, phys_in, phys_out)``.
        3. Fuse right bonds first: ``merge_axes(C, [2, 3], 'R', OUT)``
           → axes ``(R, L1, L2, r, u)``.
        4. Fuse left bonds: ``merge_axes(C, [1, 2], 'L', IN)``
           → axes ``(L, R, phys_in, phys_out)``. *(Correct index order.)*
        5. Retag to standard MPO bond itags.

        The resulting bond dimensions are ``χ(self) × χ(other)`` before any
        `compact()` call.

        Parameters
        ----------
        other:
            Right operand, must have the same chain length and physical
            space as ``self``.

        Returns
        -------
        NormalMPO
            Raw product MPO with ``scale = self._scale * other._scale``.
        """
        L = self.L
        ndigits = max(2, len(str(L)))
        sites: List[Tensor] = []

        for i in range(L):
            W1 = self._tensors[i].clone()
            W2 = other._tensors[i].clone()

            # Temporarily retag bond axes to avoid itag collisions between
            # the two MPOs (both use W{i:02d} itags).
            W1.retag([0, 1], ['L1', 'R1'])
            W2.retag([0, 1], ['L2', 'R2'])

            # Contract shared physical axis: W1's phys_out (s{i:02d}, OUT)
            # pairs with W2's phys_in (s{i:02d}, IN) automatically.
            # Result C has axes (L1=a, L2=c, R1=b, R2=d, phys_in=r, phys_out=u).
            C = einsum('abrs,cdsu->acbdru', W1, W2)

            # Step 3: fuse right bonds first so that after the left fusion
            # the final axis order is (L, R, phys_in, phys_out).
            # merge_axes places the merged axis at position 0.
            C, _ = merge_axes(C, [2, 3], merged_tag='R', direction=Direction.OUT)
            # C axes: (R=0, L1=1, L2=2, phys_in=3, phys_out=4)

            # Step 4: fuse left bonds.
            C, _ = merge_axes(C, [1, 2], merged_tag='L', direction=Direction.IN)
            # C axes: (L=0, R=1, phys_in=2, phys_out=3)

            # Step 5: restore standard MPO itags.
            C.retag(
                [0, 1, 2, 3],
                [f'W{i:0{ndigits}d}', f'W{i+1:0{ndigits}d}', f's{i:02d}', f's{i:02d}'],
            )
            sites.append(C)

        return NormalMPO(sites, scale=self._scale * other._scale, bc=self.bc)

    # ------------------------------------------------------------------
    # MPO-MPO sum
    # ------------------------------------------------------------------

    def __add__(self, other: NormalMPO) -> NormalMPO:
        """Return the MPO sum ``self + other`` without canonicalization.

        Distributes the relative weight ``alpha = other._scale / self._scale``
        uniformly across sites as ``alpha^(1/L)`` per site of ``other``,
        then combines site tensors via `oplus` following the same bond-axis
        convention as `build_hamiltonian`:

        - Site 0 (leading): fuse right bond (axis 1).
        - Interior sites: fuse both bonds (axes 0 and 1).
        - Site L-1 (terminal): fuse left bond (axis 0).

        Requires ``L >= 2``.

        Parameters
        ----------
        other:
            Right summand. Must have the same chain length and physical
            space as ``self``.

        Returns
        -------
        NormalMPO
            Raw sum MPO with ``scale = self._scale``.

        Raises
        ------
        ValueError
            If ``L < 2``.
        """
        L = self.L
        if L < 2:
            raise ValueError("NormalMPO.__add__ requires L >= 2")

        alpha = other._scale / self._scale
        # Distribute |alpha|^(1/L) uniformly across all sites; apply the
        # sign of alpha to the first site to avoid complex fractional powers
        # when alpha < 0.
        abs_alpha = abs(alpha)
        per_site = abs_alpha ** (1.0 / L)

        sites: List[Tensor] = []
        for i in range(L):
            # Site 0 absorbs the sign of alpha so the product over all L sites
            # equals alpha (not |alpha|).
            factor = math.copysign(per_site, alpha) if i == 0 else per_site
            scaled_other = other._tensors[i] * factor

            if i == 0:
                # Leading site: fuse right bond (axis 1).
                C = oplus(self._tensors[i], scaled_other, axes=[1])
            elif i == L - 1:
                # Terminal site: fuse left bond (axis 0).
                C = oplus(self._tensors[i], scaled_other, axes=[0])
            else:
                # Interior sites: fuse both bond axes.
                C = oplus(self._tensors[i], scaled_other, axes=[0, 1])
            sites.append(C)

        return NormalMPO(sites, scale=self._scale, bc=self.bc)

    # ------------------------------------------------------------------
    # Scalar multiplication
    # ------------------------------------------------------------------

    def __mul__(self, scalar: float) -> NormalMPO:
        """Return a copy scaled by `scalar`.

        Only `_scale` is updated; the internal tensors are not modified.

        Parameters
        ----------
        scalar:
            Real scaling factor.

        Returns
        -------
        NormalMPO
            Scaled copy.
        """
        return NormalMPO(
            [t.clone() for t in self._tensors],
            scale=self._scale * float(scalar),
            bc=self.bc,
            center=self._center,
        )

    def __rmul__(self, scalar: float) -> NormalMPO:
        """Support ``scalar * mpo``; delegates to `__mul__`."""
        return self.__mul__(scalar)


# ---------------------------------------------------------------------------
# Thermal MPO via Taylor expansion
# ---------------------------------------------------------------------------

def thermal_mpo(
    H: MPO,
    beta: float,
    order: int,
    spc: Index,
) -> NormalMPO:
    r"""Approximate the thermal density matrix via a Taylor expansion.

    Computes

    .. math::

        \rho(\beta) = e^{-\beta H} \approx \sum_{n=0}^{N}
            \frac{(-\beta)^n}{n!} H^n

    where :math:`H^0 = I` (identity). The computation is performed in
    MPO arithmetic, keeping the underlying MPO normalized at each step
    (via explicit `compact()` calls) to prevent numerical blow-up.

    Parameters
    ----------
    H:
        Hamiltonian as an MPO (will not be modified).
    beta:
        Inverse temperature :math:`\beta \geq 0`.
    order:
        Truncation order :math:`N` of the Taylor series.
    spc:
        Physical space index used to build the identity MPO
        (:math:`H^0 = I`).

    Returns
    -------
    NormalMPO
        Approximation of :math:`e^{-\beta H}` with physical scale stored
        in ``_scale`` (which equals :math:`Z = \operatorname{Tr}[\rho]`
        up to the MPO norm).

    Notes
    -----
    `compact()` is called after every addition and every power step to
    keep bond dimensions under control. Bond arrow directions are restored
    by `capcup` inside `compact()`, so the result is always in a
    well-defined state for subsequent `trace()` or `observe()` calls.
    """
    # H_n: the normalized version of H used for repeated multiplication.
    H_n = NormalMPO.from_mpo(H)

    # rho: accumulator, initialized to the identity (zeroth-order term).
    rho = NormalMPO.from_mpo(_identity_mpo(spc, H.L))

    # H_pow: running power H^n, starting at H^1 = H.
    H_pow = NormalMPO.from_mpo(H)

    for n in range(1, order + 1):
        coeff = (-beta) ** n / factorial(n)
        rho = rho + H_pow * coeff
        rho.compact()
        H_pow = H_pow @ H_n
        H_pow.compact()

    return rho
