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
magnitude in log form as `log_scale`, keeping it representable far outside
float64 range. Any sign the physical operator carries lives directly in
the tensor data (e.g. folded into site 0 by `__mul__`) — see the
`NormalMPO` class docstring. `thermal_mpo` approximates `ρ(β) = exp(−βH)`
via a truncated Taylor series.

The one-directional import chain is: `thermal.py` → `network.py`. No
circular dependency is introduced.
"""

from __future__ import annotations

import math
from math import factorial
from typing import List, Tuple, Optional

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
# Scale decomposition helper
# ---------------------------------------------------------------------------

def _decompose_scale(x: float) -> Tuple[float, float]:
    """Decompose a raw magnitude into `(log(|x|), sign(x))`.

    `x == 0.0` maps to the sentinel `(-inf, 0.0)`, since `log(0)` is
    undefined. Used internally by `NormalMPO.__mul__` (to split a scalar
    factor into a log-magnitude and a sign to fold into site 0) and by
    `NormalMPO.log_trace()` (to split the raw contracted trace value into
    the same form).

    Parameters
    ----------
    x:
        Raw magnitude, positive, negative, or zero.

    Returns
    -------
    float
        `log(|x|)`, or `-inf` if `x == 0.0`.
    float
        `sign(x)` as `+1.0`/`-1.0`, or `0.0` if `x == 0.0`.
    """
    if x == 0.0:
        return -math.inf, 0.0
    return math.log(abs(x)), math.copysign(1.0, x)


# ---------------------------------------------------------------------------
# NormalMPO
# ---------------------------------------------------------------------------

class NormalMPO(MPO):
    """MPO with a separately tracked overall scale magnitude.

    Represents the physical operator as `exp(log_scale) × mpo`, where the
    internal MPO satisfies `mpo.norm() ≈ 1` after `compact()` or
    `from_mpo()`. Tracking the magnitude in log form (rather than as a raw
    float) keeps it representable even when it is far outside float64
    range — this matters for XTRG, where `Tr[ρ]` is repeatedly squared and
    can reach `~10^500` or beyond at low temperature.

    The *sign* of the physical operator lives directly in the internal
    MPO's tensor data. This works because `mpo.norm() ≈ 1` does not pin
    down a sign (`-X` and `X` have the same Frobenius norm), and because
    canonicalization (`compact()`, `canonical()`) is an exact gauge
    transform that cannot change the value of any fully-contracted
    quantity such as `Tr[ρ]` — so whatever sign is baked into the tensors
    survives compaction unchanged. Whenever an operation needs to apply a
    sign flip (e.g. `__mul__` by a negative scalar, or the alternating
    `(-β)^n` terms in `thermal_mpo`'s Taylor accumulation), it is folded
    into site 0's tensor by multiplying it by `-1.0`. Site 0 is used by
    convention because, in an OBC chain, its left bond is the trivial
    boundary dimension, making it generically the cheapest tensor to
    touch. Because tensor contraction and addition are linear, sign flips
    baked into the operands' tensors propagate correctly through
    `__matmul__`/`__add__` — see those methods' docstrings.

    All arithmetic operations (`+`, `@`, `*`) preserve this representation,
    updating `log_scale` analytically (by addition, never by exponentiating
    a possibly-astronomical magnitude) without altering the unit-norm
    convention until `compact()` is called explicitly.

    Parameters
    ----------
    tensors:
        Non-empty list of site tensors, each with 4 axes
        `(left_bond, right_bond, phys_in, phys_out)`.
    log_scale:
        `log(scale)`, where `scale = ||exp(log_scale) * mpo||_F` is the
        (always non-negative) Frobenius-norm magnitude of the physical
        operator. Defaults to `0.0` (i.e. `scale = 1.0`).
    bc:
        Boundary condition: `'OBC'` (default) or `'PBC'`.
    center:
        Orthogonality center site index, or `None` if unspecified.
    """

    def __init__(
        self,
        tensors: List[Tensor],
        log_scale: float = 0.0,
        bc: str = 'OBC',
        center: Optional[int] = None,
    ) -> None:
        super().__init__(tensors, bc=bc, center=center)
        self._log_scale: float = float(log_scale)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_mpo(cls, mpo: MPO) -> NormalMPO:
        """Create a `NormalMPO` from a plain `MPO`.

        The input MPO is not modified. The returned object has
        `mpo.norm() ≈ 1` and `scale == original_frobenius_norm`.

        Parameters
        ----------
        mpo:
            Source MPO. May have any canonical form.

        Returns
        -------
        NormalMPO
            Normalized copy with `center=0`.

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
        return cls(copy._tensors, log_scale=math.log(n), bc=mpo.bc, center=0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def log_scale(self) -> float:
        """`log(scale)` (read-only). The primary, overflow-safe representation."""
        return self._log_scale

    @property
    def scale(self) -> float:
        """Overall scale magnitude as a raw, non-negative float (read-only).

        Reconstructs `exp(log_scale)`. For magnitudes too large to
        represent in float64, `math.exp` raises `OverflowError`; this is
        caught here and `inf` is returned instead, since a getter should
        never crash. Callers needing to represent genuinely astronomical
        magnitudes should use `log_scale` directly instead of this
        property.
        """
        try:
            return math.exp(self._log_scale)
        except OverflowError:
            return math.inf

    # ------------------------------------------------------------------
    # In-place scale update
    # ------------------------------------------------------------------

    def scale_by(self, log_scale_delta: float) -> None:
        """Multiply the tracked scale magnitude in-place by `exp(log_scale_delta)`.

        This is the safe, in-place counterpart to `__init__`: it combines
        two log-magnitudes by addition rather than ever exponentiating
        either one, so it cannot overflow even when the combined scale
        would be far outside float64 range. Used by XTRG's `_fit_mpo` to
        fold pre-computed physical scale magnitudes into an
        already-compacted `NormalMPO`.

        Parameters
        ----------
        log_scale_delta:
            `log(factor)` of the multiplicative magnitude to apply.
        """
        self._log_scale += log_scale_delta

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compact(self, trunc: Optional[dict] = None) -> None:
        """Compress bond dimensions in-place, keeping the scale consistent.

        Performs the same two-sweep canonicalization as `MPO.compact()`, but
        folds the extracted norm into `log_scale` instead of redistributing
        it across the site tensors. After the call, the internal MPO is
        approximately unit-normed and `log_scale` absorbs the physical
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
            `{'thresh': 1e-14}`.
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

        # Fold the extracted norm into log_scale; tensors remain unit-normed.
        # n is a Frobenius norm (always >= 0); guard n == 0 explicitly since
        # math.log(0.0) raises rather than giving -inf.
        self._log_scale += math.log(n) if n != 0.0 else -math.inf

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

    def log_trace(self) -> Tuple[float, float]:
        r"""Compute \(\log|\operatorname{Tr}[\rho]|\) and its sign.

        Performs a left-to-right transfer-matrix sweep. At each site the
        physical indices (phys_in and phys_out, sharing itag `s{i:02d}`
        with opposite directions) are traced using Nicole's `trace` function,
        yielding a 2-leg bond tensor. Adjacent bond tensors are chained with
        `einsum`.

        This is the overflow-safe counterpart to `trace()`: it never forms
        `Tr[ρ]` itself, so it remains finite even when the physical trace
        would be far outside float64 range (as happens for XTRG runs at low
        temperature, where `Tr[ρ]` can reach `~10^500`).

        Returns
        -------
        float
            `log|Tr[ρ]|`, or `-inf` if the trace is exactly zero (sentinel,
            see `_decompose_scale`).
        float
            `sign(Tr[ρ])`: `+1.0`, `-1.0`, or `0.0` if the trace is exactly
            zero. Computed directly from the contracted tensor data (see
            the class docstring for where sign lives).
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
        raw = float(val.item().real) * weight

        log_raw, sign_raw = _decompose_scale(raw)
        if sign_raw == 0.0:
            return -math.inf, 0.0
        return self._log_scale + log_raw, sign_raw

    def trace(self) -> float:
        r"""Compute the MPO trace \(\operatorname{Tr}[\rho]\).

        Thin wrapper over `log_trace()` that exponentiates back to a raw
        magnitude. For magnitudes too large to represent in float64, `±inf`
        is returned instead of raising (see `scale`). Prefer `log_trace()`
        directly when the trace magnitude may be extreme.

        Returns
        -------
        float
            `Tr[ρ]`, or `±inf` if it overflows float64.
        """
        log_abs, sign = self.log_trace()
        if sign == 0.0:
            return 0.0
        try:
            return sign * math.exp(log_abs)
        except OverflowError:
            return math.copysign(math.inf, sign)

    # ------------------------------------------------------------------
    # MPO-MPO product
    # ------------------------------------------------------------------

    def __matmul__(self, other: NormalMPO) -> NormalMPO:
        """Return the MPO product `self @ other` without canonicalization.

        For each site `i`:

        1. Retag the bond axes of `self[i]` to `'L1'`/`'R1'` and
           `other[i]` to `'L2'`/`'R2'` (isolates bond structure from
           shared physical itags).
        2. Contract over the shared physical axis (`self`'s `phys_out`
           × `other`'s `phys_in`) via
           `einsum('abrs,cdsu->acbdru', W1, W2)`, producing a 6-axis
           tensor `(L1, L2, R1, R2, phys_in, phys_out)`.
        3. Fuse right bonds first: `merge_axes(C, [2, 3], 'R', OUT)`
           → axes `(R, L1, L2, r, u)`.
        4. Fuse left bonds: `merge_axes(C, [1, 2], 'L', IN)`
           → axes `(L, R, phys_in, phys_out)`. *(Correct index order.)*
        5. Retag to standard MPO bond itags.

        The resulting bond dimensions are `χ(self) × χ(other)` before any
        `compact()` call.

        Parameters
        ----------
        other:
            Right operand, must have the same chain length and physical
            space as `self`.

        Returns
        -------
        NormalMPO
            Raw product MPO with `log_scale = self.log_scale + other.log_scale`
            (combined additively so the product's scale cannot overflow
            even if the raw magnitude would).
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

        return NormalMPO(
            sites,
            log_scale=self._log_scale + other._log_scale,
            bc=self.bc,
        )

    # ------------------------------------------------------------------
    # MPO-MPO sum
    # ------------------------------------------------------------------

    def __add__(self, other: NormalMPO) -> NormalMPO:
        """Return the MPO sum `self + other` without canonicalization.

        Distributes the relative magnitude `alpha = other.scale / self.scale`
        uniformly across sites as `alpha^(1/L)` per site of `other`,
        then combines site tensors via `oplus` following the same bond-axis
        convention as `build_hamiltonian`:

        - Site 0 (leading): fuse right bond (axis 1).
        - Interior sites: fuse both bonds (axes 0 and 1).
        - Site L-1 (terminal): fuse left bond (axis 0).

        Requires `L >= 2`.

        Parameters
        ----------
        other:
            Right summand. Must have the same chain length and physical
            space as `self`.

        Returns
        -------
        NormalMPO
            Raw sum MPO with `log_scale = self.log_scale` (the result is
            expressed relative to `self`'s scale).

        Raises
        ------
        ValueError
            If `L < 2`.
        """
        L = self.L
        if L < 2:
            raise ValueError("NormalMPO.__add__ requires L >= 2")

        # log(alpha) = log|other| - log|self|, computed via subtraction so
        # it cannot overflow/underflow even when self and other differ by
        # many orders of magnitude in physical scale.
        log_alpha = other._log_scale - self._log_scale
        # Distribute alpha^(1/L) uniformly across all sites. alpha is a pure
        # magnitude here, so no fractional power of a negative number arises.
        per_site = math.exp(log_alpha / L)

        sites: List[Tensor] = []
        for i in range(L):
            scaled_other = other._tensors[i] * per_site

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

        return NormalMPO(sites, log_scale=self._log_scale, bc=self.bc)

    # ------------------------------------------------------------------
    # Scalar multiplication
    # ------------------------------------------------------------------

    def __mul__(self, scalar: float) -> NormalMPO:
        """Return a copy scaled by `scalar`.

        `log_scale` absorbs `log(|scalar|)`. If `scalar < 0`, the sign is
        folded into site 0's tensor by negating it (exact, lossless — see
        the class docstring for why site 0 is used). All other tensors are
        left untouched (cloned only).

        Parameters
        ----------
        scalar:
            Real scaling factor.

        Returns
        -------
        NormalMPO
            Scaled copy.
        """
        log_c, sign_c = _decompose_scale(float(scalar))
        tensors = [t.clone() for t in self._tensors]
        if sign_c < 0.0:
            tensors[0] = tensors[0] * -1.0
        return NormalMPO(
            tensors,
            log_scale=self._log_scale + log_c,
            bc=self.bc,
            center=self._center,
        )

    def __rmul__(self, scalar: float) -> NormalMPO:
        """Support `scalar * mpo`; delegates to `__mul__`."""
        return self.__mul__(scalar)


# ---------------------------------------------------------------------------
# Thermal MPO via Taylor expansion
# ---------------------------------------------------------------------------

def thermal_mpo(
    H: MPO,
    beta: float,
    order: int,
    spc: Index,
    trunc: Optional[dict] = None,
    coeff_thresh: float = 1e-15,
) -> NormalMPO:
    r"""Approximate the thermal density matrix via a Taylor expansion.

    Computes

    \[\rho(\beta) = e^{-\beta H} \approx \sum_{n=0}^{N}
        \frac{(-\beta)^n}{n!} H^n\]

    where \(H^0 = I\) (identity). The computation is performed in
    MPO arithmetic, keeping the underlying MPO normalized at each step
    (via explicit `compact()` calls) to prevent numerical blow-up.

    Parameters
    ----------
    H:
        Hamiltonian as an MPO (will not be modified).
    beta:
        Inverse temperature \(\beta \geq 0\).
    order:
        Maximum truncation order \(N\) of the Taylor series. The loop
        terminates early once the actual contribution of the \(n\)-th
        term — measured as \(|\beta^n / n!| \times \|H^n\|_F\) — drops
        below `coeff_thresh`.
    spc:
        Physical space index used to build the identity MPO
        (\(H^0 = I\)).
    trunc:
        Truncation options forwarded to each `compact()` call during the
        Taylor series accumulation. Defaults to `{'thresh': 1e-14}`.
    coeff_thresh:
        Early-stopping threshold on the actual contribution magnitude of
        the \(n\)-th Taylor term, \(|\beta^n / n!| \times \|H^n\|_F\).
        Once this falls below `coeff_thresh` the remaining contributions
        are negligible and the loop exits. Defaults to `1e-15`. Note
        that checking the bare coefficient \(|\beta^n / n!|\) alone is
        insufficient for Hamiltonians with large operator norm, because
        \(\|H^n\|_F\) can be much larger than 1.

    Returns
    -------
    NormalMPO
        Approximation of \(e^{-\beta H}\) with physical magnitude stored
        in `log_scale` (which equals \(Z = \operatorname{Tr}[\rho]\) up to
        the MPO norm and sign, the latter living in the tensor data).

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
        # Early stop: compare the actual contribution magnitude
        # |coeff| * ||H^n||_F against the threshold, not the bare
        # coefficient alone.  The bare coefficient beta^n/n! can be
        # below machine precision while |coeff| * ||H^n||_F is still
        # non-negligible for Hamiltonians with large operator norm.
        if abs(coeff) * H_pow.scale < coeff_thresh:
            break
        rho = rho + H_pow * coeff
        rho.compact(trunc)
        if n < order:
            H_pow = H_pow @ H_n
            H_pow.compact(trunc)

    return rho
