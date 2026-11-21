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


"""Krylov (Lanczos) imaginary-time propagator for TgTRG local updates.

Computes `exp(-tau H) v` for a Hermitian operator `H` given only a
matrix-vector product routine, all that every local update of the cooling
sweep needs. This module plays the role `davidson.py` plays for DMRG and
`krylov.py` plays for TDVP: the one iterative kernel the sweep logic
delegates its linear algebra to.

The Lanczos construction itself is the one TDVP uses — three-term recurrence,
full re-orthogonalization, and an exact `eigh` of the projected tridiagonal
matrix. Two things differ, both consequences of the propagator being real
instead of unitary.

Scale factored out
------------------
`exp(-tau H_eff)` is not norm-preserving, and the effective Hamiltonian of a
chain carries *extensive* eigenvalues. At the step sizes TgTRG is built for
(the exponential stage takes `tau = beta`, so `tau` reaches the final `beta`
itself) a bare `exp(-tau lambda)` leaves float64 range in both directions
long before anything is physically wrong. The result is therefore returned
factored, as a tensor times a log-magnitude:

    exp(-tau H) v = exp(log_scale) * w,    ‖w‖ = ‖v‖

The caller folds `log_scale` into `NormalMPO.log_scale`, where the magnitude
of the density matrix lives anyway, so nothing is lost and nothing is ever
exponentiated at full scale.

The split is exact, and `w` is pinned to the norm of `v`. Subtracting the
extremal eigenvalue before exponentiating — enough to stop the coefficients
from overflowing — is *not* enough to stop them from underflowing: when the
starting tensor has little overlap with the eigenvector the propagator
favors, every surviving coefficient is small, and at the large `tau` of the
exponential stage they can collapse to zero together, leaving a tensor with
no direction left in it. `_propagator_coeffs` therefore measures the largest
*term* of the expansion, overlap included, subtracts its log, and normalizes
what remains, so the returned coefficients are a unit vector whatever `tau`
does.

That maximum is measured on every call, and has to be: the overlaps belong
to the state, not to `H`, and change at every site, half-sweep, and step, so
no quantity computed once before the run can stand in for them. It is also
not worth avoiding. The eigendecomposition is required to apply
`exp(-tau T)` at all, `T` is tridiagonal of dimension at most `max_krylov`,
and over an `L = 8` run of 14 steps — 2,925 solves, mean dimension 5.3 —
this whole module accounted for half a percent of the wall time and the
log-space assembly within it for a twelfth of that.

Real arithmetic
---------------
With a real Hamiltonian the whole recurrence stays in float64: the projected
matrix is real symmetric, and so are the propagator coefficients. Nothing
here promotes a real density matrix to complex, unlike real-time evolution,
where the propagator itself forces it. A complex state is still handled
correctly — the coefficients multiplying it are simply real.

Termination
-----------
Growth stops on whichever comes first:

- **Happy breakdown**: `beta` underflows, meaning the Krylov space is
  `H`-invariant and the result is exact within the space already built.
- **Error estimate**: the weight the propagator puts on the newest Krylov
  direction, measured *relative* to the whole coefficient vector, falls
  below `tol`. The imaginary-time case needs the relative form: the
  coefficients decay with `tau`, so an absolute comparison would call any
  large step converged after one direction.
- **Dimension cap**: `max_krylov` vectors have been built.

The error estimate is returned so callers can log it, mirroring how
`davidson` reports its residual norm.
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np

from nicole import Tensor, einsum


# ---------------------------------------------------------------------------
# Inner-product and linear-combination helpers
# ---------------------------------------------------------------------------

def inner_product(v1: Tensor, v2: Tensor) -> complex:
    """Compute the inner product ⟨v1|v2⟩ = Σ_{all indices} conj(v1) * v2.

    For the density matrices this module propagates, the sum runs over both
    physical axes as well as the virtual ones, which is exactly the Frobenius
    product `Tr[v1† v2]` of the operators they represent — the inner product
    the tangent-space projection is defined with respect to.

    Works for tensors of any order ≥ 2, so the same helper serves the
    2nd-order bond tensor of the zero-site update, the 4th-order site tensor
    of the 1-site update, and the 6th-order bulk tensor of the 2-site update
    alike.

    Parameters
    ----------
    v1:
        Bra tensor.
    v2:
        Ket tensor with the same index structure as `v1`.

    Returns
    -------
    complex
        The inner product value. Real to roundoff for a real density matrix,
        but returned as `complex` so the caller decides where to drop the
        imaginary part (see `_cast_for`).
    """
    n = len(v1.indices)
    letters = ''.join(chr(ord('a') + k) for k in range(n))
    ip_tensor = einsum(f'{letters},{letters}->', v1.conj(), v2)
    # The value is a scalar tensor; extract the numeric value, including the
    # Bridge weight that carries the CG normalization for generic groups.
    k, v = next(iter(ip_tensor.data.items()))
    weight = (
        1.0 + 0.0j if ip_tensor.intw is None
        else complex(ip_tensor.intw[k].weights[0, 0].item())
    )
    return complex(v.item()) * weight


def _cast_for(v: Tensor) -> Callable[[complex], complex]:
    """Return the scalar cast that keeps `v`'s dtype intact.

    Scaling a float64 tensor by a Python `complex` would promote it, and the
    promotion would then spread through the whole density matrix one local
    update at a time. For a real `v` the inner products are real to roundoff
    (the operator is Hermitian), so dropping the imaginary part is exact up
    to the error it already carries.
    """
    if v.dtype.is_complex:
        return lambda z: z
    return lambda z: z.real


def _axpy(alpha: complex, x: Tensor, y: Tensor) -> Tensor:
    """Compute `y + alpha * x` for two compatible tensors."""
    return y + x * alpha


def _scale(alpha: complex, x: Tensor) -> Tensor:
    """Return `alpha * x`."""
    return x * alpha


# ---------------------------------------------------------------------------
# Krylov propagator
# ---------------------------------------------------------------------------

def lanczos_expv(
    matvec_fn: Callable[[Tensor], Tensor],
    v: Tensor,
    tau: float,
    max_krylov: int = 16,
    tol: float = 1e-12,
) -> Tuple[Tensor, float, float]:
    """Apply the propagator `exp(-tau H)` to a tensor via a Krylov expansion.

    The result is returned *factored*, as a tensor times a log-magnitude:

        exp(-tau H) v = exp(log_scale) * w

    See the module docstring for why: at TgTRG step sizes the magnitude
    alone can leave float64 range, while `w` stays at the scale of `v`.

    Parameters
    ----------
    matvec_fn:
        Callable applying the effective Hamiltonian to a tensor, returning a
        tensor of the same index structure. Must represent a Hermitian
        operator; the Lanczos recurrence relies on it.
    v:
        Tensor to propagate. May be real or complex.
    tau:
        Imaginary-time increment. A positive value cools (`exp(-tau H)`), a
        negative value heats, which the zero-site and back-evolution updates
        use to undo the double counting of the tangent-space projector.
    max_krylov:
        Maximum number of Krylov vectors to build.
    tol:
        Convergence threshold on the relative weight of the last propagator
        coefficient.

    Returns
    -------
    Tensor
        `w`, the direction of the propagated tensor, carrying the norm of
        `v` exactly.
    float
        `log_scale`, the log of the magnitude divided out. Multiply `exp` of
        it into `w` to recover `exp(-tau H) v`, or add it to a tracked
        log-magnitude.
    float
        Final error estimate, the relative weight of the last coefficient at
        exit.

    Raises
    ------
    ValueError
        If `v` has zero norm, or `max_krylov < 1`.
    """
    if max_krylov < 1:
        raise ValueError(f"max_krylov must be >= 1, got {max_krylov}")

    norm_v = v.norm()
    if norm_v == 0.0:
        raise ValueError("cannot propagate a tensor of zero norm")

    cast = _cast_for(v)

    # The Krylov basis is built from the normalized vector; the norm is
    # reinstated at the end, keeping the tridiagonal matrix scale-free.
    V: List[Tensor] = [_scale(cast(complex(1.0 / norm_v)), v)]
    # alpha[j] = ⟨V[j]|H|V[j]⟩, beta[j] = ‖residual after step j‖, both real
    # for a Hermitian operator.
    alpha: List[float] = []
    beta: List[float] = []

    error = 0.0
    w = matvec_fn(V[0])

    for j in range(max_krylov):
        alpha.append(inner_product(V[j], w).real)

        # Three-term recurrence: strip the components along the current and
        # previous basis vectors.
        w = _axpy(-alpha[j], V[j], w)
        if j > 0:
            w = _axpy(-beta[j - 1], V[j - 1], w)

        # Full re-orthogonalization. The three-term recurrence is exact in
        # theory, but roundoff reintroduces components along earlier vectors,
        # and a propagator built on a basis that is no longer orthogonal
        # misreports both the step and its own error.
        for u in V:
            w = _axpy(-cast(inner_product(u, w)), u, w)

        # Evaluate the propagator in the space built so far, both to decide
        # whether another direction is worth its matvec and to assemble the
        # answer once the loop exits. The weight the propagator puts on the
        # newest direction bounds the truncation error; for a single Krylov
        # vector that relative weight is 1, correctly reporting "no
        # information".
        coeffs, log_scale = _propagator_coeffs(alpha, beta, tau)
        error = float(abs(coeffs[-1]) / np.linalg.norm(coeffs))

        nrm = w.norm()
        # Happy breakdown: the Krylov space is invariant under H, so the
        # propagator restricted to it is already exact.
        if nrm < 1e-14:
            error = 0.0
            break
        if error < tol or j == max_krylov - 1:
            break

        beta.append(float(nrm))
        V.append(_scale(cast(complex(1.0 / nrm)), w))
        w = matvec_fn(V[-1])

    # Reassemble the propagated tensor and restore the original norm.
    out = _scale(cast(complex(coeffs[0] * norm_v)), V[0])
    for j in range(1, len(V)):
        out = _axpy(cast(complex(coeffs[j] * norm_v)), V[j], out)

    return out, log_scale, error


def _propagator_coeffs(
    alpha: List[float],
    beta: List[float],
    tau: float,
) -> Tuple[np.ndarray, float]:
    """Return the normalized first column of `exp(-tau T)` and its log-magnitude.

    `T` is real symmetric, so `eigh` gives an exact spectral decomposition
    and the exponential is applied to the eigenvalues directly. Only the
    first column is needed, since the Krylov expansion starts from `e₁`:

        exp(-tau T) e₁ = Q exp(-tau Λ) Qᵀ e₁

    and `Qᵀ e₁` is `Q`'s first row, the overlap of the starting vector with
    each eigenvector.

    The column is assembled in log space. Each term carries the log-magnitude
    `-tau λ_j + log|overlap_j|`, the largest of which is subtracted before
    anything is exponentiated. That keeps the terms inside float64 range
    whatever `tau` is, and — unlike subtracting a bare eigenvalue — it
    accounts for the overlaps, so the largest term is exactly `1` and the
    assembled column cannot collapse to zero. What remains is normalized,
    putting the whole magnitude in the second return value.

    Parameters
    ----------
    alpha:
        Diagonal entries of `T` (length `k`).
    beta:
        Off-diagonal entries of `T` (length `k-1`).
    tau:
        Imaginary-time increment.

    Returns
    -------
    numpy.ndarray
        Real unit vector of length `k`, the direction of
        `[exp(-tau T)]_{:,0}`.
    float
        `log` of the magnitude divided out, so that
        `exp(log_scale) * coeffs` is the column itself.
    """
    k = len(alpha)
    T = np.zeros((k, k), dtype=np.float64)
    T[np.diag_indices(k)] = alpha
    if k > 1:
        idx = np.arange(k - 1)
        T[idx, idx + 1] = beta
        T[idx + 1, idx] = beta

    evals, evecs = np.linalg.eigh(T)
    overlap = evecs[0, :]
    # An eigenvector orthogonal to the starting vector contributes nothing,
    # and its -inf log is harmless: the exponential below sends it to zero.
    with np.errstate(divide='ignore'):
        logs = -tau * evals + np.log(np.abs(overlap))
    # `overlap` is a unit row of an orthogonal matrix, so at least one entry
    # is nonzero and the maximum is finite.
    log_max = float(np.max(logs))
    terms = np.exp(logs - log_max) * np.sign(overlap)

    coeffs = evecs @ terms
    norm = float(np.linalg.norm(coeffs))

    return coeffs / norm, log_max + math.log(norm)
