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


"""Krylov (Lanczos) matrix exponential for TDVP local time evolution.

Computes `exp(-i dt H) v` for a Hermitian operator `H` given only a
matrix-vector product routine, which is what every TDVP local update needs.
This module plays the role `davidson.py` plays for DMRG: the one iterative
kernel the sweep logic delegates its linear algebra to.

Algorithm outline
-----------------
1. Normalize `v` and seed the Krylov basis `V = [v]`.
2. Grow the basis by the Lanczos three-term recurrence, which for a
   Hermitian `H` reduces the projected operator to a real symmetric
   tridiagonal matrix `T` with diagonal `alpha` and off-diagonal `beta`.
3. Re-orthogonalize each new direction against the whole basis. Plain
   Lanczos loses orthogonality rapidly in floating point, and because the
   result here is a propagator rather than a single eigenvalue, that loss
   shows up directly as a non-unitary step.
4. Exponentiate `T` by diagonalizing it with `numpy.linalg.eigh`. For a
   Hermitian `T` this is exact up to roundoff, so no dedicated matrix
   exponential (and no SciPy dependency) is required:

       exp(-i dt T) = Q exp(-i dt Λ) Qᴴ

5. Return `‖v‖ · Σ_j [exp(-i dt T) e₁]_j V[j]`.

Termination
-----------
Growth stops on whichever comes first:

- **Happy breakdown**: `beta` underflows, meaning the Krylov space is
  `H`-invariant and the result is exact within the space already built.
- **Error estimate**: the modulus of the last coefficient of
  `exp(-i dt T) e₁` falls below `tol`. That coefficient is the weight the
  propagator puts on the newest direction, and hence the leading term of the
  truncation error.
- **Dimension cap**: `max_krylov` vectors have been built.

The residual error estimate is returned so callers can log it, mirroring how
`davidson` reports its residual norm.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from nicole import Tensor, einsum


# ---------------------------------------------------------------------------
# Inner-product and linear-combination helpers
# ---------------------------------------------------------------------------

def inner_product(v1: Tensor, v2: Tensor) -> complex:
    """Compute the inner product ⟨v1|v2⟩ = Σ_{all indices} conj(v1) * v2.

    Works for tensors of any order ≥ 2; the einsum pattern is built from the
    actual number of indices so the helper serves the 2nd-order bond tensor of
    the zero-site update, the 3rd-order site tensor of the 1-site update, and
    the 4th-order bulk tensor of the 2-site update alike.

    Unlike its DMRG counterpart, the result is always returned as `complex`:
    TDVP vectors carry physical phases, so dropping the imaginary part would
    silently corrupt the Lanczos recurrence.

    Parameters
    ----------
    v1:
        Bra tensor.
    v2:
        Ket tensor with the same index structure as `v1`.

    Returns
    -------
    complex
        The inner product value.
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
    dt: float,
    max_krylov: int = 16,
    tol: float = 1e-12,
) -> Tuple[Tensor, float]:
    """Apply the propagator `exp(-i dt H)` to a tensor via a Krylov expansion.

    Parameters
    ----------
    matvec_fn:
        Callable applying the effective Hamiltonian to a tensor, returning a
        tensor of the same index structure. Must represent a Hermitian
        operator; the Lanczos recurrence relies on it.
    v:
        Tensor to propagate. Must be complex, since the propagator is.
    dt:
        Time increment. A positive value evolves forward (`exp(-i dt H)`), a
        negative value backward, which is how the zero-site update undoes the
        double counting of the tangent-space projector.
    max_krylov:
        Maximum number of Krylov vectors to build.
    tol:
        Convergence threshold on the modulus of the last propagator
        coefficient.

    Returns
    -------
    Tensor
        `exp(-i dt H) v`, carrying the norm of `v` (the propagator is
        unitary, so this preserves `‖v‖` up to the Krylov truncation).
    float
        Final error estimate, the modulus of the last coefficient at exit.

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

    # The Krylov basis is built from the normalized vector; the norm is
    # reinstated at the end, keeping the tridiagonal matrix scale-free.
    V: List[Tensor] = [_scale(1.0 / norm_v, v)]
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
        # and here that directly costs unitarity of the step.
        for u in V:
            w = _axpy(-inner_product(u, w), u, w)

        # Evaluate the propagator in the space built so far, both to decide
        # whether another direction is worth its matvec and to assemble the
        # answer once the loop exits. The weight the propagator puts on the
        # newest direction bounds the truncation error; for a single Krylov
        # vector that modulus is 1, correctly reporting "no information".
        coeffs = _propagator_coeffs(alpha, beta, dt)
        error = float(abs(coeffs[-1]))

        nrm = w.norm()
        # Happy breakdown: the Krylov space is invariant under H, so the
        # propagator restricted to it is already exact.
        if nrm < 1e-14:
            error = 0.0
            break
        if error < tol or j == max_krylov - 1:
            break

        beta.append(float(nrm))
        V.append(_scale(1.0 / nrm, w))
        w = matvec_fn(V[-1])

    # Reassemble the propagated tensor and restore the original norm.
    out = _scale(complex(coeffs[0]) * norm_v, V[0])
    for j in range(1, len(V)):
        out = _axpy(complex(coeffs[j]) * norm_v, V[j], out)

    return out, error


def _propagator_coeffs(
    alpha: List[float],
    beta: List[float],
    dt: float,
) -> np.ndarray:
    """Return the first column of `exp(-i dt T)` for the tridiagonal `T`.

    `T` is real symmetric, so `eigh` gives an exact spectral decomposition
    and the exponential is applied to the eigenvalues directly. Only the
    first column is needed, since the Krylov expansion starts from `e₁`.

    Parameters
    ----------
    alpha:
        Diagonal entries of `T` (length `k`).
    beta:
        Off-diagonal entries of `T` (length `k-1`).
    dt:
        Time increment.

    Returns
    -------
    numpy.ndarray
        Complex vector of length `k` holding `[exp(-i dt T)]_{:,0}`.
    """
    k = len(alpha)
    T = np.zeros((k, k), dtype=np.float64)
    T[np.diag_indices(k)] = alpha
    if k > 1:
        idx = np.arange(k - 1)
        T[idx, idx + 1] = beta
        T[idx + 1, idx] = beta

    evals, evecs = np.linalg.eigh(T)
    # exp(-i dt T) e₁ = Q exp(-i dt Λ) Qᴴ e₁, and Qᴴ e₁ is Q's first row.
    phases = np.exp(-1j * dt * evals)
    return evecs @ (phases * evecs[0, :].conj())
