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


"""Davidson iterative eigensolver for DMRG local optimisation.

Finds the lowest eigenvalue and corresponding eigenvector of a large symmetric
operator given only a matrix-vector product routine (`matvec_fn`). No
preconditioning is applied — the plain residual is used as the correction
vector. This avoids the need to extract the diagonal of the effective
Hamiltonian, which is expensive for symmetric tensors.

Algorithm outline (residual variant)
-------------------------------------
1. Normalise the initial guess `v0` and seed the Krylov subspace `V = [v0]`.
2. Apply the operator: `HV = [matvec_fn(v0)]`.
3. Build the 1×1 projected matrix `H_sub = [[⟨v0|Hv0⟩]]`.
4. Iterate:
   a. Diagonalise `H_sub` (real symmetric) → lowest eigenvalue `θ` and
      Ritz coefficient vector `u`.
   b. Form the Ritz vector `q = Σ_i u[i] V[i]`.
   c. Compute the residual `r = Σ_i u[i] HV[i] − θ·q`.
   d. If `‖r‖ < tol`: converged; return `(θ, q)`.
   e. If the subspace has reached `max_subspace`: collapse to `q` and
      restart (thick restart with a single vector).
   f. Otherwise: orthogonalise `r` against `V` (Gram-Schmidt), append the
      new direction to `V` and `HV`, and extend `H_sub` by one row/column.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from nicole import Tensor, einsum


# ---------------------------------------------------------------------------
# Inner-product and linear-combination helpers
# ---------------------------------------------------------------------------

def _inner_product(v1: Tensor, v2: Tensor) -> complex:
    """Compute the inner product ⟨v1|v2⟩ = Σ_{all indices} conj(v1) * v2.

    Works for tensors of any rank ≥ 2. The einsum pattern is built
    dynamically from the actual number of indices so that the helper is not
    tied to the 3-index MPS site-tensor convention.

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
    pattern = f'{letters},{letters}->'
    ip_tensor = einsum(pattern, v1.conj(), v2)
    # The value is a scalar tensor; extract the numeric value.
    k, v = next(iter(ip_tensor.data.items()))
    weight = 1.0 if ip_tensor.intw is None else float(ip_tensor.intw[k].weights[0, 0])
    val = v.item()
    # Preserve the natural scalar type: return float for real-valued tensors and
    # complex for complex-valued tensors. Returning complex when the data is
    # float64 would cause Nicole to promote tensors to complex128 in _axpy,
    # producing a dtype mismatch on the subsequent contraction.
    return val * weight


def _axpy(alpha: complex, x: Tensor, y: Tensor) -> Tensor:
    """Compute `y + alpha * x` for two compatible site tensors.

    Parameters
    ----------
    alpha:
        Scalar coefficient.
    x:
        Tensor to scale.
    y:
        Tensor to add to.

    Returns
    -------
    Tensor
        The linear combination `y + alpha * x`.
    """
    # Nicole Tensor supports scalar multiplication and addition.
    return y + x * alpha


def _scale(alpha: complex, x: Tensor) -> Tensor:
    """Return `alpha * x`."""
    return x * alpha


# ---------------------------------------------------------------------------
# Davidson solver
# ---------------------------------------------------------------------------

def davidson(
    matvec_fn: Callable[[Tensor], Tensor],
    v0: Tensor,
    max_iter: int = 100,
    tol: float = 1e-10,
    max_subspace: int = 20,
) -> Tuple[float, Tensor, float]:
    """Find the lowest eigenvalue of a symmetric operator via the Davidson method.

    Uses the plain residual as the correction vector (no preconditioning).
    When the Krylov subspace reaches `max_subspace` vectors without converging,
    the subspace is collapsed to the current Ritz vector (single-vector thick
    restart) and the iteration continues.

    Parameters
    ----------
    matvec_fn:
        Callable that applies the operator to a site tensor and returns a
        tensor of the same shape. Must represent a Hermitian operator.
    v0:
        Initial guess tensor with axes `(left_bond, right_bond, physical)`.
    max_iter:
        Maximum number of Davidson iterations.
    tol:
        Convergence threshold on the residual norm `‖r‖`.
    max_subspace:
        Maximum number of vectors in the Krylov subspace before a restart.

    Returns
    -------
    float
        Approximate lowest eigenvalue `θ`.
    Tensor
        Approximate eigenvector (Ritz vector) `q`, normalised to unit norm.
    float
        Final residual norm `‖r‖` at exit (convergence, subspace collapse, or
        max iterations reached).

    Raises
    ------
    ValueError
        If `v0` has zero norm.
    """
    # Normalise the initial guess.
    norm0 = v0.norm()
    if norm0 == 0.0:
        raise ValueError("initial guess v0 has zero norm")
    v0 = _scale(1.0 / norm0, v0)

    # Seed the subspace with the normalised initial guess.
    V: List[Tensor] = [v0]
    HV: List[Tensor] = [matvec_fn(v0)]

    # Build the 1×1 projected Hamiltonian.
    ip = _inner_product(V[0], HV[0])
    H_sub = np.array([[ip.real]], dtype=np.float64)

    theta = 0.0
    q = v0

    for iteration in range(max_iter):
        # Diagonalise the small projected matrix (real symmetric for Hermitian H).
        evals, evecs = np.linalg.eigh(H_sub)
        # Select the lowest eigenvalue and its coefficient vector.
        theta = float(evals[0])
        u = evecs[:, 0]  # shape (k,) where k = len(V)

        # Form the Ritz vector q = Σ_i u[i] V[i].
        q = _scale(float(u[0]), V[0])
        for j in range(1, len(V)):
            q = _axpy(float(u[j]), V[j], q)

        # Compute the residual r = Σ_i u[i] HV[i] − θ·q.
        r = _scale(float(u[0]), HV[0])
        for j in range(1, len(V)):
            r = _axpy(float(u[j]), HV[j], r)
        r = _axpy(-theta, q, r)

        # Check for convergence.
        res_norm = r.norm()
        if res_norm < tol:
            return theta, q, float(res_norm)

        # Decide whether to restart or expand the subspace.
        # Use strict greater-than so the subspace can grow to max_subspace
        # vectors before collapsing; == would prevent any expansion when
        # max_subspace=1 and the subspace is seeded with one vector.
        if len(V) > max_subspace:
            # Collapse the subspace to the current Ritz vector and restart.
            Hq = matvec_fn(q)
            V = [q]
            HV = [Hq]
            ip = _inner_product(q, Hq)
            H_sub = np.array([[ip.real]], dtype=np.float64)
        else:
            # Orthogonalise the residual against the current subspace (Gram-Schmidt).
            v_new = r
            for vi in V:
                coeff = _inner_product(vi, v_new)
                v_new = _axpy(-coeff, vi, v_new)

            new_norm = v_new.norm()
            if new_norm < 1e-14:
                # The new direction is linearly dependent; force convergence.
                break  # res_norm holds the value from the most recent check above
            v_new = _scale(1.0 / new_norm, v_new)

            # Apply the operator to the new basis vector.
            Hv_new = matvec_fn(v_new)

            # Extend H_sub by one row and column (incremental construction).
            k = len(V)
            H_sub_new = np.zeros((k + 1, k + 1), dtype=np.float64)
            H_sub_new[:k, :k] = H_sub
            for j in range(k):
                # Off-diagonal: ⟨V[j]|H|v_new⟩
                entry = _inner_product(V[j], Hv_new).real
                H_sub_new[j, k] = entry
                H_sub_new[k, j] = entry
            # Diagonal: ⟨v_new|H|v_new⟩
            H_sub_new[k, k] = _inner_product(v_new, Hv_new).real

            V.append(v_new)
            HV.append(Hv_new)
            H_sub = H_sub_new

    # Return the best estimate without raising, to allow graceful degradation.
    # The caller (optimize_site) can inspect davidson_error if needed.
    return theta, q, float(res_norm)
