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
# Author of code: Madhav Menon.


"""Symmetry-preserving Krylov exponentials on Nicole tensors.

The discarded-projector BUG integrator advances three local objects per bond by a
matrix exponential of a linear map that is supplied as a *tensor-in / tensor-out*
action (a closure), never as a dense matrix:

* the **K** and **L** factors evolve under the discarded-projected effective
  Hamiltonian ``G = P⊥ · H``. That generator is **non-Hermitian** (the projector
  is one-sided), so its exponential uses an Arnoldi (modified Gram–Schmidt)
  Krylov iteration — :func:`tensor_arnoldi_expv`.
* the **S** (core) factor evolves under the augmented-basis Galerkin Hamiltonian
  ``Û† H V̂``-projected, which **is Hermitian**, so its exponential uses the
  cheaper symmetric Lanczos iteration — :func:`tensor_lanczos_expv`.

Both build their Krylov basis out of Nicole tensors and keep only the small dense
Hessenberg/tridiagonal projection in memory. Because every vector stays in the
block-sparse (symmetry-resolved) representation, no amplitude is ever produced
outside the admissible U(1) charge blocks — unlike a dense standard-basis Krylov,
which would mix sectors and be rejected by Nicole. This matches the reference
Julia discarded-BUG, whose K/L substeps call ``KrylovKit.exponentiate(...,
issymmetric=false)`` and whose S substep uses the Hermitian path.

This module is deliberately self-contained: it depends only on ``torch`` and the
public ``nicole`` API, so the :mod:`alice.algorithm.discarded_bug` package does
not couple to any other integrator.
"""

from __future__ import annotations

from typing import Callable

import torch
from nicole import Tensor, conj, contract


def to_complex(tensor: Tensor) -> Tensor:
    """Return a copy of ``tensor`` with every block cast to ``complex128``.

    Real-time evolution exponentiates ``-i dt H``, so the state, the Hamiltonian
    MPO, and the environment tensors must all share the ``complex128`` dtype of
    the PyTorch backend before any effective-Hamiltonian contraction.

    Parameters
    ----------
    tensor:
        Nicole tensor with real or complex blocks.

    Returns
    -------
    Tensor
        Tensor with identical indices and itags but ``complex128`` block data.
    """
    return Tensor(
        indices=tensor.indices,
        itags=tensor.itags,
        data={key: block.to(torch.complex128) for key, block in tensor.data.items()},
        dtype=torch.complex128,
    )


def _norm(tensor: Tensor) -> float:
    """Return the Frobenius norm of a Nicole tensor as a real Python float."""
    value = tensor.norm()
    return float(value.real if hasattr(value, "real") else value)


def tensor_inner(left: Tensor, right: Tensor) -> complex:
    """Return the Hermitian inner product ``⟨left | right⟩`` of two tensors.

    Both tensors must share the same index structure; every axis is contracted
    between ``conj(left)`` and ``right``.

    Parameters
    ----------
    left, right:
        Nicole tensors with identical indices and itags.

    Returns
    -------
    complex
        The scalar ``⟨left | right⟩``.
    """
    rank = len(left.indices)
    axes = (list(range(rank)), list(range(rank)))
    return contract(conj(left), right, axes=axes).item()


def tensor_arnoldi_expv(
    apply: Callable[[Tensor], Tensor],
    tau: complex,
    x: Tensor,
    *,
    maxiter: int = 30,
    tol: float = 1e-15,
) -> Tensor:
    """Return ``exp(tau · A) x`` for a **non-Hermitian** tensor action ``A``.

    A tensor-native Arnoldi iteration: it builds an orthonormal Krylov basis of
    Nicole tensors and a small dense upper-Hessenberg matrix ``H`` by modified
    Gram–Schmidt, then forms ``y = β · V · exp(tau H) e₁``. Everything stays in
    the symmetry-blocked representation, so no amplitude is ever produced outside
    the admissible charge blocks. This is the non-Hermitian counterpart of
    :func:`tensor_lanczos_expv`, used for the discarded-projected K/L generators
    ``G = P⊥ · H`` (which are not Hermitian).

    Parameters
    ----------
    apply:
        Linear action ``A`` as a closure mapping a Nicole tensor to a tensor of
        the same index structure.
    tau:
        Scalar multiplying the generator inside the exponential (e.g.
        ``-1j * dt`` for real-time evolution).
    x:
        Tensor the exponential is applied to.
    maxiter:
        Maximum Krylov dimension (number of Arnoldi steps).
    tol:
        Early-stop tolerance on the residual norm of the next Krylov vector.

    Returns
    -------
    Tensor
        ``exp(tau · A) x`` with the same index structure as ``x``.
    """
    beta0 = _norm(x)
    if beta0 == 0.0:
        return x
    m = max(int(maxiter), 1)
    basis = [(1.0 / beta0) * x]
    # H[i, j] = ⟨basis[i] | A basis[j]⟩; the sub-diagonal H[j+1, j] is the residual
    # norm after orthogonalising A basis[j] against basis[0..j].
    hessenberg = torch.zeros((m, m), dtype=torch.complex128)
    used = 1
    for j in range(m):
        w = apply(basis[j])
        for i in range(j + 1):
            overlap = tensor_inner(basis[i], w)
            hessenberg[i, j] = overlap
            w = w + (-overlap) * basis[i]
        used = j + 1
        residual = _norm(w)
        if residual <= tol or j == m - 1:
            break
        hessenberg[j + 1, j] = residual
        basis.append((1.0 / residual) * w)

    coeff = torch.linalg.matrix_exp(tau * hessenberg[:used, :used])[:, 0] * beta0
    out = coeff[0] * basis[0]
    for idx in range(1, used):
        out = out + coeff[idx] * basis[idx]
    return out


def tensor_lanczos_expv(
    apply: Callable[[Tensor], Tensor],
    tau: complex,
    x: Tensor,
    *,
    maxiter: int = 30,
    tol: float = 1e-15,
) -> Tensor:
    """Return ``exp(tau · A) x`` for a **Hermitian** tensor action ``A``.

    A tensor-native symmetric Lanczos iteration: it builds an orthonormal Krylov
    basis of Nicole tensors and a small real-symmetric tridiagonal matrix
    ``T = tridiag(beta, alpha, beta)``, then forms ``y = β · V · exp(tau T) e₁``.
    Used for the discarded-BUG S-step, whose augmented-basis Galerkin generator
    is Hermitian.

    Parameters
    ----------
    apply:
        Hermitian linear action ``A`` as a closure mapping a Nicole tensor to a
        tensor of the same index structure.
    tau:
        Scalar multiplying the generator inside the exponential.
    x:
        Tensor the exponential is applied to.
    maxiter:
        Maximum Krylov dimension (number of Lanczos steps).
    tol:
        Early-stop tolerance on the off-diagonal ``beta`` (Krylov breakdown).

    Returns
    -------
    Tensor
        ``exp(tau · A) x`` with the same index structure as ``x``.
    """
    beta0 = _norm(x)
    if beta0 == 0.0:
        return x
    m = max(int(maxiter), 1)
    basis = [(1.0 / beta0) * x]
    alpha = torch.zeros(m, dtype=torch.complex128)
    beta = torch.zeros(m, dtype=torch.complex128)

    w = apply(basis[0])
    diag = tensor_inner(basis[0], w)
    alpha[0] = diag
    w = w + (-diag) * basis[0]
    used = 1
    for j in range(1, m):
        off = _norm(w)
        if off <= tol:
            break
        beta[j] = off
        basis.append((1.0 / off) * w)
        used = j + 1
        w = apply(basis[j])
        diag = tensor_inner(basis[j], w)
        alpha[j] = diag
        w = w + (-diag) * basis[j] + (-off) * basis[j - 1]

    tridiagonal = torch.zeros((used, used), dtype=torch.complex128)
    for i in range(used):
        tridiagonal[i, i] = alpha[i]
        if i + 1 < used:
            tridiagonal[i, i + 1] = beta[i + 1]
            tridiagonal[i + 1, i] = beta[i + 1]

    coeff = torch.linalg.matrix_exp(tau * tridiagonal)[:, 0] * beta0
    out = coeff[0] * basis[0]
    for idx in range(1, used):
        out = out + coeff[idx] * basis[idx]
    return out
