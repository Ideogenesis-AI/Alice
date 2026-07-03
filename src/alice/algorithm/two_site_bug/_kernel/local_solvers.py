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


"""Pluggable local time-integrators for the (imaginary-time) BUG substeps.

Every BUG local substep computes ``y = exp(tau * A) x`` for a matrix-free tensor
action ``A`` (``apply``) and a (generally complex) local timestep ``tau``. In
**unitary** real-time evolution this must be an *exact* exponential, so the
faithful kernel uses a Krylov ``expv``. In **imaginary time** (cooling toward the
ground state) the evolution is no longer unitary, and ``y = exp(tau A) x`` is just
the exact flow of the linear ODE ``x'(s) = A x(s)`` over ``s in [0, tau]`` — *any*
stable integrator of that ODE may be used. This module provides a family of them
behind one uniform ``(apply, tau, x)`` call surface so the discarded-projector BUG
and the two-site BUG (``variant='discarded'``) can swap the local solver:

  * ``'krylov'``    : Lanczos (Hermitian) / Arnoldi (general) exponential — ``≈`` exact.
  * ``'midpoint'``  : explicit midpoint (RK2), ``substeps`` internal steps — 2nd order,
                      explicit (fast), conditionally stable.
  * ``'rk4'``       : classical Runge–Kutta 4, ``substeps`` internal steps — 4th order,
                      explicit, conditionally stable.
  * ``'trapezoid'`` : implicit trapezoidal / Crank–Nicolson, ``substeps`` internal steps —
                      2nd order, A-stable (unconditionally stable), one matrix-free
                      linear solve per substep.

For the substepped integrators the local exponential error is ``O((tau/substeps)^p)``
(``p = 2`` for midpoint/trapezoid, ``p = 4`` for rk4); raising ``substeps`` converges
monotonically to the exact action (and, for the two explicit schemes, also restores
stability when ``|tau| * ||A||`` is large). Everything stays in the symmetry-blocked
Nicole tensor representation (only ``apply``, tensor add/scale, and inner products are
used), so no admissible-block structure is ever broken.
"""

from __future__ import annotations

import torch
from nicole import Tensor

from .krylov import tensor_inner, tensor_lanczos_expv

# Solver names accepted by :func:`local_expv`.
LOCAL_SOLVERS = ('krylov', 'midpoint', 'rk4', 'trapezoid')


def _norm(x: Tensor) -> float:
    n = x.norm()
    return float(n.real if hasattr(n, 'real') else n)


# ---------------------------------------------------------------------------
# Krylov (general / non-Hermitian) exponential — the Arnoldi counterpart of the
# Hermitian tensor Lanczos in :mod:`.krylov`.
# ---------------------------------------------------------------------------

def tensor_arnoldi_expv(apply, tau: complex, x: Tensor, *, maxiter: int = 30, tol: float = 1e-15) -> Tensor:
    """Return ``exp(tau * A) @ x`` for a NON-Hermitian Nicole-tensor action ``apply``.

    A tensor-native Arnoldi (modified Gram–Schmidt) exponential: builds an
    orthonormal Krylov basis of Nicole tensors and a small dense upper-Hessenberg
    matrix ``H``, then forms ``y = beta * V * exp(tau H) e1``. Stays in the
    symmetry-blocked representation throughout (the non-Hermitian counterpart of
    :func:`alice.algorithm.two_site_bug._kernel.krylov.tensor_lanczos_expv`).
    """
    beta0 = _norm(x)
    if beta0 == 0.0:
        return x
    m = max(int(maxiter), 1)
    basis = [(1.0 / beta0) * x]
    H = torch.zeros((m, m), dtype=torch.complex128)
    used = 1
    for j in range(m):
        w = apply(basis[j])
        for i in range(j + 1):
            hij = tensor_inner(basis[i], w)
            H[i, j] = hij
            w = w + (-hij) * basis[i]
        used = j + 1
        nrm = _norm(w)
        if nrm <= tol or j == m - 1:
            break
        H[j + 1, j] = nrm
        basis.append((1.0 / nrm) * w)

    Hk = H[:used, :used]
    coeff = torch.linalg.matrix_exp(tau * Hk)[:, 0] * beta0
    out = coeff[0] * basis[0]
    for idx in range(1, used):
        out = out + coeff[idx] * basis[idx]
    return out


# ---------------------------------------------------------------------------
# Explicit Runge–Kutta exponential actions (midpoint / RK4)
# ---------------------------------------------------------------------------

def tensor_rk_expv(apply, tau: complex, x: Tensor, *, order: int, substeps: int) -> Tensor:
    """Approximate ``exp(tau A) x`` by explicit RK integration of ``x' = A x``.

    Integrates the linear ODE over ``s in [0, tau]`` with ``substeps`` equal steps
    ``h = tau / substeps``. ``order=2`` is the explicit midpoint rule (RK2),
    ``order=4`` the classical RK4. Explicit and matrix-free (only ``apply`` and
    tensor arithmetic), so it is fast but conditionally stable: the local error is
    ``O(h^order)`` and stability needs ``|h| * ||A||`` inside the method's stability
    region — raise ``substeps`` if either is violated.
    """
    n = max(int(substeps), 1)
    h = tau / n
    y = x
    if order == 2:
        for _ in range(n):
            k1 = apply(y)
            k2 = apply(y + (0.5 * h) * k1)
            y = y + h * k2
    elif order == 4:
        for _ in range(n):
            k1 = apply(y)
            k2 = apply(y + (0.5 * h) * k1)
            k3 = apply(y + (0.5 * h) * k2)
            k4 = apply(y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        raise ValueError(f"tensor_rk_expv: unsupported order {order!r} (use 2 or 4)")
    return y


# ---------------------------------------------------------------------------
# Implicit trapezoidal (Crank–Nicolson) exponential action + matrix-free GMRES
# ---------------------------------------------------------------------------

def tensor_gmres(linop, b: Tensor, *, tol: float = 1e-12, maxiter: int = 60) -> Tensor:
    """Solve ``linop(y) = b`` for a matrix-free Nicole-tensor linear operator.

    Full (non-restarted) GMRES from the zero initial guess, working entirely in the
    symmetry-blocked tensor representation (Arnoldi + a small dense least-squares on
    the Hessenberg matrix). The local systems here are tiny, so a handful of
    iterations reach ``tol``; ``maxiter`` is the hard cap.
    """
    bnorm = _norm(b)
    if bnorm == 0.0:
        return b
    m = max(int(maxiter), 1)
    V = [(1.0 / bnorm) * b]
    H = torch.zeros((m + 1, m), dtype=torch.complex128)
    e1 = torch.zeros(m + 1, dtype=torch.complex128)
    e1[0] = bnorm
    for j in range(m):
        w = linop(V[j])
        for i in range(j + 1):
            H[i, j] = tensor_inner(V[i], w)
            w = w + (-H[i, j]) * V[i]
        hjj = _norm(w)
        H[j + 1, j] = hjj
        # Least-squares solve of the (j+2, j+1) Hessenberg system for the residual.
        y, *_ = torch.linalg.lstsq(H[:j + 2, :j + 1], e1[:j + 2].unsqueeze(1))
        y = y.squeeze(1)
        resid = float(torch.linalg.norm(e1[:j + 2] - H[:j + 2, :j + 1] @ y).real)
        if hjj <= tol * bnorm or resid <= tol * bnorm or j == m - 1:
            sol = y[0] * V[0]
            for idx in range(1, j + 1):
                sol = sol + y[idx] * V[idx]
            return sol
        V.append((1.0 / hjj) * w)
    # Unreachable: the loop always returns at j == m - 1.
    raise RuntimeError("tensor_gmres did not return")


def tensor_trapezoid_expv(apply, tau: complex, x: Tensor, *, substeps: int,
                          tol: float = 1e-12, maxiter: int = 60) -> Tensor:
    """Approximate ``exp(tau A) x`` by the implicit trapezoidal rule (Crank–Nicolson).

    Each of the ``substeps`` steps ``h = tau / substeps`` advances
    ``(I - (h/2) A) y_{k+1} = (I + (h/2) A) y_k`` — the (1,1)-Padé approximant of
    ``exp(h A)``. It is 2nd order and A-stable (the stability function maps the left
    half-plane into the unit disc), so it never blows up however large ``|h| * ||A||``
    is; raising ``substeps`` drives the ``O(h^2)`` error down. The implicit solve is
    a matrix-free GMRES (:func:`tensor_gmres`).
    """
    n = max(int(substeps), 1)
    h = tau / n
    c = 0.5 * h
    y = x
    for _ in range(n):
        rhs = y + c * apply(y)                       # (I + (h/2) A) y_k
        y = tensor_gmres(lambda z: z + (-c) * apply(z), rhs, tol=tol, maxiter=maxiter)
    return y


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def local_expv(apply, tau: complex, x: Tensor, *, solver: str = 'krylov', substeps: int = 1,
               hermitian: bool = False, krylov_maxiter: int = 30, krylov_tol: float = 1e-15) -> Tensor:
    """Compute ``exp(tau A) x`` with the requested local integrator.

    Parameters
    ----------
    apply:
        Matrix-free tensor action ``x -> A x``.
    tau:
        Local timestep (already including the evolution prefactor, e.g. ``-dt`` for
        imaginary time, ``-1j*dt`` for real time).
    x:
        Input tensor.
    solver:
        One of :data:`LOCAL_SOLVERS`.
    substeps:
        Number of internal steps for the substepped integrators (ignored by
        ``'krylov'``).
    hermitian:
        Whether ``A`` is Hermitian — selects Lanczos vs Arnoldi for ``'krylov'``
        (ignored by the other solvers).
    krylov_maxiter, krylov_tol:
        Krylov dimension cap and tolerance for ``'krylov'`` (also used as the GMRES
        tolerance / cap for ``'trapezoid'``).
    """
    if solver == 'krylov':
        if hermitian:
            return tensor_lanczos_expv(apply, tau, x, maxiter=krylov_maxiter, tol=krylov_tol)
        return tensor_arnoldi_expv(apply, tau, x, maxiter=krylov_maxiter, tol=krylov_tol)
    if solver == 'midpoint':
        return tensor_rk_expv(apply, tau, x, order=2, substeps=substeps)
    if solver == 'rk4':
        return tensor_rk_expv(apply, tau, x, order=4, substeps=substeps)
    if solver == 'trapezoid':
        return tensor_trapezoid_expv(apply, tau, x, substeps=substeps,
                                     tol=krylov_tol, maxiter=max(krylov_maxiter, 60))
    raise ValueError(f"unknown local solver {solver!r}; recognised values are: {', '.join(LOCAL_SOLVERS)}")
