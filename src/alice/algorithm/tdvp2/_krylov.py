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


"""Self-contained Krylov exponential for the 2-site TDVP local substeps.

TDVP advances a site or bond tensor by ``exp(prefactor * dt * H_eff)`` where the
effective Hamiltonian ``H_eff`` is Hermitian and available only as a matrix-free
action on a Nicole `Tensor` (the contraction of the MPO environments with the MPO
site tensors). This module provides a Hermitian tensor Lanczos exponential for
exactly that situation, plus the evolution-prefactor context manager (``-1j`` for
real time, ``-1`` for imaginary time).

Keeping these helpers inside the ``tdvp2`` package makes the integrator depend
only on the shared `network` layer and the DMRG effective-Hamiltonian
contractions — no other algorithm package — so it can be reviewed and merged on
its own.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Iterable

import torch
from nicole import Tensor, conj as _nconj, einsum as _neinsum

_ACTIVE_PREFACTOR = [complex(0.0, -1.0)]


@contextlib.contextmanager
def with_time_prefactor(c: complex):
    """Temporarily set the global evolution prefactor.

    Parameters
    ----------
    c:
        New complex prefactor (``-1j`` for real time, ``-1`` for imaginary time).

    Returns
    -------
    contextlib._GeneratorContextManager
        A context manager that restores the previous prefactor on exit.
    """
    previous = _ACTIVE_PREFACTOR[0]
    _ACTIVE_PREFACTOR[0] = complex(c)
    try:
        yield
    finally:
        _ACTIVE_PREFACTOR[0] = previous


def active_time_prefactor() -> complex:
    """Return the currently active evolution prefactor.

    Returns
    -------
    complex
        The active complex prefactor set by :func:`with_time_prefactor`
        (default ``-1j``).
    """
    return _ACTIVE_PREFACTOR[0]


def to_complex(tensor: Tensor) -> Tensor:
    """Return a copy of ``tensor`` with every block cast to ``complex128``.

    Real-time evolution exponentiates the effective Hamiltonian, so the state, the
    MPO, and the environment boundary blocks must all share the ``complex128``
    backend dtype.

    Parameters
    ----------
    tensor:
        Nicole tensor with real or complex blocks.

    Returns
    -------
    Tensor
        Tensor with identical indices and itags but ``complex128`` block data.
    """
    new_intw = None
    if tensor.intw is not None:
        new_intw = {
            key: bridge.to(tensor.device, dtype=torch.complex128)
            for key, bridge in tensor.intw.items()
        }
    return Tensor(
        indices=tensor.indices,
        itags=tensor.itags,
        data={key: block.to(torch.complex128) for key, block in tensor.data.items()},
        intw=new_intw,
        dtype=torch.complex128,
    )


def _tensor_inner(a: Tensor, b: Tensor) -> complex:
    """Return the canonical inner product ``<a|b>`` for two same-shape tensors.

    Parameters
    ----------
    a:
        Left (bra) tensor.
    b:
        Right (ket) tensor with the same index structure as ``a``.

    Returns
    -------
    complex
        The scalar inner product ``sum(conj(a) * b)``.
    """
    equation = "".join(chr(97 + axis) for axis in range(len(a.itags)))
    return _neinsum(f"{equation},{equation}->", _nconj(a), b).item()


def _tridiagonal_exp_first_column(
    alpha: Iterable[float],
    beta: Iterable[float],
    dt: complex,
) -> torch.Tensor:
    """Return ``exp(dt * T) e_1`` for the Hermitian tridiagonal Lanczos matrix ``T``.

    Parameters
    ----------
    alpha:
        Diagonal entries of the tridiagonal matrix.
    beta:
        Off-diagonal entries (length ``len(alpha) - 1``).
    dt:
        Scalar prefactor in the exponential.

    Returns
    -------
    torch.Tensor
        The first column of ``exp(dt * T)``, as a complex vector of length
        ``len(alpha)``.
    """
    alpha_t = torch.as_tensor(tuple(alpha), dtype=torch.float64)
    beta_t = torch.as_tensor(tuple(beta), dtype=torch.float64)
    if alpha_t.numel() == 0:
        return torch.empty((0,), dtype=torch.complex128)
    tridiagonal = torch.diag(alpha_t)
    if beta_t.numel() > 0:
        tridiagonal = tridiagonal + torch.diag(beta_t, 1) + torch.diag(beta_t, -1)
    evals, evecs = torch.linalg.eigh(tridiagonal)
    evecs_c = evecs.to(torch.complex128)
    weights = torch.exp(dt * evals.to(torch.complex128)) * evecs_c[0, :]
    return evecs_c @ weights


# Opt-in Krylov-depth instrumentation (off by default => zero overhead). When
# enabled, every tensor_lanczos_expv call appends its Krylov dimension (number of
# matrix-free H applications) to KRYLOV_LOG, for the N_Krylov diagnostic.
KRYLOV_LOG: list[int] = []
_KRYLOV_RECORD = False


def enable_krylov_log() -> None:
    global _KRYLOV_RECORD
    _KRYLOV_RECORD = True
    KRYLOV_LOG.clear()


def disable_krylov_log() -> None:
    global _KRYLOV_RECORD
    _KRYLOV_RECORD = False


def get_krylov_log() -> list[int]:
    return list(KRYLOV_LOG)


def tensor_lanczos_expv(
    apply: Callable[[Tensor], Tensor],
    dt: complex,
    x: Tensor,
    *,
    maxiter: int = 30,
    tol: float = 1e-13,
) -> Tensor:
    """Return ``exp(dt * H) @ x`` for a Hermitian Nicole-tensor action ``apply``.

    Builds an orthonormal Krylov basis of Nicole tensors via the Hermitian Lanczos
    three-term recurrence, exponentiates the small tridiagonal projection, and
    recombines the basis. Everything stays in the symmetry-blocked Nicole
    representation; the operator is never materialised as a dense matrix.

    Parameters
    ----------
    apply:
        Matrix-free Hermitian action ``H`` on a Nicole tensor, returning a tensor
        with the same index structure as its input.
    dt:
        Scalar prefactor in the exponential (already including any evolution
        prefactor such as ``-1j``).
    x:
        Input Nicole tensor.
    maxiter:
        Maximum Krylov dimension (number of Lanczos steps).
    tol:
        Off-diagonal threshold at which the Lanczos recurrence terminates early.

    Returns
    -------
    Tensor
        The evolved tensor ``exp(dt * H) @ x`` with the same index structure as
        ``x``.
    """
    beta0 = x.norm()
    if float(abs(beta0)) == 0.0:
        if _KRYLOV_RECORD:
            KRYLOV_LOG.append(0)
        return x

    v = (1.0 / beta0) * x
    basis = [v]
    alpha: list[float] = []
    betas: list[float] = []

    w = apply(v)
    a = _tensor_inner(v, w).real
    alpha.append(a)
    w = w + (-a) * v

    for _ in range(1, maxiter):
        b = w.norm()
        if float(b) < tol:
            break
        betas.append(float(b))
        v = (1.0 / b) * w
        basis.append(v)
        w = apply(v)
        a = _tensor_inner(v, w).real
        alpha.append(a)
        w = w + (-a) * v + (-b) * basis[-2]

    if _KRYLOV_RECORD:
        KRYLOV_LOG.append(len(alpha))
    coeff = _tridiagonal_exp_first_column(alpha, betas, dt) * beta0
    evolved = coeff[0] * basis[0]
    for idx in range(1, len(alpha)):
        evolved = evolved + coeff[idx] * basis[idx]
    return evolved
