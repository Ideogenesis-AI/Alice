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

"""Krylov and micro-step helpers for dense vectors and Nicole tensors.

The original Julia code threaded method names, tolerances, iteration caps, and
backend choices through many low-level calls. This module keeps the numerical
behavior but packages the runtime controls into small option objects so the
higher-level BUG code can call it in a more readable way.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Iterable

import torch
from nicole import Tensor, conj as _nconj, einsum as _neinsum

from .nicole_helpers import flatten_fortran, to_dense

BUG_DEFAULT_EXPV_BACKEND = "krylovkit"
BUG_ALLOWED_EXPV_BACKENDS = ("krylovkit", "native_hermitian_lanczos")
_ACTIVE_BACKEND = [BUG_DEFAULT_EXPV_BACKEND]
_ACTIVE_PREFACTOR = [complex(0.0, -1.0)]


@dataclass(frozen=True)
class LanczosOptions:
    """Options for Hermitian Lanczos exponentiation.

    Args:
        tol: Termination tolerance for the Lanczos recurrence.
        krylovdim: Maximum Krylov basis size.
    """

    tol: float = 1e-13
    krylovdim: int = 60


@dataclass(frozen=True)
class LinearSubstepOptions:
    """Options for one dense or matrix-free linear micro-step.

    Args:
        method: Local integrator name. Supported values are ``"expv"``,
            ``"euler"``, and ``"rk4"``.
        lanczos_tol: Tolerance passed to Lanczos-based ``expv`` routines.
        lanczos_maxiter: Maximum Lanczos basis size.
        restart: Reserved compatibility flag retained from the Julia API.
        issymmetric: Optional hint used by :func:`general_linear_substep`.
    """

    method: str = "expv"
    lanczos_tol: float = 1e-13
    lanczos_maxiter: int = 60
    restart: int = 1
    issymmetric: bool = True

    def lanczos_options(self) -> LanczosOptions:
        """Return the matching :class:`LanczosOptions` view.

        Returns:
            A :class:`LanczosOptions` instance derived from the substep fields.
        """
        return LanczosOptions(tol=self.lanczos_tol, krylovdim=self.lanczos_maxiter)


_LANCZOS_OPTION_FIELDS = {field.name for field in LanczosOptions.__dataclass_fields__.values()}
_SUBSTEP_OPTION_FIELDS = {field.name for field in LinearSubstepOptions.__dataclass_fields__.values()}
_LANCZOS_ALIASES = {"maxiter": "krylovdim"}
_SUBSTEP_ALIASES = {"tol": "lanczos_tol", "maxiter": "lanczos_maxiter", "krylovdim": "lanczos_maxiter"}

__all__ = [
    "BUG_ALLOWED_EXPV_BACKENDS",
    "BUG_DEFAULT_EXPV_BACKEND",
    "LanczosOptions",
    "LinearSubstepOptions",
    "active_expv_backend",
    "active_time_prefactor",
    "complex_tensor_array",
    "complex_tensor_vec",
    "general_linear_substep",
    "hermitian_tridiagonal_exp_coeffs",
    "linear_substep",
    "native_hermitian_lanczos_exponentiate",
    "tensor_inner",
    "tensor_lanczos_expv",
    "with_expv_backend",
    "with_time_prefactor",
]


def _coerce_lanczos_options(options: LanczosOptions | None = None, **kwargs: Any) -> LanczosOptions:
    """Normalize Lanczos options from an object or legacy kwargs.

    Args:
        options: Existing :class:`LanczosOptions` instance.
        **kwargs: Field overrides or legacy aliases such as ``maxiter``.

    Returns:
        A normalized :class:`LanczosOptions` instance.
    """
    overrides: dict[str, Any] = {}
    for name, value in kwargs.items():
        normalized = _LANCZOS_ALIASES.get(name, name)
        if normalized not in _LANCZOS_OPTION_FIELDS:
            raise TypeError(f"Unknown Lanczos option: {name}")
        overrides[normalized] = value
    if options is None:
        return LanczosOptions(**overrides)
    return replace(options, **overrides)


def _coerce_substep_options(options: LinearSubstepOptions | None = None, **kwargs: Any) -> LinearSubstepOptions:
    """Normalize linear-substep options from an object or legacy kwargs.

    Args:
        options: Existing :class:`LinearSubstepOptions` instance.
        **kwargs: Field overrides or legacy aliases such as ``tol``.

    Returns:
        A normalized :class:`LinearSubstepOptions` instance.
    """
    overrides: dict[str, Any] = {}
    for name, value in kwargs.items():
        normalized = _SUBSTEP_ALIASES.get(name, name)
        if normalized not in _SUBSTEP_OPTION_FIELDS:
            raise TypeError(f"Unknown linear-substep option: {name}")
        overrides[normalized] = value
    if options is None:
        return LinearSubstepOptions(**overrides)
    return replace(options, **overrides)


def _as_complex_tensor(x) -> torch.Tensor:
    """Convert input data to a complex128 torch tensor.

    Args:
        x: Tensor-like object.

    Returns:
        A ``torch.complex128`` tensor.
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.complex128)
    return torch.as_tensor(x, dtype=torch.complex128)


def hermitian_tridiagonal_exp_coeffs(
    alpha: Iterable[float] | torch.Tensor,
    beta: Iterable[float] | torch.Tensor,
    dt: complex,
) -> torch.Tensor:
    """Compute the Krylov coefficients for ``exp(dt*T)e1``.

    Args:
        alpha: Diagonal entries of the Hermitian tridiagonal matrix.
        beta: Off-diagonal entries.
        dt: Scalar prefactor used in the exponential.

    Returns:
        The coefficient vector in the Lanczos basis.
    """
    alpha_t = torch.as_tensor(tuple(alpha) if not isinstance(alpha, torch.Tensor) else alpha, dtype=torch.float64)
    beta_t = torch.as_tensor(tuple(beta) if not isinstance(beta, torch.Tensor) else beta, dtype=torch.float64)
    if alpha_t.numel() == 0:
        return torch.empty((0,), dtype=torch.complex128)

    tridiagonal = torch.diag(alpha_t)
    if beta_t.numel() > 0:
        tridiagonal = tridiagonal + torch.diag(beta_t, diagonal=1) + torch.diag(beta_t, diagonal=-1)
    evals, evecs = torch.linalg.eigh(tridiagonal)
    evecs_c = evecs.to(torch.complex128)
    weights = torch.exp(dt * evals.to(torch.complex128)) * evecs_c[0, :]
    return evecs_c @ weights


def native_hermitian_lanczos_exponentiate(
    matvec: Callable[[torch.Tensor], torch.Tensor | object],
    dt: complex,
    x,
    *,
    options: LanczosOptions | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, int]:
    """Apply ``exp(dt * H)`` to ``x`` using a native Hermitian Lanczos solve.

    Args:
        matvec: Matrix-free Hermitian action on dense vectors.
        dt: Scalar prefactor used in the exponential.
        x: Input vector.
        options: Optional :class:`LanczosOptions` instance.
        **kwargs: Legacy option overrides such as ``tol=...``.

    Returns:
        A pair ``(y, numops)`` containing the evolved vector and the number of
        matrix-vector products performed.
    """
    options = _coerce_lanczos_options(options, **kwargs)
    x_work = _as_complex_tensor(x).reshape(-1).clone()
    n = int(x_work.numel())
    if n == 0:
        return torch.empty((0,), dtype=torch.complex128), 0

    norm_x = torch.linalg.norm(x_work)
    if norm_x == 0:
        return torch.zeros_like(x_work), 0

    mmax = min(max(int(options.krylovdim), 1), n)
    basis = torch.empty((n, mmax), dtype=torch.complex128)
    alpha = torch.empty((mmax,), dtype=torch.float64)
    beta = torch.empty((max(mmax - 1, 0),), dtype=torch.float64)

    basis[:, 0] = x_work / norm_x
    numops = 0
    final_dim = 1

    # Standard Hermitian Lanczos recurrence on dense vectors.
    for j in range(mmax):
        vj = basis[:, j]
        work = _as_complex_tensor(matvec(vj)).reshape(-1)
        numops += 1

        if j > 0:
            work = work - beta[j - 1] * basis[:, j - 1]

        alpha[j] = torch.real(torch.vdot(vj, work))
        work = work - alpha[j] * vj

        if j == mmax - 1:
            final_dim = j + 1
            break

        beta_j = torch.linalg.norm(work)
        if float(beta_j) <= options.tol:
            final_dim = j + 1
            break

        beta[j] = beta_j.real
        basis[:, j + 1] = work / beta_j
        final_dim = j + 2

    coeff = hermitian_tridiagonal_exp_coeffs(alpha[:final_dim], beta[: max(final_dim - 1, 0)], dt)
    y = norm_x.to(torch.complex128) * (basis[:, :final_dim] @ coeff)
    return y, numops


@contextlib.contextmanager
def with_expv_backend(backend: str):
    """Temporarily set the active expv backend name.

    Args:
        backend: Backend label from :data:`BUG_ALLOWED_EXPV_BACKENDS`.

    Returns:
        A context manager that restores the previous backend on exit.
    """
    if backend not in BUG_ALLOWED_EXPV_BACKENDS:
        raise ValueError(f"Unknown expv backend: {backend}")
    previous = _ACTIVE_BACKEND[0]
    _ACTIVE_BACKEND[0] = backend
    try:
        yield
    finally:
        _ACTIVE_BACKEND[0] = previous


def active_expv_backend() -> str:
    """Return the currently active expv backend label.

    Returns:
        The active backend name.
    """
    return _ACTIVE_BACKEND[0]


@contextlib.contextmanager
def with_time_prefactor(c: complex):
    """Temporarily override the global evolution prefactor.

    Args:
        c: New complex prefactor.

    Returns:
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

    Returns:
        The active complex prefactor.
    """
    return _ACTIVE_PREFACTOR[0]


def _matrix_linear_substep(
    H,
    dt: complex,
    x,
    *,
    options: LinearSubstepOptions,
) -> tuple[torch.Tensor, int]:
    """Apply one micro-step when the operator is available as a dense matrix.

    Args:
        H: Dense matrix.
        dt: Step size or exponential prefactor.
        x: Input vector.
        options: Linear-substep configuration.

    Returns:
        A pair ``(y, numops)`` describing the updated vector and the number of
        explicit matvecs counted for Krylov methods.
    """
    H_dense = _as_complex_tensor(H)
    x_dense = _as_complex_tensor(x).reshape(-1)
    if options.method == "expv":
        if active_expv_backend() == "native_hermitian_lanczos":
            return native_hermitian_lanczos_exponentiate(
                lambda v: H_dense @ v,
                dt,
                x_dense,
                options=options.lanczos_options(),
            )
        return torch.linalg.matrix_exp(dt * H_dense) @ x_dense, 0

    return linear_substep(
        lambda v: H_dense @ v,
        dt,
        x_dense,
        options=options,
    )


def linear_substep(
    H_or_matvec,
    dt: complex,
    x,
    *,
    options: LinearSubstepOptions | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, int]:
    """Advance one dense or matrix-free micro-step.

    Args:
        H_or_matvec: Dense matrix or matrix-free action.
        dt: Step size or exponential prefactor.
        x: Input vector.
        options: Optional :class:`LinearSubstepOptions` instance.
        **kwargs: Legacy option overrides such as ``method="expv"``.

    Returns:
        A pair ``(y, numops)`` containing the updated vector and the counted
        operator applications.
    """
    options = _coerce_substep_options(options, **kwargs)
    if callable(H_or_matvec):
        x_vec = _as_complex_tensor(x).reshape(-1)
        matvec = H_or_matvec

        if options.method == "expv":
            return native_hermitian_lanczos_exponentiate(matvec, dt, x_vec, options=options.lanczos_options())

        if options.method == "euler":
            return x_vec + dt * _as_complex_tensor(matvec(x_vec)).reshape(-1), 1

        if options.method == "rk4":
            # Keep the explicit stages readable; the dimensions are tiny compared
            # with the conceptual cost of understanding hidden helper machinery.
            k1 = _as_complex_tensor(matvec(x_vec)).reshape(-1)
            k2 = _as_complex_tensor(matvec(x_vec + (dt / 2) * k1)).reshape(-1)
            k3 = _as_complex_tensor(matvec(x_vec + (dt / 2) * k2)).reshape(-1)
            k4 = _as_complex_tensor(matvec(x_vec + dt * k3)).reshape(-1)
            return x_vec + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4), 4

        raise ValueError(f"Unknown substep method: {options.method}. Supported: expv, euler, rk4.")

    return _matrix_linear_substep(H_or_matvec, dt, x, options=options)


def general_linear_substep(
    matvec,
    dt: complex,
    x,
    *,
    options: LinearSubstepOptions | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, int]:
    """Variant of :func:`linear_substep` with explicit symmetry dispatch.

    Args:
        matvec: Matrix-free operator action.
        dt: Step size or exponential prefactor.
        x: Input vector.
        options: Optional :class:`LinearSubstepOptions` instance.
        **kwargs: Legacy option overrides such as ``issymmetric=False``.

    Returns:
        A pair ``(y, numops)`` containing the updated vector and the counted
        operator applications.
    """
    options = _coerce_substep_options(options, **kwargs)
    x_vec = _as_complex_tensor(x).reshape(-1)
    if options.method != "expv":
        return linear_substep(matvec, dt, x_vec, options=options)

    if options.issymmetric:
        return native_hermitian_lanczos_exponentiate(
            matvec,
            dt,
            x_vec,
            options=LanczosOptions(
                tol=options.lanczos_tol,
                krylovdim=min(len(x_vec), max(options.lanczos_maxiter, 4)),
            ),
        )

    # The nonsymmetric fallback is intentionally dense because the current port
    # only needs it for very small diagnostic problems.
    n = len(x_vec)
    eye = torch.eye(n, dtype=torch.complex128)
    dense = torch.empty((n, n), dtype=torch.complex128)
    for col in range(n):
        dense[:, col] = _as_complex_tensor(matvec(eye[:, col])).reshape(-1)
    return torch.linalg.matrix_exp(dt * dense) @ x_vec, n


def tensor_inner(a: Tensor, b: Tensor) -> complex:
    """Return the canonical inner product ``<a|b>`` for same-shape tensors.

    Args:
        a: Left tensor.
        b: Right tensor.

    Returns:
        The complex scalar inner product.
    """
    equation = "".join(chr(97 + axis) for axis in range(len(a.itags)))
    return _neinsum(f"{equation},{equation}->", _nconj(a), b).item()


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
    options: LanczosOptions | None = None,
    **kwargs: Any,
) -> Tensor:
    """Return ``exp(dt * H) @ x`` for Hermitian Nicole tensor actions.

    Args:
        apply: Matrix-free Hermitian action on Nicole tensors.
        dt: Scalar prefactor used in the exponential.
        x: Input Nicole tensor.
        options: Optional :class:`LanczosOptions` instance.
        **kwargs: Legacy option overrides such as ``maxiter=60``.

    Returns:
        The evolved Nicole tensor.
    """
    options = _coerce_lanczos_options(options, **kwargs)
    beta0 = x.norm()
    if beta0 == 0:
        if _KRYLOV_RECORD:
            KRYLOV_LOG.append(0)
        return x

    v = (1.0 / beta0) * x
    basis = [v]
    alpha: list[float] = []
    betas: list[float] = []

    w = apply(v)
    a = tensor_inner(v, w).real
    alpha.append(a)
    w = w + (-a) * v

    # This is the same Hermitian recurrence as the dense version, but each basis
    # vector is now a Nicole tensor instead of a flat torch vector.
    for _ in range(1, options.krylovdim):
        b = w.norm()
        if float(b) < options.tol:
            break
        betas.append(float(b))
        v = (1.0 / b) * w
        basis.append(v)
        w = apply(v)
        a = tensor_inner(v, w).real
        alpha.append(a)
        w = w + (-a) * v + (-b) * basis[-2]

    if _KRYLOV_RECORD:
        KRYLOV_LOG.append(len(alpha))
    coeff = hermitian_tridiagonal_exp_coeffs(alpha, betas, dt) * beta0
    out = coeff[0] * basis[0]
    for idx in range(1, len(alpha)):
        out = out + coeff[idx] * basis[idx]
    return out


def complex_tensor_array(tensor: Tensor, itag_order):
    """Convert a Nicole tensor to a dense complex128 torch array.

    Args:
        tensor: Nicole tensor to densify.
        itag_order: Tag order passed to :func:`bug_nicole.nicole_helpers.to_dense`.

    Returns:
        A dense ``torch.complex128`` tensor.
    """
    return to_dense(tensor, itag_order).to(dtype=torch.complex128)


def complex_tensor_vec(tensor: Tensor, itag_order):
    """Flatten :func:`complex_tensor_array` in Fortran/column-major order.

    Args:
        tensor: Nicole tensor to flatten.
        itag_order: Tag order used for densification.

    Returns:
        A one-dimensional dense vector.
    """
    return flatten_fortran(complex_tensor_array(tensor, itag_order))
