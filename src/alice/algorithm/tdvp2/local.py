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


"""Local real-time substeps for the 2-site TDVP integrator.

2-site TDVP advances the orthogonality window by exponentiating the *effective
Hamiltonian* — the MPS bond tensor evolved under ``exp(prefactor·dt·H_eff)`` with
the left/right MPO environments held fixed — rather than applying a pre-formed
gate. The two effective Hamiltonians are exactly the DMRG ones, so this module
reuses the DMRG contractions verbatim:

- the 2-site action ``H_eff^{(2)}`` is :func:`alice.algorithm.dmrg.scheme_2s.matvec_2s`
  (``E_left · W_i · W_{i+1} · E_right`` applied to Θ), and
- the 1-site action ``H_eff^{(1)}`` is :func:`alice.algorithm.dmrg.scheme_1s.matvec`,

each fed to the Hermitian tensor Lanczos exponential in this package's
:mod:`alice.algorithm.tdvp2._krylov` (``tensor_lanczos_expv``). H_eff is Hermitian,
so the Lanczos path is the right one; the evolution prefactor (``-1j`` real time,
``-1`` imaginary) is the active prefactor of that module.

The forward sweep evolves each 2-site block forward by ``dt`` then evolves the
carried 1-site tensor *backward* by ``dt`` (the inverse-free single-site
correction that prevents double counting the shared bond); the reverse sweep
mirrors it. A symmetric (Strang) step composes a forward half-sweep and a reverse
half-sweep.
"""

from __future__ import annotations

from functools import partial

from nicole import Tensor

from ..dmrg.scheme_1s import matvec as _matvec_1s
from ..dmrg.scheme_2s import matvec_2s as _matvec_2s
from ._krylov import active_time_prefactor, tensor_lanczos_expv


def evolve_two_site(
    theta: Tensor,
    W_i: Tensor,
    W_i1: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    dt: complex,
    *,
    lanczos_tol: float,
    lanczos_maxiter: int,
) -> Tensor:
    """Return ``exp(prefactor·dt·H_eff^{(2)})|Θ⟩`` for the 2-site bond tensor Θ.

    Parameters
    ----------
    theta:
        Bond tensor with axes ``(ket_left, ket_right, phys_ket_i, phys_ket_{i+1})``.
    W_i, W_i1:
        MPO tensors at sites ``i`` and ``i+1``.
    E_left, E_right:
        Left/right MPO environments bracketing the two-site window.
    dt:
        Real time advanced by this substep (multiplied by the active evolution
        prefactor internally).
    lanczos_tol, lanczos_maxiter:
        Lanczos termination tolerance and maximum Krylov dimension.
    """
    mv = partial(_matvec_2s, W_i=W_i, W_i1=W_i1, E_left=E_left, E_right=E_right)
    return tensor_lanczos_expv(
        mv, active_time_prefactor() * dt, theta,
        maxiter=lanczos_maxiter, tol=lanczos_tol,
    )


def evolve_one_site(
    site: Tensor,
    W: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    dt: complex,
    *,
    lanczos_tol: float,
    lanczos_maxiter: int,
) -> Tensor:
    """Return ``exp(prefactor·dt·H_eff^{(1)})|M⟩`` for the 1-site tensor M.

    Used with a *negative* ``dt`` for the TDVP backward correction on the carried
    bond tensor between two 2-site updates.

    Parameters
    ----------
    site:
        Center site tensor with axes ``(ket_left, ket_right, phys_ket)``.
    W:
        MPO tensor at that site.
    E_left, E_right:
        Left/right MPO environments bracketing the site.
    dt:
        Real time advanced by this substep (multiplied by the active evolution
        prefactor internally). The caller passes ``-tau`` for the backward step.
    lanczos_tol, lanczos_maxiter:
        Lanczos termination tolerance and maximum Krylov dimension.
    """
    mv = partial(_matvec_1s, W=W, E_left=E_left, E_right=E_right)
    return tensor_lanczos_expv(
        mv, active_time_prefactor() * dt, site,
        maxiter=lanczos_maxiter, tol=lanczos_tol,
    )
