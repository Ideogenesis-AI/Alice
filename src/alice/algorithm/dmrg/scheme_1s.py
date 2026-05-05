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


"""1-site DMRG update scheme.

This module contains all operations that are specific to the 1-site update:

- `matvec`: apply the effective Hamiltonian `H_eff = E_left ⊗ W ⊗ E_right`
  to the center site tensor.
- `optimize_1site`: pure function that runs the Davidson solver to find the
  optimal site tensor given the surrounding environment.

The 2-site update scheme lives in `scheme_2s.py` and follows the same
interface. Future schemes (1-site-plus, etc.) each get their own
`scheme_*.py` file.

Index convention (shared with `environ.py`)
--------------------------------------------
- `a, b, c, d` — virtual bond indices of the MPS (bra or ket side)
- `o, p`       — virtual bond indices of the MPO
- `r, s`       — physical indices

Effective Hamiltonian contraction layout:

    E_left [a, o, b]    — (bra_left,  mpo_left,  ket_left)
    W      [o, p, r, s] — (mpo_left,  mpo_right, phys_bra, phys_ket)
    E_right[c, p, d]    — (bra_right, mpo_right, ket_right)
    M      [b, d, s]    — (ket_left,  ket_right, phys_ket)
    output [a, c, r]    — (bra_left,  bra_right, phys_bra)  ← same shape as M
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

from nicole import Tensor, einsum

from .davidson import davidson


def matvec(M: Tensor, W: Tensor, E_left: Tensor, E_right: Tensor) -> Tensor:
    """Apply the 1-site effective Hamiltonian H_eff to the site tensor M.

    Computes `H_eff|M⟩` as a single four-tensor einsum contraction.

    Parameters
    ----------
    M:
        Center site tensor with axes `(ket_left, ket_right, phys_ket)`.
    W:
        MPO site tensor with axes `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    E_left:
        Left environment with axes `(bra_left, mpo_left, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_right, ket_right)`.

    Returns
    -------
    Tensor
        `H_eff|M⟩` with axes `(bra_left, bra_right, phys_bra)`, identical
        in shape to `M`.
    """
    # 'aob,bds,oprs,cpd->acr'
    # contracted: b (ket_left), d (ket_right), s (phys_ket), o (mpo_left), p (mpo_right)
    # free:       a (bra_left), c (bra_right), r (phys_bra)
    # Contraction order: (1) E_left × M on b → (a,o,d,s);
    #                    (2) × W on o,s        → (a,d,p,r);
    #                    (3) × E_right on d,p  → (a,c,r).
    # Starting with E_left × M eliminates the large MPS virtual bond first,
    # keeping intermediates small.
    return einsum('aob,bds,oprs,cpd->acr', E_left, M, W, E_right)


def optimize_1site(
    M: Tensor,
    W: Tensor,
    E_left: Tensor,
    E_right: Tensor,
    davidson_opts: dict,
) -> Tuple[float, Tensor, float]:
    """Find the optimal center site tensor via the Davidson eigensolver.

    This is a pure function: it takes tensors and returns tensors without
    mutating any MPS state. Being pure, it is the natural point for future
    parallel or distributed dispatch — the sweep orchestrator can in principle
    hand this call off to a remote worker.

    Parameters
    ----------
    M:
        Current center site tensor with axes `(ket_left, ket_right, phys_ket)`,
        used as the initial guess for Davidson.
    W:
        MPO site tensor with axes `(mpo_left, mpo_right, phys_bra, phys_ket)`.
    E_left:
        Left environment with axes `(bra_left, mpo_left, ket_left)`.
    E_right:
        Right environment with axes `(bra_right, mpo_right, ket_right)`.
    davidson_opts:
        Keyword arguments forwarded to `davidson`: `max_iter`, `tol`,
        `max_subspace`.

    Returns
    -------
    float
        Variational energy estimate (lowest Ritz value).
    Tensor
        Optimised site tensor `M_opt` with the same axis layout as `M`.
    float
        Final Davidson residual norm at convergence (or at exit if not
        converged).
    """
    # Bind the environment tensors so the caller only passes the site tensor.
    mv = partial(matvec, E_left=E_left, W=W, E_right=E_right)
    energy, M_opt, davidson_error = davidson(mv, M, **davidson_opts)

    return energy, M_opt, davidson_error
