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


"""1-site variational MPO-MPO compression update.

At each site `i`, the optimal `C_i` that minimises the local contribution to
`‖C − A·B‖²_F` (given all other sites of C are fixed) is obtained by a
direct tensor contraction — no eigensolver is needed:

    C_i = einsum('cab, aprx, bqxs, dpq -> cdrs', E_left, A_i, B_i, E_right)

Index convention (shared with `environ.py`):

- `c`, `d` — left and right bond of C at site i
- `a`, `p` — left and right bond of A at site i
- `b`, `q` — left and right bond of B at site i
- `r`       — phys_in  (A axis 2 = C axis 2)
- `x`       — phys_internal (A axis 3 = B axis 2, contracted)
- `s`       — phys_out (B axis 3 = C axis 3)
"""

from __future__ import annotations

from nicole import Tensor, einsum


def local_update_1s(
    E_left: Tensor,
    A_i: Tensor,
    B_i: Tensor,
    E_right: Tensor,
) -> Tensor:
    """Compute the optimal 1-site update for the compressed MPO C at site `i`.

    Given the current left and right environments and the factor tensors A and B
    at site i, returns the C_i that minimises the Frobenius-norm residual

        ‖C − A·B‖²_F

    subject to C being in mixed canonical form with orthogonality center at i.
    This is a direct contraction with no eigensolver.

    Parameters
    ----------
    E_left:
        Left environment with axes `(c, a, b)` = `(C_left, A_left, B_left)`.
    A_i:
        Factor MPO A at site i with axes `(left, right, phys_in, phys_internal)`.
    B_i:
        Factor MPO B at site i with axes `(left, right, phys_internal_in, phys_out)`.
    E_right:
        Right environment with axes `(d, p, q)` = `(C_right, A_right, B_right)`.

    Returns
    -------
    Tensor
        New `C_i` with axes `(C_left, C_right, phys_in, phys_out)`.
    """
    # Decomposed into sequential 2-tensor einsums to avoid itag-matching ambiguity
    # when A, B, C share the same bond itag family.
    # Step 1: E_left(c,a,b) × A_i(a,p,r,x) → (c,b,p,r,x), contracting 'a'.
    t1 = einsum('cab,aprx->cbprx', E_left, A_i)
    # Step 2: t1(c,b,p,r,x) × B_i(b,q,x,s) → (c,p,r,q,s), contracting 'b','x'.
    t2 = einsum('cbprx,bqxs->cprqs', t1, B_i)
    # Step 3: t2(c,p,r,q,s) × E_right(d,p,q) → (c,d,r,s), contracting 'p','q'.
    return einsum('cprqs,dpq->cdrs', t2, E_right)
