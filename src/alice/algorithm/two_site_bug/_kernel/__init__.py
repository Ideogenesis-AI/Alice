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


"""Vendored faithful-KLS (Lübich BUG) local-bond kernel.

This subpackage is the Nicole-native faithful Basis-Update & Galerkin (BUG)
local kernel — the Ceruti–Kusch–Lubich K/L/S two-site update (arXiv:2304.05660),
ported from the reference Julia implementation. It is symmetry-aware (works with
the U(1) charge sectors of an Alice `MPS`) and depends only on `nicole` + torch:

- `_faithful_kls_local_bond_candidate` — one K/L/S local bond update.
- `Ix` / `fresh_itag` — lightweight Nicole-index handles used by the kernel.
- `qr` / `lq` — Nicole-backed decompositions returning `Ix` metadata.
- `dag` / `tcontract` / `make_tensor` / `to_dense` — Nicole tensor helpers.
- `with_time_prefactor` / `with_expv_backend` — evolution-prefactor and Krylov
  backend context managers used to drive the local `expv` substeps.

It is private to `alice.algorithm.two_site_bug`; the Alice-facing driver in
`two_site_bug.py` builds the bond Hamiltonians from AutoMPO and runs the
odd/even Strang sweep on an Alice `MPS` through this kernel.
"""

from .indices import Ix, fresh_itag
from .krylov import with_expv_backend, with_time_prefactor
from .kls import _faithful_kls_local_bond_candidate
from .linalg import lq, qr
from .nicole_helpers import dag, make_tensor, tcontract, to_dense

__all__ = [
    'Ix',
    'fresh_itag',
    'with_expv_backend',
    'with_time_prefactor',
    '_faithful_kls_local_bond_candidate',
    'lq',
    'qr',
    'dag',
    'make_tensor',
    'tcontract',
    'to_dense',
]
