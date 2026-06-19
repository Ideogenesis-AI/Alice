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

"""Local BUG/KLS bond updates for dense and U(1)-symmetric tensors.

This module contains the Python port of the local Lubich-style K/L/S update used
by the higher-level bug sweep. The main user-facing helper is
``_faithful_kls_local_bond_candidate``. Internally, the code is organized around
one explicit concept:

- ``LocalBondFrame`` gives names to the tensors and indices on the active bond
  so the update logic reads like the algorithm rather than a raw dictionary walk.
"""

from .augment import (
    _augmented_left_isometry_from_k,
    _augmented_right_isometry_from_l,
    _pick_left_update,
    _pick_right_update,
    _truncate_quantum_s_step,
    _truncate_quantum_s_step_reverse,
)
from .candidate import (
    _faithful_kls_local_bond_candidate,
    _faithful_reverse_kls_local_bond_candidate,
    _symmetric_local_bond_candidate,
)
from .symmetric_completion import (
    _symmetric_augmented_left_isometry_from_k,
    _symmetric_augmented_right_isometry_from_l,
)
from .frame import LocalBondFrame

__all__ = [
    "LocalBondFrame",
    "_augmented_left_isometry_from_k",
    "_augmented_right_isometry_from_l",
    "_faithful_kls_local_bond_candidate",
    "_faithful_reverse_kls_local_bond_candidate",
    "_pick_left_update",
    "_pick_right_update",
    "_symmetric_augmented_left_isometry_from_k",
    "_symmetric_augmented_right_isometry_from_l",
    "_symmetric_local_bond_candidate",
    "_truncate_quantum_s_step",
    "_truncate_quantum_s_step_reverse",
]
