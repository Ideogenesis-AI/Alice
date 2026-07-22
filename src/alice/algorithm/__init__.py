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


"""Algorithm module: tensor network algorithms built on the network layer.

`bond_update_bug` is the single Basis-Update & Galerkin time integrator (the
discarded-projector K/L/S sweep, mirrored by `bond_update_bug!` in BUG-Julia);
`dmrg` and `tdvp2` are the ground-state and TDVP algorithms.
"""

from . import bond_update_bug
from . import dmrg
from . import tdvp2

__all__ = [
    'bond_update_bug',
    'dmrg',
    'tdvp2',
]
