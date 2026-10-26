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


"""Infinite DMRG algorithm package.

Optimizes one unit cell of an infinite MPS against an infinite MPO. The
public API so far covers the environments:

- `left_environment` / `right_environment` — regularized boundary fixed
  points, each returning the energy of one unit cell alongside.
- `energy_density` — energy per site.
- `transfer_left` / `transfer_right` — MPS unit-cell transfer operators.
- `cell_left_env` / `cell_right_env` — MPO unit-cell transfer operators.
"""

from .environ import (
    cell_left_env,
    cell_right_env,
    energy_density,
    left_density,
    left_environment,
    right_density,
    right_environment,
    transfer_left,
    transfer_right,
)

__all__ = [
    'cell_left_env',
    'cell_right_env',
    'energy_density',
    'left_density',
    'left_environment',
    'right_density',
    'right_environment',
    'transfer_left',
    'transfer_right',
]
