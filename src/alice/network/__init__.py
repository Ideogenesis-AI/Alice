# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Alice library.
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


"""Network module for MPS/MPO operations."""

from .intrcmap import Interaction, generate_snake_order, intrcmap_square
from .network import MPS, MPO, Network
from .observe import observe

__all__ = [
    # intrcmap
    'Interaction',
    'generate_snake_order',
    'intrcmap_square',
    # network
    'MPS',
    'MPO',
    'Network',
    # observe
    'observe',
]
