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


"""Alice: 1D tensor network algorithms built on Nicole."""

from .network import (
    Interaction, Interaction1Site, Interaction2Site,
    build_interaction,
    build_hamiltonian,
    MPS, MPO, Network,
    observe,
)
from .algorithm import dmrg

__all__ = [
    # data types
    'Interaction',
    'Interaction1Site',
    'Interaction2Site',
    # high-level entry points
    'build_interaction',
    'build_hamiltonian',
    # tensor network objects
    'MPS',
    'MPO',
    'Network',
    # measurement
    'observe',
    # algorithms (as submodules)
    'dmrg',
]
