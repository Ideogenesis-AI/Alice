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


"""Physics module: physical space, geometry, and model builders."""

from .system import build_bosonic, build_fermionic, build_conductor
from .chain import intrcmap_1dchain
from .square import intrcmap_square
from .kagome import intrcmap_kagome
from .geometry import Geometry, build_geometry, build_intrcmap
from .models import build_heisenberg, build_free_fermion, build_hubbard

__all__ = [
    # system — operator-set builders
    'build_bosonic',
    'build_fermionic',
    'build_conductor',
    # geometry — struct and interaction map builders
    'Geometry',
    'build_geometry',
    'build_intrcmap',
    'intrcmap_1dchain',
    'intrcmap_square',
    'intrcmap_kagome',
    # models — Hamiltonian model builders
    'build_heisenberg',
    'build_free_fermion',
    'build_hubbard',
]
