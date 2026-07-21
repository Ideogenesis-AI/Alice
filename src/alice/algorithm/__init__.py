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

`discarded_bug` -- the global-sweep BUG -- was moved to `exploratory/global_sweep`
and is no longer importable from here. The supported discarded-projector kernel is
`two_site_bug` with `variant='discarded'`, which is the one mirrored by
`bond_update_bug!` in BUG-Julia.
"""

from . import two_site_bug
from . import dmrg
from . import tdvp2

__all__ = [
    'two_site_bug',
    'dmrg',
    'tdvp2',
]
