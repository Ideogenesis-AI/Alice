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
# Author of code: Madhav Menon.


"""Pytest fixtures and exact-diagonalization helpers for discarded-projector BUG tests.

The discarded-projector BUG shares its model, dense Hamiltonian, and dense-vector
plumbing with the faithful two-site BUG, so the exact-diagonalization helpers are
imported from the two-site BUG test conftest and re-exported here. Only the
spin-1/2 U(1) ``spin_space`` fixture and the working-directory isolation fixture
are redeclared so pytest discovers them in this package.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pytest
from nicole import Index, Tensor, load_space

# Reuse the faithful BUG test's exact-diagonalization helpers verbatim.
from tests.algorithm.two_site_bug.conftest import (  # noqa: F401
    dense_hamiltonian,
    dense_heisenberg,
    dense_total_sz,
    exact_evolve,
    heisenberg_chain,
    mps_to_vector,
    product_vector,
)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Run every test in a fresh working directory."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope='session')
def spin_space() -> Tuple[Index, Dict[str, Tensor]]:
    """Spin-1/2 U(1) physical space and operators (shared across the session)."""
    return load_space('Spin', 'U1', {'J': 0.5})
