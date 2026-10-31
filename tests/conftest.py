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


"""Working-directory isolation shared by the whole test suite.

Every driver resolves `checkpoint_dir=None` to `Path.cwd()` when `run()` is
called, so a test that leaves the options at their defaults writes its
checkpoints and artifacts wherever the session happens to stand. These two
fixtures keep that out of the project root, and live here rather than in the
per-algorithm conftests so a new test directory is covered without having to
remember them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# The canonical basis cache that yuzuha keeps at `.yuzuha/cgbasis.db` is
# resolved relative to the working directory too, but unlike a checkpoint it
# is worth keeping: rebuilding the Clebsch-Gordan data for every session costs
# far more than the SU(2) tests themselves.
CACHE_DIR = Path(__file__).resolve().parent.parent / '.yuzuha'


@pytest.fixture(scope='session', autouse=True)
def _isolate_cwd_session(tmp_path_factory):
    """Run the whole session with cwd set to a session-wide temporary path.

    The function-scoped fixture below cannot cover this on its own: pytest
    instantiates fixtures from the widest scope inward, so a session-, module-,
    or class-scoped fixture that calls `run()` does so before any function-
    scoped `chdir` has happened. This one is in place before any of them.
    """
    # `monkeypatch` is function-scoped, so patch through a context of our own.
    with pytest.MonkeyPatch.context() as patch:
        # An explicit cache path from the environment wins, so a run can still
        # be pointed at a cache of its own.
        if 'YUZUHA_CACHE_PATH' not in os.environ:
            patch.setenv('YUZUHA_CACHE_PATH', str(CACHE_DIR))
        patch.chdir(tmp_path_factory.mktemp('cwd'))
        yield


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Run every test with cwd set to a fresh `tmp_path`.

    Beyond keeping the project root clean, this stops one test from reading a
    checkpoint another one left behind.
    """
    monkeypatch.chdir(tmp_path)
