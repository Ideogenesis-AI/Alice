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


"""Logging configuration helpers for Alice.

Typical usage at the start of a script or notebook:

    import alice
    alice.configure_logging()          # uses ALICE_LOGGING env var or .logging/alice.log
    alice.configure_logging("run.log") # explicit file path
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def configure_logging(log_file: Optional[str] = None) -> None:
    """Set up Alice's two-handler logging scheme.

    Attaches a stream handler (INFO and above) and a file handler (DEBUG and
    above) to the `alice` root logger. Calling this function more than once is
    safe: if the `alice` logger already has handlers the function returns
    immediately without adding duplicates.

    The stream handler emits plain messages with no timestamp or level tag, so
    INFO output is readable on a console without noise. The file handler adds a
    timestamp and a fixed-width level tag for easy grepping:

        2026-04-28 12:14:35 [INFO ] sweep 1 / 10: E = -1.234567890123, |ΔE| = 1.2345e-08
        2026-04-28 12:14:35 [DEBUG]   site  3 / 15  local E = -1.234567890123

    Parameters
    ----------
    log_file:
        Path to the log file. If `None`, the value of the `ALICE_LOGGING`
        environment variable is used. If that variable is also unset, the file
        defaults to `.logging/alice_YYYY-MM-DD_HH-MM-SS.log` relative to the
        current working directory. The parent directory is created automatically
        if absent.
    """
    alice_logger = logging.getLogger('alice')

    # Guard against duplicate handler registration on repeated calls.
    if alice_logger.handlers:
        return

    alice_logger.setLevel(logging.DEBUG)

    # --- stream handler: INFO and above, plain message only ---
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    alice_logger.addHandler(stream_handler)

    # --- file handler: DEBUG and above, timestamped with fixed-width level tag ---
    if log_file is None:
        log_file = os.environ.get('ALICE_LOGGING')
    if log_file is None:
        now = datetime.now()
        log_file = str(Path.cwd() / '.logging' / f'alice_{now:%Y-%m-%d}_{now:%H-%M-%S}.log')

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt='%(asctime)s [%(levelname)-5s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
    )
    alice_logger.addHandler(file_handler)
