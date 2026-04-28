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
    alice.configure_logging()   # uses ALICE_LOGGING env var
                                # or .logging/alice_YYYY-MM-DD_HH-MM-SS.log
    alice.configure_logging("run.log") # explicit file path
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import re
import sys

from datetime import datetime
from pathlib import Path
from typing import Optional


def _version_status(version: str) -> str:
    """Map a PEP 440 version string to a short release-level label."""
    m = re.search(r'(a|b|rc)\d*$', version)
    if m is None:
        return 'stable'
    return {'a': 'alpha', 'b': 'beta', 'rc': 'rc'}[m.group(1)]


def _log_banner(log: logging.Logger) -> None:
    """Emit the Alice startup banner."""
    width = 60
    title = 'Alice  \u2014  1D Tensor Network Algorithms'
    sep   = '\u2014' * (len(title) + 2)   # 1 em-dash wider on each side

    # Gather metadata.
    alice_ver    = importlib.metadata.version('alice')
    alice_status = _version_status(alice_ver)
    py_ver       = (f"{sys.version_info.major}.{sys.version_info.minor}"
                    f".{sys.version_info.micro}")
    py_status    = ('stable' if sys.version_info.releaselevel == 'final'
                    else sys.version_info.releaselevel)

    _sys  = platform.system()
    _mach = platform.machine()
    if _sys == 'Darwin':
        system_str = f"macOS {platform.mac_ver()[0]} ({_mach})"
    elif _sys == 'Linux':
        system_str = f"Linux {platform.release()} ({_mach})"
    elif _sys == 'Windows':
        system_str = f"Windows {platform.release()} ({_mach})"
    else:
        system_str = f"{_sys} ({_mach})"

    session = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Fixed key column width aligns all values.
    key_col    = 10
    info_lines = [
        f"{'Version:'.ljust(key_col)}v{alice_ver} ({alice_status})",
        f"{'Python:'.ljust(key_col)}v{py_ver} ({py_status})",
        f"{'System:'.ljust(key_col)}{system_str}",
        f"{'Author:'.ljust(key_col)}Changkai Zhang",
        f"{'License:'.ljust(key_col)}GNU GPL-v3",
        f"{'Session:'.ljust(key_col)}{session}",
    ]

    # Block-centre: all info lines share the same left padding.
    block_pad = ' ' * max(0, (width - max(len(l) for l in info_lines)) // 2)

    log.info('=' * width)
    log.info('')
    log.info(sep.center(width))
    log.info(title.center(width))
    log.info(sep.center(width))
    log.info('')
    for line in info_lines:
        log.info(block_pad + line)
    log.info('')
    log.info('=' * width)
    log.info('')

    # Legal notice — follows the banner as a separate block.
    log.info('  Alice is created and maintained by Changkai Zhang as')
    log.info('  a collection of 1D tensor network algorithms. Each')
    log.info('  implementation is credited to its respective author(s).')
    log.info('')
    log.info('  This software is distributed without any warranty;')
    log.info('  without even the implied warranty of merchantability')
    log.info('  or fitness for a particular purpose.')
    log.info('')


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

    _log_banner(alice_logger)
