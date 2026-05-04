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


"""Base classes for algorithm options and summaries.

`AlgorithmOptions` provides TOML-based configuration I/O for dataclass
subclasses. `AlgorithmSummary` is an abstract base that defines the
serialize/deserialize contract and implements save/load via `torch.save`
and `torch.load`.
"""

from __future__ import annotations

import dataclasses
import pathlib
import tomllib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


# ---------------------------------------------------------------------------
# TOML serialisation helper
# ---------------------------------------------------------------------------

def _to_toml_value(v: Any) -> Optional[str]:
    """Convert a Python value to its TOML literal representation.

    Parameters
    ----------
    v:
        The value to convert. Supported types: `bool`, `int`, `float`, `str`.
        `None` signals that the field should be omitted.

    Returns
    -------
    str or None
        TOML literal string, or `None` when the field should be skipped.

    Raises
    ------
    TypeError
        If `v` is not a supported type.
    """
    # bool must be checked before int because bool is a subclass of int.
    if v is None:
        return None
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # repr preserves full precision and always produces a valid TOML float.
        return repr(v)
    if isinstance(v, str):
        # Escape backslashes and double-quotes to produce a valid basic string.
        escaped = v.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'
    raise TypeError(
        f"_to_toml_value: unsupported type {type(v).__name__!r} for value {v!r}; "
        "only bool, int, float, str, and None are supported"
    )


# ---------------------------------------------------------------------------
# AlgorithmOptions
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AlgorithmOptions:
    """Base class for algorithm run options with TOML I/O.

    Subclasses are plain `@dataclass` classes whose fields hold configuration
    values. All three I/O methods (`from_toml`, `load_toml`, `to_toml`) work
    generically on any subclass via `dataclasses.fields`.

    Supported field types for TOML serialisation: `bool`, `int`, `float`,
    `str`, and `Optional` variants of the above (the `None` case is silently
    skipped when writing).
    """

    @classmethod
    def from_toml(cls, section: Dict) -> AlgorithmOptions:
        """Construct an options instance from a TOML section dictionary.

        Unknown keys in `section` are silently ignored so that the same TOML
        block can carry extra metadata without breaking this parser.

        Parameters
        ----------
        section:
            Dictionary loaded from a TOML sub-table, e.g. the result of
            `tomllib.load(f)["mysection"]["algorithm"]`.

        Returns
        -------
        AlgorithmOptions
            Populated options instance of the concrete subclass.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in section.items() if k in known}
        return cls(**kwargs)

    @classmethod
    def load_toml(
        cls,
        path: Union[str, pathlib.Path],
        *,
        section: Optional[str] = None,
    ) -> AlgorithmOptions:
        """Load options from a TOML file.

        Parameters
        ----------
        path:
            Path to the TOML file.
        section:
            Dot-separated key path into the parsed TOML dict. For example,
            `section="heisenberg.algorithm"` is equivalent to
            `cfg["heisenberg"]["algorithm"]`. When `None`, the top-level dict
            is used directly.

        Returns
        -------
        AlgorithmOptions
            Options instance of the concrete subclass.

        Raises
        ------
        KeyError
            If any component of `section` does not exist in the file.
        """
        with open(pathlib.Path(path), 'rb') as fh:
            cfg: Dict = tomllib.load(fh)
        if section is not None:
            for key in section.split('.'):
                cfg = cfg[key]
        return cls.from_toml(cfg)

    def to_toml(
        self,
        path: Union[str, pathlib.Path],
        *,
        section: Optional[str] = None,
    ) -> None:
        """Write options to a TOML file.

        Fields whose value is `None` are omitted (TOML has no null type).
        All other fields must be of type `bool`, `int`, `float`, or `str`.

        Parameters
        ----------
        path:
            Destination file path. The file is created or overwritten.
        section:
            If given, the key-value pairs are written under a `[section]`
            table header, making the file a valid single-table TOML document.

        Raises
        ------
        TypeError
            If a field value is not a supported TOML type.
        """
        lines = []
        if section is not None:
            lines.append(f'[{section}]')
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            toml_val = _to_toml_value(v)
            # Skip None-valued optional fields — TOML has no null.
            if toml_val is None:
                continue
            lines.append(f'{f.name} = {toml_val}')
        pathlib.Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ---------------------------------------------------------------------------
# AlgorithmSummary
# ---------------------------------------------------------------------------

class AlgorithmSummary(ABC):
    """Abstract base class for algorithm output summaries.

    Subclasses implement `serialize` and `deserialize` to define how the
    summary is converted to and from a plain dict. The concrete `save` and
    `load` methods then delegate to those to persist the summary using
    `torch.save` / `torch.load`.

    The dict returned by `serialize` must contain only types safe for
    `torch.load(..., weights_only=True)`: tensors, scalars, lists, and dicts.
    Subclasses are responsible for this contract.
    """

    @abstractmethod
    def serialize(self) -> Dict:
        """Convert the summary to a plain dict compatible with `torch.save`.

        Returns
        -------
        Dict
            A dict containing only `weights_only`-safe values (tensors,
            scalars, lists, dicts).
        """

    @classmethod
    @abstractmethod
    def deserialize(cls, data: Dict, **kwargs) -> AlgorithmSummary:
        """Reconstruct a summary from a dict produced by `serialize`.

        Parameters
        ----------
        data:
            Dict previously returned by `serialize`.
        **kwargs:
            Additional keyword arguments forwarded from `load` (e.g. `device`
            for summaries that contain tensors).

        Returns
        -------
        AlgorithmSummary
            Reconstructed summary instance.
        """

    def save(self, path: Union[str, pathlib.Path]) -> None:
        """Serialize and save the summary to a file using `torch.save`.

        Parameters
        ----------
        path:
            Destination file path. The file is created or overwritten.
        """
        import torch
        torch.save(self.serialize(), pathlib.Path(path))

    @classmethod
    def load(cls, path: Union[str, pathlib.Path], **kwargs) -> AlgorithmSummary:
        """Load a summary from a file saved with `save`.

        Parameters
        ----------
        path:
            Path to a file previously written by `save`.
        **kwargs:
            Forwarded verbatim to `deserialize` (e.g. `device="cuda"`).

        Returns
        -------
        AlgorithmSummary
            Reconstructed summary instance.
        """
        import torch
        data = torch.load(pathlib.Path(path), weights_only=True)
        return cls.deserialize(data, **kwargs)
