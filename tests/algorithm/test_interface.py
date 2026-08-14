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


"""Tests for alice.algorithm.interface (AlgorithmOptions, AlgorithmSummary).

Uses inline concrete subclasses to isolate the base-class behavior from any
algorithm-specific logic. No DMRG or MPS imports appear here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pytest

from alice.algorithm.interface import AlgorithmOptions, AlgorithmSummary


# ---------------------------------------------------------------------------
# Inline concrete subclasses used only in these tests
# ---------------------------------------------------------------------------

@dataclass
class _ConcreteOptions(AlgorithmOptions):
    n: int = 5
    tol: float = 1e-8
    label: str = "test"
    flag: bool = True
    extra: Optional[int] = None   # None → omitted in TOML output


@dataclass
class _ConcreteSummary(AlgorithmSummary):
    value: float = 0.0

    def serialize(self) -> Dict:
        return {'value': self.value}

    @classmethod
    def deserialize(cls, data: Dict, **kwargs) -> _ConcreteSummary:
        return cls(value=data['value'])


# ---------------------------------------------------------------------------
# TestAlgorithmOptions
# ---------------------------------------------------------------------------

class TestAlgorithmOptions:
    """Tests for AlgorithmOptions TOML I/O."""

    def test_from_toml_known_fields(self):
        opts = _ConcreteOptions.from_toml({'n': 10, 'tol': 1e-12, 'label': 'run'})
        assert opts.n == 10
        assert abs(opts.tol - 1e-12) < 1e-20
        assert opts.label == 'run'

    def test_from_toml_ignores_unknown_keys(self):
        opts = _ConcreteOptions.from_toml({'n': 3, 'future_key': 'ignored'})
        assert opts.n == 3

    def test_from_toml_defaults(self):
        opts = _ConcreteOptions.from_toml({})
        assert opts.n == 5
        assert opts.flag is True
        assert opts.extra is None

    def test_to_toml_round_trip(self, tmp_path):
        original = _ConcreteOptions(n=42, tol=1e-6, label='roundtrip', flag=False)
        path = tmp_path / 'opts.toml'
        original.to_toml(path)
        loaded = _ConcreteOptions.load_toml(path)
        assert loaded.n == 42
        assert abs(loaded.tol - 1e-6) < 1e-20
        assert loaded.label == 'roundtrip'
        assert loaded.flag is False

    def test_to_toml_with_section(self, tmp_path):
        path = tmp_path / 'opts.toml'
        _ConcreteOptions(n=7).to_toml(path, section='mysection')
        loaded = _ConcreteOptions.load_toml(path, section='mysection')
        assert loaded.n == 7

    def test_to_toml_skips_none(self, tmp_path):
        path = tmp_path / 'opts.toml'
        _ConcreteOptions(extra=None).to_toml(path)
        content = path.read_text()
        # 'extra' must not appear since its value is None.
        assert 'extra' not in content

    def test_to_toml_bool_values(self, tmp_path):
        path = tmp_path / 'opts.toml'
        _ConcreteOptions(flag=True).to_toml(path)
        content = path.read_text()
        # TOML booleans must be lowercase; Python 'True'/'False' are invalid TOML.
        assert 'true' in content
        assert 'True' not in content

    def test_load_toml_bad_section_raises(self, tmp_path):
        path = tmp_path / 'opts.toml'
        _ConcreteOptions().to_toml(path, section='good')
        with pytest.raises(KeyError):
            _ConcreteOptions.load_toml(path, section='nonexistent')


# ---------------------------------------------------------------------------
# TestAlgorithmSummary
# ---------------------------------------------------------------------------

class TestAlgorithmSummary:
    """Tests for AlgorithmSummary save/load I/O."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AlgorithmSummary()  # type: ignore[abstract]

    def test_save_load_round_trip(self, tmp_path):
        original = _ConcreteSummary(value=3.14)
        path = tmp_path / 'summary.pt'
        original.save(path)
        loaded = _ConcreteSummary.load(path)
        assert isinstance(loaded, _ConcreteSummary)
        assert abs(loaded.value - 3.14) < 1e-12

    def test_load_forwards_kwargs(self, tmp_path):
        """Extra keyword arguments passed to load are forwarded to deserialize."""
        path = tmp_path / 'summary.pt'
        _ConcreteSummary(value=1.0).save(path)
        # _ConcreteSummary.deserialize accepts **kwargs, so device='cpu' is valid.
        loaded = _ConcreteSummary.load(path, device='cpu')
        assert abs(loaded.value - 1.0) < 1e-12
