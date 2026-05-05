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


"""Interaction dataclasses and TOML-driven dispatcher for AutoMPO construction."""

from __future__ import annotations

import importlib.util
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from nicole import Index, Tensor

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Interaction:
    """Base class for a single Hamiltonian interaction term.

    Attributes
    ----------
    cpl:
        Coupling constant. Set to `0.0` by the geometry builder; the model
        builder assigns the physical value. `build_hamiltonian` multiplies
        the terminal (or on-site) tensor by this value when constructing the
        MPO.
    label:
        List of string labels encoding the bond topology (e.g. `['NN', 'N2X']`,
        `['NNN', 'N3D']`). Set by the geometry builder; used by the model
        builder to assign the correct coupling per bond type.
    """

    cpl:   float     = 0.0
    label: List[str] = field(default_factory=list)


@dataclass
class Interaction1Site(Interaction):
    """On-site (1-site) interaction term.

    Attributes
    ----------
    site:
        Site index (0-based).
    tnsr:
        4-index MPO tensor in format `(L_trivial_IN, R_trivial_OUT, bra_OUT,
        ket_IN)`. The coupling `cpl` is applied by `build_hamiltonian` and
        must NOT be baked in. Set by the model builder.
    """

    site: int = 0
    tnsr: Optional[Tensor] = None


@dataclass
class Interaction2Site(Interaction):
    """Two-site interaction term.

    Attributes
    ----------
    leading_site:
        Index of the leading (left) site (0-based).
    terminal_site:
        Index of the terminal (right) site (0-based). Must satisfy
        `terminal_site > leading_site`.
    leading_tnsr:
        4-index MPO tensor for the leading site, in format
        `(L_trivial_IN, op_OUT, bra_OUT, ket_IN)`. The op axis carries the
        operator channel contracted with the terminal site. Set by the model
        builder.
    terminal_tnsr:
        4-index MPO tensor for the terminal site, in format
        `(op_IN, R_trivial_OUT, bra_OUT, ket_IN)`. `build_hamiltonian`
        scales this tensor by `cpl`; the model builder must NOT bake the
        coupling in. Set by the model builder.
    intermid_tnsr:
        4-index MPO tensor for intermediate sites (between `leading_site` and
        `terminal_site`), in format `(left_op_IN, right_op_OUT, bra_OUT,
        ket_IN)`. Required when `terminal_site > leading_site + 1`. For
        bosonic systems this is typically the physical identity dressed with
        op-sector bonds; for fermionic systems it is the Jordan-Wigner string.
        Set by the model builder.
    """

    leading_site:  int = 0
    terminal_site: int = 0
    leading_tnsr:  Optional[Tensor] = None
    terminal_tnsr: Optional[Tensor] = None
    intermid_tnsr: Optional[Tensor] = None


# ---------------------------------------------------------------------------
# Plugin loading helper
# ---------------------------------------------------------------------------

def _load_plugin(spec_str: str, base_dir: Optional[Path] = None) -> Callable:
    """Load a callable from a `"path/to/file.py:function_name"` plugin spec.

    Parameters
    ----------
    spec_str:
        Plugin specification in the form `"path/to/file.py:function_name"`.
        The path may be absolute or relative; relative paths are resolved
        against `base_dir` (or the current working directory if `None`).
    base_dir:
        Directory used as the root for resolving relative paths.

    Returns
    -------
    Callable
        The named function loaded from the specified file.

    Raises
    ------
    ValueError
        If `spec_str` does not contain exactly one `':'` separator.
    ImportError
        If the module cannot be loaded or the function is not found.
    """
    if ':' not in spec_str:
        raise ValueError(
            f"Plugin spec must be 'path/to/file.py:function_name', got '{spec_str}'"
        )
    file_part, fn_name = spec_str.rsplit(':', 1)
    file_path = Path(file_part)
    if not file_path.is_absolute() and base_dir is not None:
        file_path = base_dir / file_path

    spec = importlib.util.spec_from_file_location('_alice_plugin', file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plugin module from '{file_path}'")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    if not hasattr(mod, fn_name):
        raise ImportError(
            f"Plugin module '{file_path}' has no attribute '{fn_name}'"
        )
    return getattr(mod, fn_name)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def build_interaction(
    config: Union[dict, str, Path],
    *,
    geometry_fn: Optional[Callable] = None,
    model_fn:    Optional[Callable] = None,
    space_fn:    Optional[Callable] = None,
) -> Tuple[List[Interaction], Index, int]:
    """Build a fully populated interaction list from a TOML config.

    Orchestrates the three-stage MPO construction pipeline:

    1. **Geometry** — constructs the bare interaction list (sites + labels,
       no tensors, `cpl == 0.0`).
    2. **Model** — fills `cpl` and tensor fields on each interaction
       (without baking coupling into the tensors).
    3. Returns `(interactions, spc, L)` ready for `build_hamiltonian`.

    Callable overrides (`geometry_fn`, `model_fn`, `space_fn`) take priority
    over `[plugin]` section entries in the config, which in turn take priority
    over the built-in dispatch tables.

    Parameters
    ----------
    config:
        Either a config dict (with `'geometry'` and `'model'` sub-dicts) or a
        path to a TOML file.
    geometry_fn:
        Optional override for the geometry builder. Signature:
        `geometry_fn(geo: dict) -> list[Interaction2Site]`.
    model_fn:
        Optional override for the model builder. Signature:
        `model_fn(interactions, L, model_cfg: dict) -> tuple[Index, dict]`.
    space_fn:
        Optional override for the operator-set builder (normally called
        internally by the model builder). Passed through to the model builder
        as a keyword argument `space_fn=space_fn`.

    Returns
    -------
    tuple
        `(interactions, spc, L)` where `interactions` is the populated list,
        `spc` is the physical `Index`, and `L` is the chain length.

    Raises
    ------
    ValueError
        If a required key is missing or an unknown `lattice`, `traverse`,
        `category`, or `label` value is encountered.
    """
    # Lazy imports to avoid circular dependency:
    # network.interaction ← physics.geometry ← network.interaction
    from alice.physics.geometry import build_geometry          # noqa: PLC0415
    from alice.physics.models import (                         # noqa: PLC0415
        build_heisenberg,
        build_free_fermion,
        build_hubbard,
    )
    from alice.physics.system import (                         # noqa: PLC0415
        build_bosonic,
        build_fermionic,
        build_conductor,
    )

    # -----------------------------------------------------------------------
    # Load config from file if a path is given.
    # -----------------------------------------------------------------------
    
    base_dir: Optional[Path] = None
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        base_dir    = config_path.parent
        with open(config_path, 'rb') as fh:
            config = tomllib.load(fh)

    geo_cfg   = config['geometry']
    model_cfg = config['model']
    plugin    = config.get('plugin', {})

    # -----------------------------------------------------------------------
    # Resolve callables: kwarg > [plugin] entry > built-in.
    # -----------------------------------------------------------------------

    _MODEL_BUILTIN: Dict[str, Callable] = {
        'Heisenberg':  build_heisenberg,
        'FreeFermion': build_free_fermion,
        'Hubbard':     build_hubbard,
    }
    _SPACE_BUILTIN: Dict[str, Callable] = {
        'bosonic':    build_bosonic,
        'fermionic':  build_fermionic,
        'conductor':  build_conductor,
    }

    # Geometry callable — build_geometry handles all registered lattice types.
    if geometry_fn is None and 'geometry' in plugin:
        geometry_fn = _load_plugin(plugin['geometry'], base_dir)
    if geometry_fn is None:
        geometry_fn = build_geometry

    # Space (operator-set) callable
    if space_fn is None and 'space' in plugin:
        space_fn = _load_plugin(plugin['space'], base_dir)
    if space_fn is None:
        category = model_cfg.get('category', '')
        if category in _SPACE_BUILTIN:
            space_fn = _SPACE_BUILTIN[category]
        # If category is empty or unknown, let the model builder use its default.

    # Model callable
    if model_fn is None and 'model' in plugin:
        model_fn = _load_plugin(plugin['model'], base_dir)
    if model_fn is None:
        label = model_cfg.get('label', '')
        if label not in _MODEL_BUILTIN:
            raise ValueError(
                f"Unknown model label '{label}'. "
                f"Available: {list(_MODEL_BUILTIN)}"
            )
        model_fn = _MODEL_BUILTIN[label]

    # -----------------------------------------------------------------------
    # Stage 1: Geometry
    # -----------------------------------------------------------------------

    interactions = geometry_fn(geo_cfg)
    L = geo_cfg['lx'] * geo_cfg.get('ly', 1)

    # -----------------------------------------------------------------------
    # Stage 2: Model (fills cpl + tensors; coupling not baked in)
    # -----------------------------------------------------------------------

    kwargs: Dict[str, object] = dict(model_cfg)
    # Remove dispatcher-internal keys before forwarding.
    for _key in ('category', 'label'):
        kwargs.pop(_key, None)
    if space_fn is not None:
        kwargs['space_fn'] = space_fn

    spc, _ = model_fn(interactions, L, **kwargs)

    # -----------------------------------------------------------------------
    # Log model specifications.
    # -----------------------------------------------------------------------

    logger.info("─" * 60)
    logger.info("Model Specifications".center(60))
    logger.info("─" * 60)
    logger.info("")

    # Model identity
    model_label = model_cfg.get('label', '<custom>')
    model_cat   = model_cfg.get('category', '')
    cat_str     = f"  ({model_cat})" if model_cat else ""
    logger.info(f"  Context:   {model_label}{cat_str}")

    # Lattice dimensions and boundary conditions
    lx  = geo_cfg['lx']
    ly  = geo_cfg.get('ly', 1)
    bcx = geo_cfg.get('bcx', 'OBC').upper()
    bcy = geo_cfg.get('bcy', 'OBC').upper()
    if ly == 1:
        logger.info(f"  Lattice:   L = {L}")
        logger.info(f"  Boundary:  {bcx}")
    else:
        logger.info(f"  Lattice:   {lx} \u00d7 {ly}  (L = {L})")
        logger.info(f"  Boundary:  {bcx} \u00d7 {bcy}")

    logger.info("")

    # Coupling parameters: all model_cfg keys except 'label' and 'category'.
    # 'symmetry' is displayed as 'symm' for brevity.
    _skip    = {'label', 'category'}
    _aliases = {'symmetry': 'symm'}
    params   = [(_aliases.get(k, k), v) for k, v in model_cfg.items() if k not in _skip]
    if params:
        col = max(len(k) for k, _ in params) + 1
        for k, v in params:
            logger.info(f"  {(k + ':').ljust(col)}  {v}")
        logger.info("")

    # Interaction summary
    n_total  = len(interactions)
    n_active = sum(1 for intr in interactions if intr.cpl != 0.0)
    logger.info(f"  Active interactions:  {n_active} / {n_total}")
    logger.info("")

    return interactions, spc, L
