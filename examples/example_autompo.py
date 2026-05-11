"""Example: Hamiltonian MPO construction from a TOML config.

Reads a named section from `example_config.toml`, builds the geometry and
model interactions via `build_interaction`, then compresses them into a
Hamiltonian MPO via `build_hamiltonian`. Any `[<name>.autompo]` sub-table in
the config is forwarded as keyword arguments to `build_hamiltonian`.

Available models (keys in example_config.toml):

    heisenberg_u1     — Heisenberg spin-1/2 chain, U1 symmetry
    heisenberg_su2    — Heisenberg spin-1/2 chain, SU2 symmetry
    free_fermion      — spinless tight-binding chain, U1 symmetry
    hubbard_u1u1      — Hubbard chain, (U1, U1) symmetry
    heisenberg_2d     — Heisenberg square lattice, NN + NNN, U1 symmetry
    hubbard_cylinder  — Hubbard 8×8 cylinder, NN + NNN, (U1, SU2) symmetry

Run with:

    uv run python examples/example_autompo.py
    uv run python examples/example_autompo.py --model hubbard_u1u1
    uv run python examples/example_autompo.py --model heisenberg_su2
    uv run python examples/example_autompo.py --model heisenberg_2d
    uv run python examples/example_autompo.py --config path/to/other.toml --model mymodel
    uv run python examples/example_autompo.py --list
    uv run python examples/example_autompo.py --help
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import alice
from alice import build_interaction, build_hamiltonian

_DEFAULT_CONFIG = Path(__file__).parent / "example_config.toml"
_DEFAULT_MODEL  = "hubbard_cylinder"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a Hamiltonian MPO from a named section of a TOML config file.\n\n"
            "Examples:\n"
            "  uv run python examples/example_autompo.py\n"
            "  uv run python examples/example_autompo.py --model hubbard_u1u1\n"
            "  uv run python examples/example_autompo.py --model heisenberg_su2\n"
            "  uv run python examples/example_autompo.py --model heisenberg_2d\n"
            "  uv run python examples/example_autompo.py --config path/to/other.toml --model mymodel\n"
            "  uv run python examples/example_autompo.py --list"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--config', type=Path, default=_DEFAULT_CONFIG, metavar='PATH',
        help=f'path to the TOML config file (default: {_DEFAULT_CONFIG.name})',
    )
    p.add_argument(
        '--model', default=_DEFAULT_MODEL, metavar='NAME',
        help=f'section name to load from the config (default: {_DEFAULT_MODEL!r})',
    )
    p.add_argument(
        '--list', action='store_true',
        help='list all available model names in the config file and exit',
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    with open(args.config, 'rb') as f:
        cfg = tomllib.load(f)

    if args.list:
        print("Available models in", args.config)
        for name in cfg:
            print(f"  {name}")
        sys.exit(0)

    if args.model not in cfg:
        print(f"error: model {args.model!r} not found in {args.config}", file=sys.stderr)
        print(f"       available: {', '.join(cfg)}", file=sys.stderr)
        sys.exit(1)

    sec = cfg[args.model]
    autompo_opts = sec.get('autompo', {})

    print(f"Config  : {args.config}")
    print(f"Model   : {args.model}")
    print()

    # Stage 1 + 2: geometry and model
    interactions, spc, geo = build_interaction(sec)

    # Stage 3: build the Hamiltonian MPO
    mpo = build_hamiltonian(interactions, geo.L, spc, **autompo_opts)

    print()
    print(f"Chain length : {geo.L}")
    print(f"MPO length   : {len(mpo)}")
    print(f"Bond dims    : {mpo.bond_dims}")


if __name__ == '__main__':
    alice.configure_logging()
    main()
