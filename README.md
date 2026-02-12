# Alice

**Alice** is a Python library for 1-dimensional tensor network algorithms, built upon the [Nicole](https://github.com/Ideogenesis-AI/Nicole) tensor library.

## Overview

Alice provides implementations of state-of-the-art tensor network algorithms for simulating 1D quantum systems, including:

- **DMRG** (Density Matrix Renormalization Group): Ground state search and optimization
- **XTRG** (eXponential Tensor Renormalization Group): Finite-temperature simulations
- **tanTRG** (tangent-space Tensor Renormalization Group): Critical systems and phase transitions
- **TDVP** (Time-Dependent Variational Principle): Real-time evolution of quantum states

## Features

- Built on Nicole's symmetry-aware tensor library for efficient computations
- Support for Abelian symmetries (U(1), Z_n, etc.)
- Modern Python implementation with type hints
- Optimized for performance using NumPy and PyTorch backends

## Requirements

- Python >= 3.11
- Nicole >= 0.2.0
- NumPy >= 2.0
- PyTorch >= 2.5

## Development Status

🚧 This project is in early development. Implementations are actively being added.

## License

This project is licensed under the GNU General Public License v3 (GPLv3) - see the LICENSE file for details.

## Acknowledgments

Alice builds upon the Nicole tensor library and draws inspiration from various tensor network implementations in the community.
