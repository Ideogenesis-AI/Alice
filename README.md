<h1 align="center">
  <img src="docs/images/Alice.png" alt="Alice Tensor Network Algorithms" width="280">
</h1>

<p align="center">
  <a href="https://pypi.org/project/alice/"><img src="https://img.shields.io/pypi/v/alice?color=red" alt="PyPI Version"></a>
  <a href="https://github.com/Ideogenesis-AI/Alice/blob/stable/LICENSE"><img src="https://img.shields.io/github/license/Ideogenesis-AI/Alice?color=orange" alt="License"></a>
  <a href="https://ideogenesis-ai.github.io/Alice"><img src="https://img.shields.io/badge/docs-github.io-c9a400" alt="Documentation"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/pypi/pyversions/alice?color=228b22" alt="Python Version"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.5+-blue?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://github.com/Ideogenesis-AI/Nicole"><img src="https://img.shields.io/badge/built%20on-Nicole-blueviolet" alt="Built on Nicole"></a>
  <a href="https://pypi.org/project/alice/"><img src="https://img.shields.io/pypi/status/alice?color=4b0082" alt="Status"></a>
</p>

Alice is an open source software for 1D tensor network algorithms, built on the [Nicole](https://github.com/Ideogenesis-AI/Nicole) symmetry-aware tensor library. The ecosystem provides production-quality implementations of state-of-the-art algorithms for simulating 1D (and quasi-1D) quantum many-body systems, with full support for Abelian and non-Abelian (currently SU(2) only) symmetries inherited from Nicole.

With the assistance of various AI coding agents, Alice pairs a physicist-friendly API — TOML-driven model definitions, concise Python entry points — with a well-tested, performant backend designed with HPC in mind, scaling from workstations to large-scale distributed computations, making cutting-edge tensor network calculations broadly accessible while maintaining the mathematical rigor for quantum physics applications.


## Algorithms

### DMRG (Density Matrix Renormalization Group)

Ground-state search and optimization via alternating sweep optimization of MPS tensors. Three complementary update schemes are available — 1-site, 2-site, and 1-site-plus (controlled bond expansion) — each with a Davidson eigensolver and energy-based convergence criterion.

### Upcoming

The following algorithms are planned for future releases. Contributions toward any of them are warmly welcomed — see the [Contributing](#contributing) section to get involved.

- **XTRG** (eXponential Tensor Renormalization Group): finite-temperature simulations with exponential cooling, reaching very low temperatures efficiently
- **tanTRG** (tangent-space Tensor Renormalization Group): finite-temperature simulations with linear cooling steps, offering high speed at moderate to high temperatures
- **TDVP** (Time-Dependent Variational Principle): real-time evolution of quantum states within the MPS manifold, conserving energy and norm during time integration
- **TaSK** (Tangent Space Krylov): dynamical properties and real-frequency spectral functions, computed by a Lanczos scheme projected onto the tangent space of the ground-state MPS


## Key Features

- **Symmetry-Aware MPS/MPO**: block-sparse matrix product states and operators supporting any symmetry group or product group that Nicole supports, leveraging its exact block-sparse engine
- **AutoMPO Construction**: TOML-configured Hamiltonian builder with built-in model presets (Heisenberg, free-fermion, Hubbard) and full support for custom models via user-defined model functions
- **Flexible Geometries**: built-in 1D and quasi-1D lattice geometries with custom geometry extensions supported via user-defined geometry functions
- **Environment Caching**: optional disk-spilling of environment blocks with a sliding in-memory window and asynchronous I/O, enabling DMRG on long chains with limited RAM
- **Systematic Logging**: comprehensive sweep-by-sweep diagnostics via Python's `logging` module, configurable with `alice.configure_logging()`
- **PyTorch Backend**: all dense block operations run on PyTorch via Nicole, with optional GPU (CUDA/MPS), Ascend NPU acceleration, and on-demand autograd support


## Contributing

Alice is structured around a shared core — the MPS, MPO, and network infrastructure — on top of which each algorithm is developed and maintained as an independent unit. This means that individual algorithms can be contributed, extended, or maintained by different people, and each contributor's work is attributed accordingly. If you have expertise in a particular algorithm or wish to take ownership of one of the upcoming implementations, your involvement is especially valued.

**Ways to contribute:**
- Implement or co-develop one of the planned algorithms (see [Upcoming](#upcoming))
- Report issues and request features via [GitHub Issues](https://github.com/Ideogenesis-AI/Alice/issues)
- Submit pull requests with bug fixes or enhancements
- Share benchmarks, use cases, or constructive feedback

**Development guidelines:**
- Ensure contributions include appropriate tests
- Follow the existing code style (enforced by `ruff`)
- Add type hints for all public function signatures
- Update documentation for user-facing changes

**Authors and Maintainers:**

Alice is created by [Changkai Zhang](https://chx-zh.cc) as part of the Ideogenesis-AI effort in studying quantum many-body systems. The core infrastructure is maintained by the original author; individual algorithms have their respective maintenance teams. For questions about contributing or collaboration opportunities, feel free to open an issue on GitHub or contact the responsible team directly.


## License

Alice is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. This means you are free to use, modify, and distribute this software under the terms of the GPL-3.0 license. We encourage you to share any improvements you make back to the community, helping Alice grow and benefit all users. See the [LICENSE](LICENSE) file for the full license text. For more information about GPL-3.0, visit https://www.gnu.org/licenses/gpl-3.0.html
