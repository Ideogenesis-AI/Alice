# What is Alice?

Alice is an open-source library of 1D tensor network algorithms, built on top of [Nicole](https://ideogenesis-ai.github.io/Nicole/) — a symmetry-aware, block-sparse tensor library for quantum physics. Alice pairs a physicist-friendly API with a well-tested, performant backend designed to scale from personal workstations to large HPC clusters.

## Motivation

Tensor network methods are among the most powerful tools for studying strongly-correlated quantum many-body systems. Yet production-quality implementations have historically lived in closed, monolithic codebases that are difficult to modify, extend, or understand. Alice addresses this by:

- Building on **Nicole's exact symmetry engine**, so every operation automatically respects Abelian (U(1), Z(2)) and non-Abelian (SU(2)) conservation laws without additional bookkeeping.
- Exposing a **TOML-driven configuration system** so that running a new model only requires editing a text file rather than writing Python.
- Structuring each algorithm as an **independent unit** on top of a shared MPS/MPO core, making it easy for different researchers to contribute, own, and maintain separate algorithms.

## Algorithms

### DMRG — Density Matrix Renormalization Group

Ground-state search via alternating sweep optimization of MPS tensors: at each step, a local effective Hamiltonian `H_eff` is formed by projecting `H` onto the current site's variational subspace, and its lowest eigenpair `H_eff |ψ⟩ = E |ψ⟩` is found and written back into the MPS, monotonically lowering the global energy `E` sweep after sweep. Three complementary update schemes are available:

- **1-site** (`1s`): single-tensor update; preserves bond dimension exactly, suitable for post-optimization refinement.
- **2-site** (`2s`): two-tensor update with SVD truncation; drives automatic bond growth toward a target dimension.
- **1-site-plus** (`1sp`): controlled bond expansion (CBE) combining the stability of 1-site with the bond-dimension flexibility of 2-site.

All three schemes share a Davidson eigensolver and energy-based convergence criterion. Once converged, the ground-state MPS supports expectation values `⟨ψ|O|ψ⟩` via `observe`, giving access to correlation functions, order parameters, and entanglement entropy from the Schmidt spectrum at each bond.

### XTRG — eXponential Tensor Renormalization Group

Finite-temperature thermodynamics via exponential cooling: the thermal density matrix `ρ(β) = e^{-βH}` is built at a small `τ₀` and repeatedly squared, `ρ(2β) ≈ compress(ρ(β) ⊗ ρ(β))`, doubling `β` at each step. Three update schemes drive the underlying variational MPO-MPO compression:

- **1-site** (`1s`): direct 1-site contraction; preserves bond dimension exactly.
- **2-site** (`2s`): 2-site SVD with truncation; drives bond growth.
- **1-site-plus** (`1sp`): controlled bond expansion (CBE) adapted to XTRG's linear least-squares fitting problem — targets near-2-site accuracy at closer-to-1-site cost.

Each XTRG update solves a linear least-squares fit; thermodynamic observables (`log Z`, free energy, internal energy, specific heat, entropy) can be derived from log-β finite differences across the exponentially spaced temperature grid.

### Upcoming

The following algorithms are planned for future releases. Contributions toward any of them are warmly welcomed — see the [Contributing](contributing.md) page.

- **tanTRG** (tangent-space TRG): finite-temperature simulations with linear cooling steps.
- **TDVP** (Time-Dependent Variational Principle): real-time evolution within the MPS manifold.
- **TaSK** (Tangent Space Krylov): dynamical spectral functions via a Lanczos scheme on the ground-state tangent space.

## Key Features

- **Symmetry-aware MPS/MPO**: block-sparse matrix product states and operators supporting any symmetry group or product group that Nicole supports.
- **AutoMPO construction**: TOML-configured Hamiltonian builder with built-in model presets (Heisenberg, free-fermion, Hubbard) and full support for custom models.
- **Flexible geometries**: built-in 1D chain, 2D square, and 2D Kagome lattice geometries with configurable traversal orders; custom geometry extensions supported via user-defined functions.
- **Environment caching**: optional disk-spilling with a sliding in-memory window and asynchronous I/O, enabling DMRG and XTRG on long chains with limited RAM.
- **Systematic logging**: comprehensive sweep-by-sweep diagnostics via Python's `logging` module.
- **PyTorch backend**: all dense block operations run on PyTorch via Nicole, with optional GPU (CUDA/MPS), Ascend NPU acceleration, and on-demand autograd support.

## Relationship to Nicole

Alice is built entirely on top of [Nicole](https://ideogenesis-ai.github.io/Nicole/). Nicole provides the `Tensor`, `Index`, and symmetry group machinery; Alice adds the MPS/MPO chain structures, the AutoMPO pipeline, the physics model library, and various tensor network algorithms. Users familiar with Nicole will find that Alice follows the same vocabulary and index conventions (see [Core Concepts](core-concepts.md)).

If you encounter a Nicole concept in the Alice API — such as an `Index`, a symmetry string like `'U1'` or `'SU2'`, or a `load_space` call — the [Nicole documentation](https://ideogenesis-ai.github.io/Nicole/) is the authoritative reference.

## AI-Assisted Development

Alice is developed with the assistance of AI coding agents. This makes it possible for a small team to maintain production-quality code, comprehensive tests, and thorough documentation simultaneously. Every contribution — human or AI-assisted — is reviewed, tested, and attributed as part of Alice's collaborative development model.

## License

Alice is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. This means you are free to use, modify, and distribute this software under the terms of the GPL-3.0 license. We encourage you to share any improvements you make back to the community, helping Alice grow and benefit all users. See the [LICENSE](https://github.com/Ideogenesis-AI/Alice/blob/stable/LICENSE) file for the full license text. For more information about GPL-3.0, visit https://www.gnu.org/licenses/gpl-3.0.html
