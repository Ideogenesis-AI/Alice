# Examples

This section provides complete, working code examples demonstrating Alice's main features.

## How Examples Work

Code blocks marked with `exec="1"` run during the documentation build and their output is shown inline. This means the results you see are always up to date with the current version of Alice.

Blocks without `exec` are displayed as-is and are intended to be copied and run locally.

## Sections

### DMRG

Complete ground-state calculations with the three update schemes.

- [Heisenberg chain](dmrg/heisenberg.md) — spin-1/2 Heisenberg model, U(1) and SU(2) symmetry.
- [Free fermion](dmrg/free-fermion.md) — spinless tight-binding chain.
- [Hubbard model](dmrg/hubbard.md) — spinful Hubbard chain with U(1)×SU(2) symmetry.

### XTRG

Finite-temperature thermodynamics via exponential cooling, with the three update schemes.

- [Free fermion](xtrg/free-fermion.md) — spinless tight-binding chain, exact `log Z` comparison.
- [Hubbard model](xtrg/hubbard.md) — spinful Hubbard chain with Z(2)×SU(2) symmetry.

### TDVP

Real-time evolution, each validated against an analytic reference or across symmetry groups.

- [Free fermion](tdvp/free-fermion.md) — Gaussian wavepacket against the exact single-particle propagator; clean `dt²` convergence.
- [Domain wall](tdvp/domain-wall.md) — many-body melting against the exact correlation matrix, with occupations *and* entanglement entropy.
- [Hubbard model](tdvp/hubbard.md) — spinful melting at U = 0, Abelian versus SU(2), against twice the spinless result.
- [Hubbard quench](tdvp/hubbard-quench.md) — interaction quench from a DMRG ground state, in all four symmetry groups.

### AutoMPO from TOML

- [AutoMPO from TOML](autompo-toml.md) — loading TOML configs and calling `build_interaction`; multiple model variants.

### Extensions

How to extend Alice with custom physics.

- [Custom geometry](extensions/custom-geometry.md) — define a new lattice type.
- [Custom local space](extensions/custom-space.md) — define a new physical Hilbert space.
- [Custom model](extensions/custom-model.md) — define a new Hamiltonian model.
