# MPS Initialization

Construct a symmetry-aware initial MPS for tensor network algorithms.

::: alice.network.automps.init_mps
    options:
      show_source: false
      heading_level: 2

## Notes

`init_mps` operates in two modes selected by `bond_dim`:

**Product state (`bond_dim=1`)** — every virtual bond carries a single sector
whose charge is determined by propagating the physical charges of the chosen
site configuration through the group fusion rules. For Abelian groups this is
additive; for SU(2) and product groups containing SU(2) the minimum-branch
(dimer/VBS) rule selects the lowest-spin channel at each step. The result is a
bond-dimension-1 MPS in a definite target sector. This mode pairs naturally
with CBE (`scheme='1sp'`) or 2-site (`scheme='2s'`) DMRG, which grow the bond
dimension during the first few sweeps.

**Random MPS (`bond_dim>1`)** — bond sectors are discovered by a breadth-first
search (BFS) of depth 2 from the center-bond charge, so the sector set is
independent of chain length and contains only charges reachable by physical
fusion steps. Tensors are filled with random entries and the MPS is
canonicalized with a two-pass sweep (right to site `L-1`, then left back to
site 0) to compress spurious bond dimension from both ends.

Both modes are particle-type agnostic: no `spin=`, `symmetry=`, or
`particle_type=` argument is required. The `(Spc, Op)` pair returned by
Nicole's `load_space` encodes all symmetry information.

### Charge Conventions

All standard Nicole physical spaces define charges relative to the half-filled
reference, so the center-bond charge for a balanced auto-config is always zero
regardless of chain length:

| Space | Sectors | Q_vac |
|-------|---------|-------|
| Spin U(1) | S~z~ = ±½ → charges ±1 | 0 |
| Spin SU(2) | multiplet label 2J | 0 |
| Ferm U(1) | empty/occupied → charges ±1 | 0 |
| Ferm Z₂ | even/odd parity | 0 |
| Band U(1)⊗U(1) | (±1, ±1) | (0, 0) |
| Band Z₂⊗U(1) | (parity, spin) | (0, 0) |
| Band U(1)⊗SU(2) | (charge, spin) | (0, 0) |
| Band Z₂⊗SU(2) | (parity, spin) | (0, 0) |

### Bond Sectors in Random Mode

Bond sectors for the random MPS (`bond_dim>1`) are found by BFS of depth 2
from Q_vac. The table below lists the resulting sector sets (for spin-½ where
applicable):

| Space | Bond sectors | Count |
|-------|-------------|-------|
| Spin U(1) | −2, −1, 0, +1, +2 | 5 |
| Spin SU(2) | 2J = 0, 1, 2 | 3 |
| Ferm U(1) | −2, −1, 0, +1, +2 | 5 |
| Ferm Z₂ | 0, 1 | 2 |
| Band U(1)⊗U(1) | (c, s) with \|c\| + \|s\| ≤ 2 | 13 |
| Band Z₂⊗U(1) | (0, 0), (0, ±2), (1, ±1) | 5 |
| Band U(1)⊗SU(2) | (c, 2J) with \|c\| ≤ 2, 2J ≤ 2 | 9 |
| Band Z₂⊗SU(2) | (0, 0), (0, 2), (1, 1) | 3 |

## See Also

- [MPS](mps.md) — the returned object type.
- [build_bosonic](../local-space/build-bosonic.md),
  [build_fermionic](../local-space/build-fermionic.md),
  [build_conductor](../local-space/build-conductor.md) — wrappers around
  Nicole's `load_space` that return `(Spc, Op)`.
- [dmrg.run](../dmrg/run.md) — optimize the initialized MPS.
