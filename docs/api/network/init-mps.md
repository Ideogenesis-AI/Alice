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

### Logic Overview

The diagram below summarizes how `target_qn`, `config`, `Q[L]`, and
`bond_dim` interact inside `init_mps`.

```
 target_qn given?
 ├─ No  ──► target = Q_vac  (half-filling default)
 └─ Yes ──► target = target_qn

 config given?
 ├─ No  ──► cfg = _auto_config(L, Spc, group, Q_vac, target)
 │             tries: single-sector fill → period-2 alternation → greedy
 └─ Yes ──► cfg = config  (used verbatim)

 Q = _bond_charges(cfg, Q_vac)   →  Q[0..L]

 Q[L] == target?
 ├─ Yes ──► (no action)
 └─ No  ──┬─ target was defaulted (Q_vac) ──► WARNING  (e.g. odd L, no target_qn given)
          └─ target was explicit          ──► ValueError  (physically unreachable)

 effective_right = target  if target was given explicitly
                 = Q[L]    if target was defaulted  (crucial for odd-L random MPS)

 bond_dim == 1?
 ├─ Yes ──► product_state_mps(cfg, Q)   right boundary = Q[L]  (rigid)
 └─ No  ──► random_mps(Q, bond_dim, effective_right)
               BFS seed  = Q[L//2]   (center-bond charge from cfg)
               right pin = effective_right
```

The key asymmetry is that in product-state mode the right boundary is always
`Q[L]` — it cannot be overridden because every intermediate bond charge is
fixed by the charge path. In random mode the right boundary is pinned
explicitly to `effective_right`, which decouples it from the config path.

### Charge Conventions

All standard Nicole physical spaces define charges relative to the half-filled
reference, so the center-bond charge for a balanced even-L auto-config is zero:

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
from `Q_c = Q[L//2]`. The table below lists the resulting sector sets for a
Q_vac-targeted auto-config (spin-½ where applicable):

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

When `target_qn` is given, `Q_c = Q[L//2]` is computed from the
`target_qn`-targeted auto-config, which shifts the center-bond charge toward
the requested sector.

### Odd-chain lengths

For odd `L`, no site configuration can return the bond charge to `Q_vac` in an
odd number of fusion steps with the standard physical sectors. The three exact
strategies in `_auto_config` (single-sector fill and period-2 alternation) all
fail, and the greedy fallback produces `Q[L] ≠ Q_vac`.

**Default behavior** (`target_qn` not given): `init_mps` targets `Q_vac` and,
finding `Q[L] ≠ Q_vac`, logs a `WARNING` that multiple target sectors may
exist and recommends passing `target_qn` explicitly. Both modes then use the
greedy's `Q[L]` as the effective right boundary.

**Explicit `target_qn`**: pass the desired sector directly. For Abelian
2-sector spaces the greedy reaches the target exactly when it is achievable
(correct parity and magnitude). If the requested sector is unreachable — e.g.
`target_qn=0` for odd-L spin-½ — `init_mps` raises a `ValueError` in both
modes rather than silently constructing an MPS in the wrong sector.

```python
# Spin-½ U(1), L=7 — both modes work with the same target_qn
Spc, Op = load_space('Spin', 'U1', {'J': 0.5})

# Product state in the Sz = +½ sector
mps_ps = init_mps(7, Spc, Op, bond_dim=1,  target_qn=1)

# Random MPS in the Sz = +½ sector, bond dimension 32
mps_rd = init_mps(7, Spc, Op, bond_dim=32, target_qn=1)
```

For Abelian spaces the two valid targets for odd-L spin-½ are `target_qn=+1`
(Sz = +½) and `target_qn=-1` (Sz = −½). Any other value is physically
unreachable and raises `ValueError`.

## See Also

- [MPS](mps.md) — the returned object type.
- [build_bosonic](../local-space/build-bosonic.md),
  [build_fermionic](../local-space/build-fermionic.md),
  [build_conductor](../local-space/build-conductor.md) — wrappers around
  Nicole's `load_space` that return `(Spc, Op)`.
- [dmrg.run](../../algorithms/dmrg/run.md) — optimize the initialized MPS.
