# iMPS Initialization

Construct a symmetry-aware initial unit cell for VUMPS.

::: alice.network.autoimps.init_imps
    options:
      show_source: false
      heading_level: 2

## Notes

`init_imps` operates in the same two modes as [init_mps](init-mps.md), selected by `bond_dim`.

**Product cell (`bond_dim=1`)** — every bond carries a single sector, fixed by propagating the physical charges of the chosen site configuration through the group fusion rules. The one-site scheme can never leave a cell this narrow, so pair it with `scheme='2s'`, which widens it from the first iteration.

**Random cell (`bond_dim>1`)** — each bond is given the charges reachable at its own position, and the tensors are filled with random entries. Ready for either scheme.

Both modes are particle-type agnostic: the `(Spc, Op)` pair returned by Nicole's `load_space` encodes all the symmetry information, so no `spin=`, `symmetry=`, or `particle_type=` argument is needed.

### What the tiling demands

Bond `N` is bond `0`, and two things follow that have no counterpart in `init_mps`.

**The cell is charge neutral.** Fusing the physical charges of the cell has to give back the charge it started from, which in Alice's particle-hole symmetric conventions means half filling for fermions and zero magnetization for spins. This is a property of a cell rather than a choice, so there is no `target_qn` argument: a doped or magnetized uniform state would need the bond charges to come back shifted, which is not a cell that tiles. `config` is what chooses the pattern within the cell, and a `config` carrying net charge is refused. So is a cell length that admits no neutral configuration at all, an odd one for spin-1/2.

**Each bond carries only the charges reachable at its own position.** Bond `k` sits after `k` physical sites, so it can only hold charges an even number of fusion steps away from the ones its neighbors hold. Giving every bond the same sectors instead produces a transfer operator with two dominant eigenvalues of opposite sign, which [iMPS](imps.md) gauge fixing reports as a two-cycle.

### Random cells that cannot be gauge fixed

Random amplitudes land often enough on a transfer operator whose two largest eigenvalues are equal and opposite, or too close to separate within the sweep budget, and neither is a cell a gauge can be found for. `init_imps` repeats the draw with a shifted seed when that happens, up to eight times, which stays deterministic for a given `seed`. If all of them fail, the error suggests the three ways out: another seed, a different width, or `bond_dim=1` with `scheme='2s'`, which grows the width from a product state instead.

## See Also

- [iMPS](imps.md) — the returned type.
- [init_mps](init-mps.md) — the finite counterpart.
- [build_ihamiltonian](../hamiltonian/build-ihamiltonian.md) — build the matching Hamiltonian iMPO.
- [vumps.run](../../algorithms/vumps/run.md) — consumes the cell.
