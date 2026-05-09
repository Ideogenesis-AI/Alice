# Local Space

The local space builders construct the physical Hilbert space and a dictionary of MPO operator templates for a single lattice site. They are called internally by the model builders but can also be used directly for custom physics.

## Functions

| Function | Description |
|----------|-------------|
| [build_bosonic](build-bosonic.md) | Spin-`s` site (bosonic) |
| [build_fermionic](build-fermionic.md) | Spinless-fermion site |
| [build_conductor](build-conductor.md) | Spinful-fermion (Band) site |

## Return Convention

All three functions return `(Spc, Op)`:

- `Spc` — Nicole `Index` representing the physical Hilbert space.
- `Op` — dict of named Nicole `Tensor` objects, including 4th-order MPO operator templates.

Operator templates follow the axis convention:

| Template suffix | Axes | Role |
|----------------|------|------|
| `4` | `(L_trivial_IN, op_OUT, bra_OUT, ket_IN)` | Leading-site template |
| `4dag` | `(op_IN, R_trivial_OUT, bra_OUT, ket_IN)` | Terminal-site template |
| `4` (on-site) | `(L_trivial_IN, R_trivial_OUT, bra_OUT, ket_IN)` | On-site template (`I4`, `N4`, etc.) |
| `4mid` | `(op_IN, op_OUT, bra_OUT, ket_IN)` | Intermediate-site (string) template |

## See Also

- [Custom local space example](../../examples/extensions/custom-space.md)
- [build_heisenberg](../hamiltonian/build-heisenberg.md), [build_free_fermion](../hamiltonian/build-free-fermion.md), [build_hubbard](../hamiltonian/build-hubbard.md) — model builders that call the space builders.
