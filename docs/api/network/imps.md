# iMPS

Infinite matrix product state: a unit cell of `N` site tensors that tiles the chain.

::: alice.iMPS
    options:
      show_source: false
      heading_level: 2

## Notes

The cell is cyclic, so bond `N` is bond `0`. That single fact is what separates an `iMPS` from an [MPS](mps.md) of the same length: there is no first bond and no last bond, nothing is open, and the bond charges have to return to themselves after one cell. A cell therefore carries no net physical charge, which is half filling for fermions and zero magnetization for spins. [init_imps](init-imps.md) enforces that when it builds one.

Gauge fixing replaces canonicalization. `canonical(0)` solves for the gauge that makes the cell left- and right-orthogonal simultaneously, by finding the fixed point of the transfer operator rather than sweeping orthogonality in from an end. It raises when no such gauge exists, which a random cell can land on: the transfer operator then has two dominant eigenvalues of opposite sign and the cell maps two classes of bond charge into each other rather than onto themselves.

## See Also

- [iNetwork](inetwork.md) — base class providing bond bookkeeping and `serialize()`.
- [iMPO](impo.md) — infinite matrix product operator.
- [MPS](mps.md) — the finite counterpart.
- [init_imps](init-imps.md) — build an initial unit cell.
- [vumps.run](../../algorithms/vumps/run.md) — optimize an iMPS with VUMPS.
