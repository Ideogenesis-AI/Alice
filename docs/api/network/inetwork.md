# iNetwork

Base class for 1D tensor network unit cells with infinite boundaries.

::: alice.iNetwork
    options:
      show_source: false
      heading_level: 2
      members:
        - N
        - L
        - bc
        - center
        - bond_dims
        - bond_states
        - phys_dims
        - itag_prefix
        - to
        - serialize
        - deserialize

## Notes

The cell is cyclic: bond `N` is bond `0`, and the constructor checks that the
right index of each site matches the left index of the next, the last site
meeting the first. `bc` is fixed to `'IBC'` for the whole family, so code that
dispatches on boundary conditions can treat finite and infinite networks
alike.

Canonicalization does not live here, unlike in [Network](network.md). An
infinite cell has no end to sweep orthogonality in from, so the gauge is
solved for rather than swept, and [iMPS](imps.md) carries that machinery
alone. An [iMPO](impo.md) is never canonicalized at all.

## See Also

- [iMPS](imps.md) — infinite matrix product state (3-index site tensors).
- [iMPO](impo.md) — infinite matrix product operator (4-index site tensors).
- [Network](network.md) — the finite counterpart.
