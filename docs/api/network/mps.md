# MPS

Matrix product state.

::: alice.MPS
    options:
      heading_level: 2

## String Representation

`repr(mps)` (and therefore the interactive display in notebooks and REPLs) renders
a text diagram of the chain:

```
                 Matrix Product State (MPS)                 
                 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾                 
           ○───○───○───○─···─⊙─···─○───○───○───○            
           ╵   ╵   ╵   ╵     ╵     ╵   ╵   ╵   ╵            
               length: 100     max bond: 64
               center: 49      norm: 1.00e+00
```

**Chain row** — each node is one site; consecutive sites are joined by `───`.
When the chain is too long to show in full, a gap `─···─` is inserted and only
a subset of sites near the edges and center is displayed.

**Physical-index row** — the downward stub `╵` beneath each node represents the
physical index of that site.

**Node symbols:**

| Symbol | Meaning |
|--------|---------|
| `○` | Regular site |
| `⊙` | Orthogonality center (`center` attribute) |

**Info block** — `length` is the number of sites `L`; `max bond` is the largest
bond dimension across all virtual indices; `center` is the orthogonality center
index (`None` if not set); `norm` is `‖ψ‖`.

## See Also

- [Network](network.md) — base class providing `canonical()`, `norm()`, `serialize()`, and more.
- [MPO](mpo.md) — matrix product operator.
- [observe](observe.md) — compute ⟨ψ\|O\|ψ⟩ for an MPS state.
- [dmrg.run](../../algorithms/dmrg/run.md) — optimize an MPS with DMRG.
