# MPO

Matrix product operator.

::: alice.MPO
    options:
      heading_level: 2
      members:
        - compact
        - redistribute_norm

## String Representation

`repr(mpo)` (and therefore the interactive display in notebooks and REPLs) renders
a text diagram of the chain:

```
               Matrix Product Operator (MPO)                
               ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾                
           ╷   ╷   ╷   ╷     ╷     ╷   ╷   ╷   ╷            
           □───□───□───□─···─⊡─···─□───□───□───□            
           ╵   ╵   ╵   ╵     ╵     ╵   ╵   ╵   ╵            
               length: 100     max bond: 64
               center: 49      norm: 1.00e+00
```

**Chain row** — each node is one site; consecutive sites are joined by `───`.
When the chain is too long to show in full, a gap `─···─` is inserted and only
a subset of sites near the edges and center is displayed.

**Physical-index rows** — the downward stub `╵` beneath each node is the
physical-input (ket) index; the upward stub `╷` above is the physical-output
(bra) index.

**Node symbols:**

| Symbol | Meaning |
|--------|---------|
| `□` | Regular site |
| `⊡` | Orthogonality center (`center` attribute) |

**Info block** — `length` is the number of sites `L`; `max bond` is the largest
bond dimension across all virtual indices; `center` is the orthogonality center
index (`None` if not set); `norm` is `‖O‖`.

## See Also

- [Network](network.md) — base class providing `canonical()`, `norm()`, `serialize()`, and more.
- [MPS](mps.md) — matrix product state.
- [build_hamiltonian](../hamiltonian/build-hamiltonian.md) — creates an MPO from a list of interactions.
- [observe](observe.md) — use this MPO as an observable.
