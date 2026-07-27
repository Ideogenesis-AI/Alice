# thermal_mpo

Approximate the thermal density matrix \(\rho(\beta) = e^{-\beta H}\) as a `NormalMPO` via a truncated Taylor series.

::: alice.network.thermal_mpo
    options:
      show_source: false
      heading_level: 2

## Notes

**Algorithm.** The Taylor expansion

\[\rho(\beta) = \sum_{n=0}^{N} \frac{(-\beta)^n}{n!} H^n\]

is accumulated iteratively. At each step the running sum `rho` and the current power `H^n` are compacted — bond dimension is truncated and the norm is folded into the tracked `log_scale` — preventing exponential growth of virtual bonds.

**Convergence.** The expansion converges when \(|\beta \lambda_{\max}| \ll N\), where \(\lambda_{\max}\) is the spectral radius of \(H\). In practice `order=20`–`25` is sufficient for \(\beta \|H\| \lesssim 4\).

**Partition function.** After the call, `rho.trace()` returns \(Z(\beta) = \operatorname{Tr}[e^{-\beta H}]\).

**Thermal expectation values.** Pass `rho` to `observe(rho, O)` to compute \(\langle O \rangle_\beta = \operatorname{Tr}[\rho O] / \operatorname{Tr}[\rho]\).

## See Also

- [NormalMPO](normal-mpo.md) — the return type; tracks `log_scale` separately from unit-normed tensors (sign lives in the tensor data).
- [observe](observe.md) — compute `Tr[ρ O] / Tr[ρ]` from the returned `NormalMPO`.
- [build_hamiltonian](../hamiltonian/build-hamiltonian.md) — construct the input Hamiltonian MPO.
