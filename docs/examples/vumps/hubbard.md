# Hubbard Chain

Ground-state VUMPS for the spinful Hubbard chain at half filling, directly in the thermodynamic limit.

\[
H = -t \sum_{\langle i,j \rangle, \sigma} (c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.})
  + U \sum_i n_{i\uparrow} n_{i\downarrow}
  - \frac{U}{2} \sum_i n_i , \quad t = 1
\]

The chain is Bethe-ansatz integrable, so there is an exact answer to compare against at every \(U\), not only at \(U = 0\). At half filling the Lieb–Wu solution gives the ground-state energy per site as a single integral,

\[
e(U) = -4t \int_0^\infty \frac{J_0(\omega) \, J_1(\omega)}
                              {\omega \left(1 + e^{\omega U / 2t}\right)} \, d\omega ,
\]

which needs only a quadrature, not a solution of coupled equations. At \(U = 0\) it collapses to \(-4t/\pi \approx -1.273240\), twice the spinless chain.

The last term is the half-filling chemical potential that Alice's Hubbard builder applies to keep the model particle-hole symmetric. At \(\langle n \rangle = 1\) it contributes exactly \(-U/2\) per site, so the energy VUMPS reports sits that far below \(e(U)\), and the comparison has to shift the reference by the same amount.

## 1. Setup

```python
import alice
alice.configure_logging()

from alice import build_interaction, build_ihamiltonian, init_imps, vumps
from nicole import load_space
```

## 2. Build the Hamiltonian iMPO

```python
config = {
    "geometry": {"lattice": "chain", "lx": 2, "bcx": "IBC", "n2x": True},
    "model": {"category": "conductor", "label": "Hubbard",
              "symmetry": "Z2,SU2", "t": 1.0, "U": 4.0, "mu": 0.0},
}
interactions, spc, geo = build_interaction(config)
impo = build_ihamiltonian(interactions, geo.L, spc)
```

All four band symmetries work: `U1,U1`, `Z2,U1`, `U1,SU2`, and `Z2,SU2`. They differ in cost by more than they differ in physics. At a unit cell of two sites and `max_bond = 16`, one iteration takes about 15 s under `Z2,SU2`, 18 s under `U1,U1`, and 53 s under `U1,SU2`, and the SU(2) groups reach a lower energy at the same bond dimension because each multiplet carries several states. `Z2,SU2` is the cheapest of the four and the default in the script.

## 3. Build an initial unit cell

```python
Spc, Op = load_space('Band', 'Z2,SU2')
imps = init_imps(geo.L, Spc, Op, bond_dim=1)
```

Half filling is not a setting here, it is what a unit cell is. The bond charge has to return to itself after `N` sites, so the cell carries no net charge, which for a band means one electron per site on average. That is exactly the filling the Lieb–Wu integral describes.

Note the `bond_dim=1`: a product cell, one electron per site, which the two-site scheme widens over the first few iterations. This is the opposite of the choice in the [spin](heisenberg.md) and [spinless](free-fermion.md) examples, and it matters more here. With four states per site a random cell of a given width spreads its weight over many more charge sectors than the ground state uses, and the two-site splits spend their iterations undoing that.

## 4. Run VUMPS

```python
opts = vumps.Options(
    scheme   = '2s',
    n_iters  = 20,
    max_bond = 16,
    grad_tol = 1e-10,
    env_tol  = 1e-8,
)

summary = vumps.run(imps, impo, opts)

exact = lieb_wu_energy(U=4.0) - 4.0 / 2.0
print(f"VUMPS energy per site: {summary.energy:.10f}")
print(f"Exact energy per site: {exact:.10f}")
print(f"Error:                 {summary.energy - exact:+.2e}")
```

`lieb_wu_energy` is in the script: it evaluates the integral above with a Bessel function built from its own periodic integral representation, so the example needs nothing beyond NumPy.

At `max_bond = 16` the run lands `2.6e-4` above the exact value, with the energy stable to ten digits and the gradient flat at `1.3e-3`. That floor is the two-site scheme working against a bond cap that bites, with a discarded weight of `3.4e-5` per split, and it is what convergence looks like here rather than a sign of trouble.

## 5. Two contrasts worth running

The same run started from a random cell of width 16 reaches only `1.2e-3`, with a gradient of `4.4e-3` and five times the discarded weight:

```bash
uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 16 --init random
```

Both runs are converged in the sense that their energies have stopped moving; they have simply converged to different states, and the product start found the better one. Widening the bond is what closes the remaining gap in either case.

Setting `U = 0` turns the Mott insulator into a metal and the exact answer into \(-4t/\pi\):

```bash
uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 16 --U 0
```

It is the easier Hamiltonian but the harder state: gapless, so the entanglement keeps growing with the bond dimension instead of saturating, and at `max_bond = 16` the error is `2.5e-3`, twice that of the gapped case.

## Command-line script

```bash
uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 16
uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 32 --n-iters 30
uv run python examples/examples_vumps/vumps_hubbard.py --symmetry U1,SU2 --bond-dim 16
uv run python examples/examples_vumps/vumps_hubbard.py --cell 2 --bond-dim 16 --history
```

## See Also

- [Heisenberg chain](heisenberg.md)
- [Free fermion](free-fermion.md)
- [VUMPS overview](../../algorithms/vumps/index.md)
- [Lessons](../../algorithms/vumps/lessons.md)
- [DMRG on the finite chain](../dmrg/hubbard.md)
