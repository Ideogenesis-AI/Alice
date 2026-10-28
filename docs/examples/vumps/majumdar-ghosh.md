# Majumdar-Ghosh

The J1-J2 spin-1/2 chain at the point where the ground state is known in closed form.

\[
H = J_1 \sum_i \mathbf{S}_i \cdot \mathbf{S}_{i+1}
  + J_2 \sum_i \mathbf{S}_i \cdot \mathbf{S}_{i+2}, \quad J_2 = J_1 / 2
\]

At \(J_2 = J_1/2\) the ground state is a product of nearest-neighbor singlets and the energy per site is exactly \(-3J_1/8 = -0.375\). This is the only example here whose exact ground state lies inside the variational class, so VUMPS reproduces it rather than converging towards it: the error reaches machine precision and the gradient reaches `grad_tol`, instead of settling on the floor a bond cap imposes in the [Heisenberg](heisenberg.md), [free-fermion](free-fermion.md), and [Hubbard](hubbard.md) examples.

The dimer covering is a uniform MPS of bond dimension two on a cell of two sites. Under SU(2) that is a single spin-1/2 multiplet per bond, so bond dimension one.

## 1. Setup

```python
import alice
alice.configure_logging()

from alice import build_interaction, build_ihamiltonian, init_imps, vumps
from nicole import load_space
```

## 2. Build the Hamiltonian iMPO

The next-nearest-neighbor coupling is `Jp` in the model, and the geometry has to emit those bonds with `n3x`:

```python
config = {
    "geometry": {"lattice": "chain", "lx": 2, "bcx": "IBC",
                 "n2x": True, "n3x": True},
    "model": {"category": "bosonic", "label": "Heisenberg",
              "symmetry": "SU2", "spin": 0.5, "J": 1.0, "Jp": 0.5},
}
interactions, spc, geo = build_interaction(config)
impo = build_ihamiltonian(interactions, geo.L, spc)
```

## 3. Build an initial unit cell

```python
Spc, Op = load_space('Spin', 'SU2', {'J': 0.5})
imps = init_imps(geo.L, Spc, Op, bond_dim=2)
```

## 4. Run VUMPS

```python
opts = vumps.Options(
    scheme   = '2s',
    n_iters  = 30,
    max_bond = 2,
    grad_tol = 1e-10,
    env_tol  = 1e-12,
)

summary = vumps.run(imps, impo, opts)

exact = -3.0 / 8.0
print(f"VUMPS energy per site: {summary.energy:.12f}")
print(f"Exact energy per site: {exact:.12f}")
print(f"Difference:            {summary.energy - exact:+.2e}")
print(f"Gradient:              {summary.gradients[-1]:.2e}")
```

Seven iterations and a few seconds later the energy is `6e-14` from exact and the gradient has fallen to `2e-11`, below `grad_tol`, so the run reports itself converged and stops. Note that `env_tol` is left at the library default rather than loosened to `1e-8` as on the critical chains: the dimer state has no correlations beyond a single bond, so the environment fixed point is reached almost immediately.

## 5. What the two symmetries cost

Running the same thing under U(1) gives:

```bash
uv run python examples/examples_vumps/vumps_mgdimer.py --symmetry U1
```

The gradient still falls to `1.3e-11`, but the energy stops `8.6e-9` above exact. That gap is the precision of Alice's spin tables, not a failure of the algorithm — the state is converged, and the number it is converged to is slightly off. SU(2) avoids it because the Clebsch-Gordan tensors are built exactly.

## 6. The product start, and the degeneracy

A cell of width one is already the dimer state under SU(2), so there is nothing left to do:

```bash
uv run python examples/examples_vumps/vumps_mgdimer.py --init product
```

One iteration, a difference of `1.8e-15`, and the bonds never widen past one multiplet. This is the fixture the test suite leans on for exactly that reason.

The ground state is twofold degenerate: the dimers can cover the odd bonds or the even ones, at the same energy. A cell of two sites silently picks whichever the initial state leans towards, and the two answers are indistinguishable by energy. That degeneracy has a visible consequence — it slows the transfer-operator fixed point, so a cell of four sites or a bond dimension of eight, both far more than the state needs, make the environment solver run out its budget and warn:

```bash
uv run python examples/examples_vumps/vumps_mgdimer.py --cell 4
uv run python examples/examples_vumps/vumps_mgdimer.py --bond-dim 8
```

Both still converge, to `6.9e-13` and `9.2e-14`. The warning is about the environment tensor, not the energy. [Lessons](../../algorithms/vumps/lessons.md#the-environment-budget) has the measurements behind this.

## Command-line script

```bash
uv run python examples/examples_vumps/vumps_mgdimer.py
uv run python examples/examples_vumps/vumps_mgdimer.py --symmetry U1
uv run python examples/examples_vumps/vumps_mgdimer.py --init product
uv run python examples/examples_vumps/vumps_mgdimer.py --cell 4 --history
```

Detuning `--J2` away from `0.5` leaves the Majumdar-Ghosh point, where the dimer state is no longer exact and the script reports no reference.

## See Also

- [Heisenberg chain](heisenberg.md)
- [Hubbard model](hubbard.md)
- [VUMPS overview](../../algorithms/vumps/index.md)
- [Lessons](../../algorithms/vumps/lessons.md)
