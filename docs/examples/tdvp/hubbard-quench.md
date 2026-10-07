# Hubbard Interaction Quench

An interaction quench from a DMRG ground state, run in all four band symmetry
groups — the example that validates TDVP where no closed-form reference exists.

\[
|\psi(0)\rangle = \text{ground state of } H(U_i), \qquad
|\psi(t)\rangle = e^{-i H(U_f) t} |\psi(0)\rangle
\]

The state is prepared as the ground state at one interaction strength and then
evolved under a *different* one. It is not stationary under the new
Hamiltonian, so it relaxes — and the quantity that moves is the double
occupancy \(D(t) = \sum_j \langle n_{j\uparrow} n_{j\downarrow}\rangle\), the
observable the quench directly drives.

## Why this example covers all four symmetries

The [U = 0 melting example](hubbard.md) starts from a product state, which
restricts it to the `U1,*` groups: under `Z2` charge the even-parity sector
merges \(|0\rangle\) with \(|\!\uparrow\downarrow\rangle\) into a single
two-dimensional sector, and a site configuration cannot name them apart.

A ground state has no such problem. It is a singlet reachable by iterative
diagonalization followed by DMRG, both of which already support every group, so
this example exercises `'U1,U1'`, `'Z2,U1'`, `'U1,SU2'`, and `'Z2,SU2'`.

## Three checks without a reference solution

**Energy conservation.** The generator is \(H(U_f)\), so \(\langle H(U_f)
\rangle\) must hold fixed while \(\langle H(U_i)\rangle\) moves freely. The
conserved value is *not* the DMRG ground-state energy unless the quench is
trivial — a distinction worth keeping straight when reading the output.

**The eigenstate limit.** With \(U_f = U_i\) the initial state is an exact
eigenvector, so nothing may happen beyond a global phase:

\[
e^{-iHt}|E\rangle = e^{-iEt}|E\rangle
\]

The magnitude \(|\langle\psi(0)|\psi(t)\rangle| = 1\) pins the state, and the
phase pins the sign convention of the propagator. This is the sharpest check
available here, and the only one sensitive to that sign — `exp(+iHt)` conserves
energy and norm just as faithfully.

**Cross-symmetry agreement.** The four groups share no tensor structure: two
carry intertwiners and two do not, two resolve \(S_z\) and two only total spin,
and `Z2` conserves fermion parity rather than particle number. Identical
observables from four genuinely different bookkeepings is strong evidence that
all of them are right.

## 1. Prepare the ground state

DMRG runs in real arithmetic — its environments are built from a real identity
and it has no reason to be complex — and `tdvp.run` promotes the state to
`complex128` itself, so the handoff needs nothing from the caller:

```python
import alice
alice.configure_logging()

from alice import MPS
from alice.network import build_hamiltonian, build_interaction, observe
from alice.algorithm import dmrg, tdvp
from conductor import iter_diag_band   # tests/diagonaliztn/conductor.py

L, t, U_init, U_final, symmetry = 6, 1.0, 2.0, 6.0, "U1,SU2"

# Iterative diagonalization of the U = 0 chain: a good guess at any U, and
# available in every symmetry group.
_, _, tensors = iter_diag_band(N=L, Nkeep=64, t=t, symmetry=symmetry,
                               verbose=False)
state = MPS(tensors, center=None)
state.canonical(0)
state.normalize()

def hubbard_mpo(U):
    cfg = {
        "geometry": {"lattice": "chain", "lx": L, "bcx": "OBC", "n2x": True},
        "model": {"category": "conductor", "label": "Hubbard",
                  "symmetry": symmetry, "t": t, "U": U, "mu": 0.0},
    }
    interactions, spc, geo = build_interaction(cfg)
    return build_hamiltonian(interactions, geo.L, spc)

summary = dmrg.run(state, hubbard_mpo(U_init),
                   dmrg.Options(scheme='2s', n_sweeps=12, max_bond=64,
                                e_tol=1e-10))
state = summary.state
```

## 2. Snapshot the initial state

`tdvp.run` evolves in place, so the overlap check needs its own copy. Clone
tensor by tensor rather than with `copy.deepcopy`, which cannot pickle the
`CGSpec` held by a non-Abelian tensor's intertwiners:

```python
reference = MPS([state[i].clone() for i in range(L)], center=state.center)
reference.to_complex()
```

## 3. Quench

```python
mpo = hubbard_mpo(U_final)
e_conserved = observe(state, mpo)   # not the DMRG ground-state energy

opts = tdvp.Options(scheme='2s', dt=0.025, n_steps=40, order=2, max_bond=64)
summary, artifact = tdvp.run(
    tdvp.Artifact(state=state, time=0.0, step=0), mpo, opts)
```

Unlike a product-state quench, `scheme='1s'` is usable here — the ground state
already has open bonds for the 1-site manifold to rotate within.

## Running the script

```bash
uv run python examples/examples_tdvp/tdvp_hubbard_quench.py \
    --symmetry U1,SU2 --U-init 2.0 --U-final 6.0 --L 6 --time 0.5
```

```text
Quench         : U = 2.0  ->  U = 6.0
Symmetry       : U1,SU2   (non-Abelian, particle number conserved)

DMRG ground state at U = 2.0
  energy <H(U_init)>  : -10.5463137944
  energy <H(U_final)> : -18.7450277022  (this is what must stay fixed)
  double occupancy    : 0.9503215231

     t          D(t)     E drift             N  |<psi0|psi>|      S_max   mult  states
----------------------------------------------------------------------------------------
 0.150    0.73514320   +6.86e-13    6.00000000   0.911912709   1.283853     35      64
 0.300    0.34184299   +1.75e-12    6.00000000   0.741226468   1.162415     35      64
 0.450    0.18988092   +3.13e-12    6.00000000   0.630599153   1.120950     35      64
 0.500    0.21791497   +3.74e-12    6.00000000   0.617098799   1.141494     35      64
```

The double occupancy falls from 0.950 to 0.190 and rebounds to 0.218 —
stronger repulsion suppressing the doubly-occupied configurations, then
overshooting into an oscillation. Particle number stays exact, the energy of
the evolving Hamiltonian holds to 4e-12, and the overlap with the initial state
decays to 0.617, confirming the quench genuinely moved the state rather than
only accumulating a phase.

## The eigenstate-phase check

```bash
uv run python examples/examples_tdvp/tdvp_hubbard_quench.py \
    --symmetry Z2,SU2 --U-init 4.0 --eigenstate --time 0.3
```

```text
     t          D(t)     E drift             N  |<psi0|psi>|      S_max   mult  states
----------------------------------------------------------------------------------------
 0.150    0.53779226   +6.43e-13    6.00000000   1.000000000   1.132930     35      64
 0.300    0.53779226   +1.33e-12    6.00000000   1.000000000   1.132930     35      64

Eigenstate-phase check  (U_final == U_init)
  |<psi(0)|psi(t)>|   : 1.000000000000  (exact 1)
  phase    measured   : -0.183572399-0.983006192i
  phase    exp(-iEt)  : -0.183572399-0.983006192i
  deviation           : 4.904e-14
```

Run in the hardest group, `Z2,SU2`, the state does not budge — double occupancy
frozen to all ten digits — and the accumulated phase matches \(e^{-iEt}\) to
4.9e-14. This verifies the propagator's sign through the full non-Abelian
machinery.

## Four-way comparison

```bash
uv run python examples/examples_tdvp/tdvp_hubbard_quench.py \
    --compare-symmetries --U-init 2.0 --U-final 6.0 --L 6 --time 0.5
```

```text
  symmetry     <H(U_final)>             N             D    |overlap|   mult  states
--------------------------------------------------------------------------------------
     U1,U1   -18.7450277022    6.00000000    0.21791497   0.61709880     64      64
     Z2,U1   -18.7450277022    6.00000000    0.21791497   0.61709880     64      64
    U1,SU2   -18.7450277022    6.00000000    0.21791497   0.61709880     35      64
    Z2,SU2   -18.7450277022    6.00000000    0.21791497   0.61709880     35      64

Deviations from U1,U1
  symmetry           dE           dN           dD     max dn_j       max dS
----------------------------------------------------------------------------
     Z2,U1    +1.05e-11    +6.84e-14    +2.89e-12     6.79e-13     3.59e-11
    U1,SU2    +4.31e-12    -1.37e-12    -5.99e-14     2.18e-12     3.99e-11
    Z2,SU2    +4.53e-12    -1.38e-12    -8.35e-13     5.00e-13     4.61e-11
```

All four agree to roughly 1e-11, and that floor is the *initial state*, not the
evolution: each group runs its own DMRG to a finite `e_tol`, so the four
starting points already differ at that level and TDVP merely carries the
difference forward. Tightening `e_tol` tightens the whole table.

## See Also

- [Hubbard model](hubbard.md) — U = 0 melting with an exact analytic reference.
- [DMRG Hubbard](../dmrg/hubbard.md) — the ground-state calculation used here.
- [tdvp.run](../../algorithms/tdvp/run.md) — in-place evolution semantics.
- [tdvp.Artifact](../../algorithms/tdvp/artifact.md) — archiving states for later measurement.
