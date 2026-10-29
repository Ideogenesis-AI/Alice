# Lessons

Hard-won notes on running VUMPS — behavior that is correct by implementation but
easy to misread, and the practical consequences for how a run should be set up
and judged. To be updated over time.

## The Environment Budget

`env_maxiter` running out is the most common warning a VUMPS run produces, and
the one most easily misread. It does not mean the fixed point is unreachable,
and it usually does not mean the energy is wrong.

### What the residual measures

The environment on each side is the fixed point of a regularized power
iteration: a block is carried around the unit cell, the diverging channel is
projected out, and the cycle repeats until the block stops moving. The number
compared against `env_tol` is the change in the block over one cell,
\(\lVert E_{k+1} - E_k \rVert\), and it falls geometrically,

\[
\lVert E_{k+1} - E_k \rVert \sim \lambda_2^{\,k},
\]

with \(\lambda_2\) the subleading eigenvalue of the MPS transfer operator.
Writing `env_tol` as \(\epsilon\), the budget it needs therefore follows from
the correlation length \(\xi = -1/\ln \lambda_2\), counted in unit cells
rather than sites:

\[
k \;\approx\; \xi \, \ln\frac{1}{\epsilon} \;\approx\; 28\,\xi
\quad \text{at} \quad \epsilon = 10^{-12}.
\]

The default `env_maxiter = 200` covers \(\xi\) up to about seven unit cells.

### Nothing stalls

Measured residuals after a given number of absorbed cells, for states
converged under the two-site scheme on a two-site cell, with the solver run to
a fixed budget rather than to a tolerance:

<div class="data-table label-col" markdown>

| model | D | 25 cells | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|---|
| Heisenberg S=1/2 (gapless) | 32 | 3.2e-3 | 1.2e-3 | 1.9e-4 | 4.7e-6 | 2.9e-9 | 1.3e-15 |
| Majumdar-Ghosh (gapped) | 8 | 4.8e-1 | 1.9e-1 | 3.1e-2 | 7.8e-4 | 5.1e-7 | 2.2e-13 |
| Majumdar-Ghosh (gapped) | 32 | 1.2e-1 | 2.4e-3 | 9.2e-7 | 1.2e-13 | — | — |
| Heisenberg S=1 (Haldane gap) | 32 | 1.3e-9 | 6.5e-14 | — | — | — | — |

</div>

Every case reaches machine precision given enough cells. There is no residual
floor short of it, so `env_tol = 1e-12` is a meaningful default rather than an
unreachable one, and the question is only ever how many cells it costs.

### Gapped models saturate with the bond dimension

For a gapped state, \(\lambda_2\) converges to a physical value once the bond
dimension is large enough to capture the correlation length, and widening the
state further does not slow the iteration down. The Haldane chain, which is
gapped with a unique ground state:

<div class="data-table" markdown>

| D | 5 cells | 10 | 20 | 40 | \(\lambda_2\) | cells for 1e-12 |
|---|---|---|---|---|---|---|
| 16 | 2.5e-2 | 2.6e-5 | 1.3e-7 | 7.9e-12 | 0.62 | ~44 |
| 32 | 2.8e-1 | 1.9e-3 | 5.9e-8 | 3.4e-12 | 0.61 | ~43 |
| 64 | 6.7e-1 | 2.5e-2 | 5.1e-6 | 1.6e-13 | 0.42 | ~38 |
| 128 | 1.1e+0 | 1.2e-1 | 8.7e-5 | 3.0e-11 | 0.48 | ~45 |

</div>

Flat across a factor of eight in `D`, at roughly a fifth of the default
budget. A production run on a gapped model at a bond dimension in the hundreds
is in the same position as this one at 16.

A critical model has no such saturation. The correlation length of a uniform
MPS at a critical point is cut off by its own bond dimension rather than by
the physics, so \(\xi\) grows with `D` without bound, and so does the budget.
On the Heisenberg spin-1/2 chain, \(\lambda_2\) measured on the converged
state climbs from 0.82 at `D = 8` (\(\xi \approx 5\) cells) to 0.91 at
`D = 16` (\(\xi \approx 11\)) to 0.96 at `D = 32` (\(\xi \approx 28\)). The
last of those needs around 800 cells for `1e-12`, four times the default.

### Two gapped cases that still exhaust the budget

**A degenerate ground state** closes the transfer-operator gap much as a
critical point does. **A state carrying far more bond dimension than its
entanglement needs** does the same for a numerical reason: the surplus Schmidt
weights are noise, and they leave the transfer operator near-degenerate.
Majumdar-Ghosh is both at once — spontaneously dimerized, so two-fold
degenerate, and exactly representable at `D = 2`:

| D | \(\lambda_2\) | cells for 1e-12 |
|---|---|---|
| 16 | 0.79 | ~121 |
| 32 | 0.86 | ~187 |
| 64 | 0.98 | ~1200 |
| 128 | 0.97 | ~1040 |

The rate gets worse as the state is widened past what it needs, which is the
opposite of the Haldane behavior above. A warning on a model believed to be
gapped is usually reporting one of these two situations, or saying that the
model is closer to critical than assumed, and all three are worth knowing
about rather than silencing.

## The Energy Is Not the Environment

The energy converges long before the environment tensor it is extracted from.
On the Heisenberg chain at `D = 32`, taking the environment at 100 cells —
where the residual is still `1.9e-4`, five orders of magnitude above the
default tolerance — gives an energy that agrees with the fully converged one
to `3e-14`:

| cells | residual | energy error | solve time |
|---|---|---|---|
| 100 | 1.9e-4 | 2.7e-14 | 0.20 s |
| 200 | 4.7e-6 | 1.4e-14 | 0.41 s |
| 400 | 2.9e-9 | 2.4e-15 | 0.83 s |
| 800 | 1.3e-15 | reference | 1.66 s |

The residual tracks the subleading tail of a tensor; the energy is a scalar
contraction that converges at the leading rate. A run reporting an energy to
the digit you care about, while warning on every iteration, is not
contradicting itself.

!!! note "Where a loose environment does cost something"

    A one-site run notices immediately, because its gradient compares
    quantities built from the environments and cannot be better than they are.
    Continuing a converged two-site state for 25 iterations under `'1s'`
    reached a gradient of `4.2e-5` at `env_tol = 1e-6`, `2.5e-5` at `1e-8`,
    `1.7e-5` at `1e-10`, and `1.2e-5` at `1e-12`. A run that reports its
    gradient under `'1s'` wants a tight tolerance, and `env_maxiter` raised
    with it.

### What it costs to raise the budget

Absorbing one site into an environment and applying the effective Hamiltonian
to a center tensor are the same contraction: a \(\chi \times \chi\) block
carried through one site tensor and one iMPO tensor, `step_left_env` and
`matvec_center` differing only in which index is left open. Both cost
\(O(\chi^3 d w)\) densely, with a subleading \(O(\chi^2 d^2 w^2)\), for bond
dimension \(\chi\), physical dimension \(d\), and iMPO width \(w\). Measured
on the spin-1 chain under U(1), per call:

| D | environment step | \(H_{AC}\) matvec | 50-cell solve |
|---|---|---|---|
| 64 | 1.3 ms | 1.4 ms | 0.15 s |
| 128 | 1.7 ms | 1.7 ms | 0.19 s |
| 256 | 2.9 ms | 3.2 ms | 0.32 s |
| 512 | 6.8 ms | 7.7 ms | 0.77 s |
| 1024 | 33 ms | 38 ms | 3.5 s |
| 2048 | 109 ms | 117 ms | 11.5 s |

The two columns stay within a tenth of each other across a factor of 32 in
width. Both grow by roughly four per doubling rather than the eight a dense
contraction would give, because the symmetry splits the tensors into many
small blocks and the block count rises with the bond dimension.

The consequence is the useful part: the environment's share of an iteration
is set by counts, not by the bond dimension. An iteration spends
\(2 k N\) environment steps, for \(k\) cells absorbed on each side of a cell
of \(N\) sites, against roughly one Davidson matvec per iteration of each
local solve. Exhaust a 200-cell budget and the environments dominate the
iteration; converge in twenty and they are a minor term. That balance is the
same at \(\chi = 64\) and at \(\chi = 2048\), so the budget is worth deciding
on the correlation length of the model rather than on the size of the
calculation. The loop exits on `env_tol`, so a budget raised for safety costs
nothing on the runs that do not need it.

## Choosing the Two Settings

Ordered by what the run is for:

- **A gapped model, any bond dimension.** The defaults are sized for this and
  need no attention. If the warning appears anyway, read it as a statement
  about the state: degenerate, over-wide, or less gapped than expected.
- **A critical model, energy wanted.** Loosen `env_tol` to what the solver
  actually reaches within the budget, around `1e-8` on the chains in the
  [examples](../../examples/vumps/heisenberg.md). The energy is unchanged and
  the output stays readable.
- **A critical model, gradient wanted, or a one-site run.** Keep `env_tol`
  tight and raise `env_maxiter` instead, to roughly \(28\xi\) with \(\xi\)
  estimated from a short scan of the residual against the budget.

## See Also

- [Options](options.md) — `env_tol`, `env_maxiter`, and the caching settings.
- [Overview](index.md) — why the environments are fixed points at all.
- [Heisenberg example](../../examples/vumps/heisenberg.md) — a critical chain
  run at a loosened tolerance.

<style>
/* The two residual tables on this page: data columns share the width
   equally instead of each sizing to its own contents. The selectors carry
   `.md-typeset` and `:not([class])` to outweigh the theme's own table rules */
.md-typeset .data-table table:not([class]) {
    display: table;
    width: 100%;
    table-layout: fixed;
}

.md-typeset .data-table table:not([class]) th,
.md-typeset .data-table table:not([class]) td {
    min-width: 0;
    padding: 0.9375em 0.6em;
}

.md-typeset .data-table.label-col table:not([class]) th:first-child,
.md-typeset .data-table.label-col table:not([class]) td:first-child {
    width: 30%;
}
</style>
