# Changelog

## [0.3.2] - 2026-10-24

**Truncation Scale and Kept-Space Projection**

A correctness release on two pieces v0.3.0 and v0.3.1 built. The controlled bond expansion
in DMRG and XTRG projected the center end of a bond with the center tensor itself, for
which `M M†` is not a projector, so part of the space the bond already carried leaked back
into the complement it was meant to be orthogonal to. XTRG's `trunc_thresh` and `z_tol`
were absolute numbers compared against a quantity that shrinks like `d^{-L/2}`, which made
the cutoff exponentially more aggressive with every site added and the fit-convergence test
satisfied before the fit had done anything. Both are now measured against the scale they
live on. `init_mps` also stops handing back a random MPS with every bond empty, and the
four TDVP example scripts accept `'1sp'`. No breaking API changes, but `trunc_thresh` is
now read relatively and its default moves from `1e-15` to `1e-12`.

### The Kept-Space Projection

- New `_center_isometry_forward` / `_center_isometry_backward` in
  `alice.algorithm.dmrg.complement` and `alice.algorithm.xtrg.complement` return the
  isometric factor of a 1-site split of the center tensor, which spans exactly what the
  center tensor spans. `P = M M†` is a projector only for an isometry: the neighbor end of
  the bond is one, the center end carries the singular values and is not.
- The isometric factor of Θ will not serve in its place — `decomp` keeps every sector at
  full dimension regardless of rank, so it can span the whole (bond, physical) space and
  annihilate the complement. TDVP already had this right; its docstring's aside about DMRG
  getting away with the raw site tensor is gone.
- **DMRG**: converged energies unchanged, still matching the 2-site scheme to twelve
  digits, reached one sweep earlier. **XTRG**: the expansion grows bonds more
  conservatively, so the spinless `L = 10` run ends at `1.2e-06` relative error in log Z
  rather than `1.1e-07` — both far below the `1.6e-03` of the plain 1-site scheme, and a
  smaller `trunc_thresh` recovers it.
- The projector helpers take their second argument as `A` / `B` after the role it plays,
  with docstrings stating the isometry each one satisfies.

### Relative Truncation and Fit Convergence in XTRG

- `trunc_thresh` is now relative: a singular value is discarded below
  `trunc_thresh × ‖C‖`, the Frobenius norm of the tensor being decomposed. New
  `_Truncation` in `sweep.py` resolves it into the absolute cutoff Nicole's `thresh` mode
  expects by measuring the norm of the first tensor a half-sweep decomposes. One
  measurement suffices — `‖C‖` grows by well under a factor of two over a sweep, and the
  first tensor carries the smallest norm, so the cutoff errs on the side of keeping.
- Resolution is deferred to first use because `mpo_c` opens a fit as the unit-norm copy of
  `mpo_a`. A zero or non-finite norm drops the threshold and leaves `max_bond` in charge;
  `0.0` is passed through. The backward 2-site half-sweep shares one resolved set of
  options between the trial SVD and the real split. The `compact()` calls in `_fit_mpo`
  take the relative number as it stands, normalizing before they truncate.
- **The default moves from `1e-15` to `1e-12`.** Read relatively, `1e-15` kept 599 states
  where `1e-12` kept 377 on an uncapped free-fermion chain, both on the same `1e-10`
  accuracy floor in the per-site log Z.
- `z_tol` now compares `|‖C‖ − ‖C_prev‖| < z_tol × ‖C_prev‖`. Absolute, it was satisfied
  trivially on a long chain and stopped the inner fit after its first sweep. A vanishing
  `prev_norm` never stops early. `run()` labels both thresholds `(relative)`.

### Random MPS Initialization

- `init_mps` with `bond_dim > 1` draws bond sectors from a depth-2 search around the
  center-bond charge alone, so a charge path that strays further leaves the boundary bonds
  with no allowed sector and the canonicalization empties the chain. That object was
  returned as a usable MPS and failed far from its cause, as a zero-norm Davidson guess;
  it now raises `ValueError` naming `bond_dim`, `target_qn`, and `L`, and pointing at
  `bond_dim=1` with an explicit `config=` or `scheme='1sp'`.
- The random MPS is normalized before return, matching the product-state path, instead of
  carrying whatever norm its blocks happened to have.

### Examples and Documentation

- `tdvp_freefermion.py`, `tdvp_domainwall.py`, `tdvp_hubbard.py`, and
  `tdvp_hubbard_quench.py` accept `scheme='1sp'` with `expand_k` and `expand_alpha`, scan
  modes included; the domain-wall bond scan refuses `'1s'`, whose bonds stay at dimension
  1 whatever `max_bond` says. The XTRG examples follow the new `trunc_thresh` default.
- `docs/algorithms/xtrg/lessons.md` is rewritten around the relative cutoff: what the
  threshold is compared against, where the conversion happens, why the scale needs
  converting, the same reasoning for `z_tol`, and an admonition that relative is not an
  error bound.
- The four TDVP example pages document `'1sp'`: a three-scheme table on the free-fermion
  page; a tip beside the frozen-`'1s'` warning on the domain-wall page, where `'1sp'`
  matches the 2-site occupation error on a smaller bond profile (and the quoted
  frozen-`'1s'` error is corrected from `0.47` to the measured `0.421`); a section on
  `expand_k` counting multiplets under SU(2) on the Hubbard page; and the eigenstate limit
  on the quench page, confirmed to `6.7e-14` at zero discarded weight.

### Tests

- New `TestCenterIsometry` in the DMRG and XTRG `test_scheme_1sp.py`: isometry of the
  factor, the space it spans, axis order and itags, the center tensor demonstrably not an
  isometry, each projection idempotent (which it was not before), and the discarded halves
  orthogonal to the kept space, with guards against assertions passing on a numerically
  zero half.
- New `TestTruncation` and `TestTruncationScale` in `tests/algorithm/xtrg/test_sweep.py`
  cover `_Truncation` directly and check that scaling a density matrix by four orders of
  magnitude leaves the bonds and the reported discarded weight unchanged in all three
  schemes.
- New `TestChainLengthPortability` in `test_xtrg.py` cools a spinless chain at four and at
  ten sites under identical options against the exact single-particle log Z, holding the
  error flat where it used to lose two orders of magnitude.
- New `TestInitMpsEmptyBonds` and `test_norm_is_one` in `tests/network/test_automps.py`.

### Statistics

- **1,357 tests** across 38 test modules (up from 1,313 in v0.3.1; no new modules).
- **30 commits** since v0.3.1.
- **24 files changed**, 1,364 insertions, 184 deletions.
- **35 source modules** in five subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`, `alice.algorithm.tdvp` (unchanged from
  v0.3.1).

### Compatibility

- **Breaking Changes:** none.
- **Behavioral Changes:** `xtrg.Options.trunc_thresh` is read relatively and defaults to
  `1e-12`, so a value tuned against the old absolute meaning will truncate far more than
  it used to. `z_tol` is read relative to `‖C_prev‖`, so long-chain runs will use more of
  their `n_sweeps` budget. DMRG and XTRG `'1sp'` expand bonds more conservatively at a
  fixed `expand_k`. `init_mps` with `bond_dim > 1` raises `ValueError` for a `target_qn`
  whose charge path strays from the center-bond sector, and the random mode returns a
  normalized MPS.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.4.2.

---

## [0.3.1] - 2026-10-21

**TDVP: The 1-Site-Plus Scheme**

Implements `'1sp'`, the third TDVP update scheme, which v0.3.0 accepted in `Options` and
rejected in `run()`. Before each local update it widens the bond ahead of it with up to
`expand_k` directions drawn from the complement of the space that bond currently spans,
evolves over the widened space, and truncates back — so the rank adapts as it does under
`'2s'` while every local exponential stays 1-site sized. It is an adaptive-rank scheme of
the `'2s'` kind at close to 1-site cost, and it is free of the frozen-product-state trap.
No breaking API changes.

### The 1-Site-Plus Scheme

- `run()` accepts `scheme='1sp'` and `_IMPLEMENTED_SCHEMES` now holds all three schemes,
  so the `NotImplementedError` v0.3.0 raised is gone.
- Where `'1s'` and `'2s'` split one fixed tangent-space projector, `'1sp'` rebuilds the
  space it projects onto at each bond out of the state it finds there, as the sweep
  reaches it.
- Its irreversibility is structural: the expansion is chosen from the state the step
  begins in, so forward and backward steps do not widen the bond into the same subspace.
  The deviation enters at the order of the integrator error and falls as `dt²`.
- `expand_k = 0` empties the complement and degenerates a `'1sp'` sweep into a `'1s'`
  sweep exactly, which separates an error in the expansion from one in the integrator.
- A bond of dimension 1 has a complement like any other, so a quench from a product state
  evolves. The frozen-bond warning now recommends `'2s'` or `'1sp'`.

### Complement Isometry

- New `alice.algorithm.tdvp.complement` with `expand_forward`/`expand_backward`. Following
  DMRG's `'1sp'` scheme, `H` is applied independently from the left and the right to a
  cheaply factored 2-site tensor, the kept subspace is projected out of each half, and the
  SVD of the doubly-discarded outer product supplies the new basis vectors.
- **Exact state preservation**: only the neighbor is expanded by `oplus`, and the center
  tensor is filled in by projecting Θ onto the expanded neighbor basis, which is exact
  because the expanded basis contains the old one. Expanding both tensors would move the
  state.
- **Exact expanded environment**: built by a single `step_*_env` call over the expanded
  neighbor, keeping the cross blocks a block-diagonal assembly would drop. DMRG can afford
  to drop them for an initial guess; here a perturbed `H_eff` perturbs the trajectory.
- **Orthonormal projectors**: the neighbor side projects against the site tensor, which is
  isometric there; the center side uses a 1-site split of the center tensor, since
  `M M†` is a projector only for an isometry. The isometric factor of Θ must not be used,
  as `decomp` keeps every sector at full dimension and that factor can span the whole
  (bond, physical) space and project the complement away entirely.
- `expand_k` is a cap, not a request: a bond gains however many directions its complement
  actually holds, typically one or two per update for a short-range Hamiltonian.
- Directions below `1e-10` times `‖lh‖·‖rh‖` are cancellation artifacts, lie in the kept
  space, and are dropped — a relative threshold cannot do this job, shrinking with exactly
  the quantity whose smallness is the thing to detect. The same absolute floor gates
  Θ'_disc as a whole, which is where the test must be made: two discarded halves can each
  keep a sizable remainder while their contraction cancels it, and the SVD returns a
  unit-norm vector from a zero block regardless.
- A bond with no complement — saturated, or out of `H`'s reach — is left alone, with the
  environment the 1-site step would have used handed back.

### Sweeps and Options

- New `_forward_1sp`/`_backward_1sp` half-sweeps in `sweep.py`, reusing every piece of
  local arithmetic from `scheme_1s`. Each site expands the bond ahead of it, evolves,
  splits with truncation under `max_bond`/`trunc_thresh`, then back-evolves the bond
  tensor and absorbs it into the expanded neighbor; the far-end site takes no expansion
  and no back-evolution.
- The zero-site step spans two bonds rather than one: `C` is rectangular, truncated bond
  on one axis and expanded bond on the other, which is what keeps it exact.
- `split_forward`/`split_backward` gain an optional `trunc` forwarded to `decomp`,
  defaulting to `None`; `'1s'` always passes `None` and nothing else in `scheme_1s`
  truncates.
- `tdvp.Options` gains `expand_k` (default `4`) and `expand_alpha` (default `None`), used
  only by `'1sp'`; `max_bond` and `trunc_thresh` now apply to `'1sp'` as well, and
  `Summary.discarded_weights` is populated by it. `run()` logs both settings at startup.
- The expansion-option guidance across all three algorithms is now `round(sqrt(max_bond))`
  rather than a bare `4`.

### Tests

- New `tests/algorithm/tdvp/test_complement.py` covering the cheap factors, the
  center-side orthonormalization, the half contractions reassembling `H·Θ̃`, the
  discarded-space projections, complement detection on a bond that has one, a saturated
  bond, and a vanishing doubly-discarded tensor, and the expansion end to end — bond
  growth, state preservation, unchanged effective energy and norm, neighbor isometry, the
  fill-in matching zero padding, chain edges, and `alpha` truncation. `TestNonAbelian`
  repeats the core cases on an SU(2) Heisenberg chain.
- New `TestTruncatingSplits` in `test_scheme_1s.py`, `TestExpansionReversibility` and
  bond-growth coverage in `test_sweep.py`, `TestFreeFermion1SitePlus` and
  `TestMeltingDomainWall` in `test_physics.py` with a `domain_wall` fixture and its exact
  reference, and `TestExpansionDriver` in `test_tdvp.py` covering achieved growth, the
  logged settings, and a resumed run reproducing an uninterrupted one.

### Documentation

- `docs/algorithms/tdvp/index.md` gains a **Controlled Bond Expansion** section with the
  option table, why `expand_k` is a cap, the escape from the frozen-product-state trap,
  and an admonition on the scheme not being exactly time-reversible; the scheme table
  lists `'1sp'` in place of the forward-compatibility note.
- `docs/algorithms/tdvp/options.md` gains a **Choosing `expand_k`** section.
- New `docs/examples/tdvp/scheme-comparison.md` and
  `examples/examples_tdvp/tdvp_scheme_comparison.py`: the three schemes on one
  free-fermion quench from a half-filled product state, scored against the exact
  correlation matrix, with `expand_k` and `dt` scans. `'1s'` stays frozen at bond
  dimension 1 with every conservation law intact; `'1sp'` and `'2s'` both land within
  9e-05 of the Gaussian solution, `'1sp'` on a smaller bond profile.

### Statistics

- **1,313 tests** across 38 test modules (up from 1,211 / 37 modules in v0.3.0).
- **30 commits** since v0.3.0.
- **17 files changed**, 3,109 insertions, 84 deletions.
- **35 source modules** in five subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`, `alice.algorithm.tdvp`.

### Compatibility

- **Breaking Changes:** none.
- **Behavioral Changes:** `tdvp.run` no longer raises `NotImplementedError` for
  `scheme='1sp'`. `tdvp.Options` gains two fields, `expand_k` and `expand_alpha`, so
  positional construction past `trunc_thresh` shifts. The frozen-bond warning text now
  recommends `'2s'` or `'1sp'`.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.4.2.

---

## [0.3.0] - 2026-10-18

**TDVP: Real-Time Evolution**

Introduces TDVP (Time-Dependent Variational Principle), Alice's third algorithm alongside
DMRG and XTRG: a real-time solver that evolves an MPS under `exp(-iHt)` by projecting the
Schrödinger equation onto the tangent space of the MPS manifold and integrating by
projector splitting. Two update schemes are available — 1-site at fixed bond dimension and
2-site with SVD truncation — each driven by a Lanczos–Krylov matrix exponential and a
first- or second-order symmetric integrator. The network layer gains complex amplitudes:
`Network.to()`, a `dtype` keyword on `init_mps`, and an `observe` that contracts a complex
state against a real observable. No breaking API changes.

### TDVP Algorithm

- New `alice.algorithm.tdvp` package exporting `Options`, `Summary`, `Artifact`, and
  `run`, added to the `alice.algorithm` and top-level `alice` namespaces along with
  `alice.xtrg`.
- `tdvp.run(artifact, H, opts)` takes the starting state bundled with its `time` and
  `step` in an `Artifact` and the Hamiltonian MPO separately, returning
  `(Summary, Artifact)`. One full sweep is one time step; there is no inner convergence
  loop and no `e_tol` analogue.
- The tangent-space projector is never formed. Its positive terms are the single-site
  evolutions of the sweep; its negative term is the zero-site **back-evolution** on each
  bond at `-dt`, which cancels the double counting between consecutive terms and without
  which the integrator loses even first-order consistency.
- **1-site** (`'1s'`): exact, non-truncating splits; conserves norm and energy to roundoff
  and holds every bond dimension fixed. **2-site** (`'2s'`, default): evolves the merged
  two-site tensor and splits it by SVD under `max_bond`/`trunc_thresh`, so bonds adapt as
  entanglement develops. `'1sp'` is accepted by `Options` and rejected by `run()` with
  `NotImplementedError`.
- The 1-site manifold has *fixed* bond dimension, so a quench from a product state stays a
  product state forever while norm and energy stay conserved. `run()` warns at startup,
  naming the bonds of dimension 1.
- `order=2` (default, Lubich symmetric) and `order=1` (Lie-Trotter, with alternating
  direction) are both globally second order at equal cost — two `order=1` steps of size
  `dt` are exactly one `order=2` step of size `2dt`. The difference is that every
  `order=2` snapshot lands at the end of a complete forward-and-backward sweep pair.

### Krylov Exponential and Environments

- New `krylov.py` computes `exp(-i dt H) v` by Lanczos with full re-orthogonalization,
  exponentiating the tridiagonal exactly via `numpy.linalg.eigh` (no SciPy dependency).
  It stops on happy breakdown, on `krylov_tol`, or at the `krylov_dim` cap, and reports
  the error estimate back as `Summary.krylov_errors`.
- New `environ.py` stores 3rd-order blocks shared by the 1-site, 2-site, and zero-site
  effective Hamiltonians, so the zero-site update adds no storage. Disk spilling reuses
  the DMRG/XTRG sliding-window-plus-async-I/O design.

### Options, Summary, Artifact

- `Options` exposes `scheme`, `dt` (may be negative), `n_steps`, `order`, `max_bond`,
  `trunc_thresh`, `krylov_dim`, `krylov_tol`, `renormalize`, the environment cache triple,
  `checkpoint_dir`, `artifacts_dir`, `save_artifacts`, and `save_artifacts_every`.
  `renormalize` defaults to `False` so truncation norm loss stays visible.
- `Summary` carries `times`, `energies`, `norms`, `max_bonds` (`n_steps + 1` entries),
  `discarded_weights` and `krylov_errors` (`n_steps` entries), and the final `bond_dims`
  profile — no state.
- `Artifact` (`state`, `time`, `step`) is both input and output of `run()`, which is what
  makes a resumed run indistinguishable from a fresh one.
- `evolution.ckpt` and `tdvp.ckpt` are written atomically every step; `tdvp.ckpt` is
  removed on success. A finished run continues from any archived
  `artifacts/step_XXXXX.ckpt` under new options, and starting before the end of the
  history truncates and recomputes it (with a `WARNING`). `n_steps` is the absolute step
  index to stop at, counted from `t = 0`.
- `save_artifacts_every` thins the archive on the absolute step index, so a stitched
  trajectory lands on the same grid as an uninterrupted one. `run()` clamps `n_steps` down
  to the last grid point and warns, since `tdvp.ckpt` is removed on success and an
  off-grid endpoint would leave no snapshot of the final state.

### Complex Amplitudes

- New `Network.to(device, dtype)` casts and moves every site tensor in place, including
  the Bridge intertwiner weights. Narrowing a complex network to a real dtype raises
  `ValueError`.
- `init_mps` gains a `dtype` keyword (default `torch.float64`), threaded through both the
  product-state and random-MPS paths.
- `tdvp.run` promotes a real state and a real MPO to `complex128` in place, logged at
  INFO, so a DMRG ground state can be handed straight to a quench.
- `observe` supports complex states, including against a real observable. Scalar
  extraction moves into a shared `_real_scalar` helper that discards a roundoff-level
  imaginary part and raises beyond it, and reads a boundary tensor with no data block as
  an exactly vanishing expectation value.

### Tests

- Seven new test modules under `tests/algorithm/tdvp/` plus a shared `conftest.py`.
  `test_physics.py` pins the algorithm against physics: the analytic single-particle
  propagator with `dt²` convergence, norm/energy/charge conservation across both schemes
  and orders, forward-backward reversibility, eigenstate phase winding, exactness at full
  rank, and the frozen product state under `'1s'`.
- New complex-dtype coverage in `test_network.py`, `test_observe.py`, and
  `test_automps.py`, including SU(2), global-phase invariance, a non-Hermitian observable,
  and a complex thermal density matrix.

### Documentation

- New `docs/algorithms/tdvp/` section (overview, `Options`, `Summary`, `Artifact`, `run`)
  and four `docs/examples/tdvp/` walkthroughs, each validated against an analytic
  reference or across symmetry groups.
- Complex-state sections added to `observe.md` and `init-mps.md`, `to` rendered on
  `network.md`, and TDVP added to the algorithms/examples indexes, `mkdocs.yml`, and the
  README — where it also leaves the `Upcoming` list.
- New `docs/algorithms/xtrg/lessons.md`: `trunc_thresh` is an absolute singular value
  cutoff, and the squaring sweeps apply it to a product of unit-norm factors whose norm is
  `2^(-N/2)` near infinite temperature, so a longer chain needs a proportionally smaller
  threshold.
- The algorithms index gains the XTRG `Artifact` row it was missing, and names every
  algorithm's entry-point page `Launch` to match the sidebar.

### Statistics

- **1,211 tests** across 37 test modules (up from 979 / 30 modules in v0.2.8).
- **80 commits** since v0.2.8.
- **53 files changed**, 10,480 insertions, 43 deletions.
- **34 source modules** in five subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`, `alice.algorithm.tdvp`.

### Compatibility

- **Breaking Changes:** none.
- **Behavioral Changes:** `observe` raises `ValueError` when the expectation value has a
  non-negligible imaginary part, and returns `0.0` for a boundary tensor with no data
  block where it previously raised `StopIteration`. `tdvp.run` mutates the `MPS` and `MPO`
  it is given, promoting both to `complex128` in place.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.4.2.

---

## [0.2.8] - 2026-09-22

**N-Site Terms as Verbatim Windows**

Adds `InteractionNSite`, a third `Interaction` subclass for Hamiltonian terms with more
than two operators. Instead of naming its sites and letting `build_hamiltonian`
synthesize the operator string in between, it supplies the entire contiguous window
verbatim — one tensor per site from `min(sites)` to `max(sites)`, including the identity
and Jordan-Wigner string tensors on sites carrying no operator. `build_hamiltonian`
accepts the new subclass and now rejects unrecognized `Interaction` subclasses instead
of silently adding a bare identity term. No breaking API changes.

### `InteractionNSite`

- New dataclass in `alice.network.interaction`, exported from `alice.network` and the
  top-level `alice` namespace, adding `sites` (operator site indices in operator order,
  e.g. `[m, n, k, l]` for `c†_m c†_n c_k c_l`) and `tnsrs` (the contiguous window of
  4-index MPO tensors).
- `sites` is metadata for the model builder: placement depends only on the span covered,
  so `build_hamiltonian` reads only `min(sites)` and `max(sites)` and accepts any
  permutation. `tnsrs[offset]` goes at site `min(sites) + offset`, and the two outer
  bonds of the window must be trivial dim-1 charge-neutral.
- Non-contiguous terms are expressed by padding the gaps with identity or string
  tensors, not by splitting the term.

### `build_hamiltonian`

- Handles `InteractionNSite` in the validation pass and the accumulation loop, applying
  the coupling to the **last** window tensor only — matching the on-site and terminal
  tensor rules for the other two subclasses. The docstring now states that `cpl` must
  *not* be baked into the tensors, correcting wording that said otherwise.
- New `ValueError` conditions naming the offending index and sites: empty `sites`,
  `tnsrs=None`, and a `tnsrs` whose length is not `max(sites) - min(sites) + 1`.
- An active interaction of an unsupported subclass now raises `TypeError`; previously it
  fell through the accumulation loop and contributed a bare identity term.

### Tests

- New `tests/network/test_autompo.py`, the first dedicated AutoMPO test module, built on
  `build_fermionic('U1')` and a dense Jordan-Wigner reference.
- `TestInteractionNSiteValidation` covers the three new error paths;
  `TestFourOperatorMPO` checks a four-fermion term plus its Hermitian conjugate on an
  `L = 4` chain against the exact two-particle sector eigenvalue, an identity-padded gap
  between disjoint one-body factors, the `p = 0` density-density term over every
  two-particle configuration, and a hopping channel matching its `G4`/`G4dag`
  `Interaction2Site` construction.

### Documentation

- New `docs/api/interaction/interaction-nsite.md` describing the verbatim-window
  contract, linked from the API and interaction indexes, the other interaction pages,
  and the `mkdocs.yml` navigation.
- `core-concepts.md` now describes four interaction dataclasses instead of three.

### Statistics

- **979 tests** across 30 test modules (up from 971 / 29 modules in v0.2.7).
- **13 commits** since v0.2.7.
- **14 files changed**, 477 insertions, 11 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** none.
- **Behavioral Changes:** `build_hamiltonian` raises `TypeError` on an active
  interaction that is not an `Interaction1Site`, `Interaction2Site`, or
  `InteractionNSite`; such objects previously produced an MPO with a spurious identity
  term.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.7] - 2026-09-19

**Small Norms Are Not Zero**

Corrects the zero-norm guards in `Network.normalize()`, `MPO.redistribute_norm()`, and
`NormalMPO.from_mpo()`. These used an absolute tolerance of `1e-15`, spuriously
rejecting small but legitimate norms — most importantly the `d^(-L/2)` norm of a squared
unit-norm identity-like MPO, the XTRG seed scenario on long chains. All three now reject
only an exact zero or a non-finite norm. Package metadata is updated with the new author
and maintainer contacts. No breaking API changes.

### Zero-Norm Guards

- `Network.normalize()`, `MPO.redistribute_norm()`, and `NormalMPO.from_mpo()` raise
  only when the relevant norm is exactly `0.0` or not finite (`inf`/`nan`). Previously
  `math.isclose(n, 0.0, abs_tol=1e-15)` rejected any norm below `1e-15`, even though
  dividing by such a value is well-conditioned in float64 and `math.log(n)` stays finite
  for any non-zero float64.
- Error messages now report the offending value (e.g. `"cannot normalize: network norm
  is 0.0"`) instead of the generic `"numerically zero"` / `"zero-norm"` wording.

### Tests

- New `test_small_norm_ok_mps` and `test_redistribute_norm_small_norm_ok` in
  `test_network.py`: scale a canonical network to norm `1e-20` and check that
  normalization and redistribution succeed.
- New `test_identity_square_small_norm_compacts` in `test_thermal.py`: square a
  unit-norm identity `NormalMPO` on an `L = 100` spin-1/2 chain (norm `2^(-50)`), then
  `compact()` it and verify `log_scale`, unit internal norm, and `log_trace()` against
  their closed forms.
- Existing zero-norm tests updated to the new error messages.

### Metadata

- `pyproject.toml`: author email moves to `c.zhang@ideogenesis.ai`; new `maintainers`
  entry for Ideogenesis AI (`developer@ideogenesis.ai`). Documentation hero image
  refreshed.

### Statistics

- **971 tests** across 29 test modules (up from 968 / 29 modules in v0.2.6).
- **7 commits** since v0.2.6.
- **7 files changed**, 79 insertions, 18 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** none.
- **Behavioral Changes:** `normalize()`, `redistribute_norm()`, and
  `NormalMPO.from_mpo()` no longer raise on norms in `(0, 1e-15)`. The `ValueError`
  messages for a genuinely zero norm changed wording; callers matching on
  `"numerically zero"` or `"zero-norm"` should match on `"norm is 0.0"` / `"norm 0.0"`.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.6] - 2026-08-20

**Continuable Cooling, Faster Thermal Measurement**

Generalizes XTRG resumption: `run()` now accepts a starting state from *any* archived
`artifacts/step_XX.ckpt`, not just the step the previous run stopped at, so a finished
run can be cooled further or a segment re-cooled under different options. Separately,
`observe()` on a thermal `NormalMPO` is rewritten as a transfer-matrix environment
sweep instead of forming and compressing the MPO product `ρ · O`. No breaking API
changes.

### Continuing a Finished XTRG Run

- `run()` accepts a state at any step covered by `thermal.ckpt`, including one *before*
  the end of that history; the later entries are truncated (with a `WARNING`), then
  recomputed and overwritten on disk together with their `step_XX.ckpt` archives.
  Previously the history had to end exactly at `state.step`.
- History recovery moves into a new `_resume_history` helper, with validation anchored
  at `state.step` — β is compared at `history.betas[state.step]` rather than at the end
  of the history, which is rejected only if it *stops before* `state.step`.
- New checks: `history.betas[0]` must match `opts.tau_0`, and `state.step` must not be
  past `opts.n_steps` (which is the absolute step index to stop at, counted from τ₀,
  not a number of additional steps).
- A resumed run's startup banner reports the starting step and steps remaining, and
  derives `beta_max` from `state.beta` instead of τ₀.

### Thermal `observe()` via Environment Sweep

- `observe(rho, O)` for a `NormalMPO` now evaluates `Tr[ρ O] / Tr[ρ]` with a
  left-to-right sweep accumulating a `(ρ_bond, O_bond)` environment, mirroring the MPS
  path, instead of forming `ρ · O` at bond dimension `χ_ρ · χ_O` and calling `compact()`
  before the trace.
- The ratio is still combined in log-space, so it stays correct when either trace alone
  would overflow float64; the boundary scalar accounts for the Bridge (`intw`) weight,
  covering generic symmetry groups.
- A length mismatch between `rho` and `observable` now raises `ValueError`.

### Tests

- New `TestResume` cases in `test_xtrg.py` for continuing a finished run from
  `step_02.ckpt`, re-cooling from an earlier step (asserting the truncation warning),
  and the three new error conditions.
- New thermal `observe()` coverage in `test_observe.py`, shared via
  `_ObserveThermalTests` and run against both U(1) and SU(2) realizations of the same
  Heisenberg chain, including an SU(2)-vs-U(1) cross-check of `⟨H⟩_β`; new session-scoped
  `heisenberg_mpo_u1`/`heisenberg_mpo_su2` fixtures in `tests/network/conftest.py`.

### Documentation

- `xtrg/index.md` gains a "Continuing a finished run" section; the module and `run()`
  docstrings document the same, including the new `ValueError` conditions.
- `xtrg/summary.md` warns that a continued run's summary may merge segments computed
  under different options, with `u`/`c_V` finite differences at the junction mixing both
  accuracies.

### Statistics

- **968 tests** across 29 test modules (up from 954 / 29 modules in v0.2.5).
- **8 commits** since v0.2.5.
- **8 files changed**, 498 insertions, 46 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** none.
- **Behavioral Changes:** a `thermal.ckpt` reaching past `state.step` is now accepted
  and truncated rather than rejected, while a mismatched τ₀ and a `state.step` past
  `opts.n_steps` are now rejected; a continued run overwrites the `step_XX.ckpt` and
  `thermal.ckpt` entries past its starting step. Thermal `observe()` results may differ
  in the last few digits, since the environment sweep skips the `compact()` compression
  the MPO-product path applied.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.5] - 2026-08-19

**Independent Artifact Storage for DMRG and XTRG**

Decouples archived-artifact storage from the checkpoint directory in both
`dmrg.run()` and `xtrg.run()`, via a new `Options.artifacts_dir`. DMRG gains
final-state archiving to match XTRG's existing per-step archiving; XTRG's mid-run
`Artifact` file is renamed from `progress.ckpt` to `xtrg.ckpt`. Two breaking changes,
both to on-disk file names.

### `artifacts_dir`: Decoupled from `checkpoint_dir`

- New `Options.artifacts_dir` on both `dmrg.Options` and `xtrg.Options` (default
  `None`, resolving to `artifacts/` under `checkpoint_dir`); an explicit path archives
  artifacts elsewhere, independent of where checkpoints themselves live.
- XTRG's `save_artifacts`/`save_artifacts_since` now write `step_XX.ckpt` under
  `artifacts_dir` instead of always under `checkpoint_dir/artifacts`.
- Both `run()`s log the resolved artifacts directory in their startup banner.

### DMRG Final-State Archiving

- **Breaking:** `dmrg.run()` now archives its final `Summary` as `state.ckpt` under
  `artifacts_dir` on success, then removes the per-sweep `dmrg.ckpt`/`dmrg_lock.ckpt`,
  since they are redundant with the archived artifact. Previously `dmrg.ckpt` was left
  on disk after a successful run.
- Unlike the per-sweep checkpoint, the archived `Summary` carries the correct
  `converged` flag, since it is built after the sweep loop exits.

### XTRG `xtrg.ckpt` Rename

- **Breaking:** the mid-run `Artifact` checkpoint is renamed from `progress.ckpt` to
  `xtrg.ckpt` (lock file `progress_lock.ckpt` → `xtrg_lock.ckpt`), aligning the name
  with DMRG's `dmrg.ckpt` as each algorithm's own per-run checkpoint file. Behavior is
  unchanged: written every step, removed on success.

### Tests

- New `TestArtifact` class in `test_dmrg.py`: `state.ckpt` creation, lock-file cleanup,
  round-trip loading with a correct `converged` flag, default placement, and
  `artifacts_dir` decoupled from `checkpoint_dir`.
- New `artifacts_dir` default/TOML-round-trip tests in both `test_dmrg.py` and
  `test_xtrg.py`; `test_progress_removed_after_success` renamed to
  `test_checkpoint_removed_after_success` and updated for `xtrg.ckpt`.

### Documentation

- `artifact.md`'s on-disk layout diagram gains `artifacts_dir` and the `xtrg.ckpt`
  rename; `index.md`'s resumption example and both example scripts updated from
  `progress.ckpt` to `xtrg.ckpt`.

### Statistics

- **954 tests** across 29 test modules (up from 946 / 29 modules in v0.2.4).
- **11 commits** since v0.2.4.
- **9 files changed**, 214 insertions, 69 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** `dmrg.run()` no longer leaves `dmrg.ckpt` on disk after a
  successful run — read `artifacts_dir/state.ckpt` instead. XTRG's mid-run
  `progress.ckpt` is renamed to `xtrg.ckpt`; callers loading it by a hardcoded path
  must update the filename.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.4] - 2026-08-18

**XTRG: Resumption from Interrupted Runs**

Allows an interrupted XTRG run to resume from its last checkpointed step rather than
restarting from `ρ(τ₀)`. This required moving state construction out of `run()`: it now
accepts a starting `Artifact` (`rho`, `beta`, `step`) supplied by the caller, instead of
building `ρ(τ₀)` internally via `thermal_mpo()`. Breaking change to the signature of
`run()`.

### Resumable `run()`

- **Breaking:** `run(H, spc, opts)` becomes `run(state: Artifact, opts=None)`; the chain
  length `L` is read from `state.rho.L`, and the internal `thermal_mpo` call is removed.
- At `state.step == 0`, `run()` records the initial `(beta, log Z)` grid point directly
  from `state`, as before. When resuming (`state.step > 0`), it loads `thermal.ckpt` to
  recover the prior `betas`/`log_z`/`discarded_weights` history, validates it against
  `state.step` and `state.beta`, and continues squaring from `state.step` onward.
- Callers now build the initial state explicitly — `thermal_mpo(...)` wrapped in an
  `Artifact` — and resume a crashed run by passing the `Artifact` from `progress.ckpt`
  back into `run()`.

### Tests

- New `_initial_state` helper builds the step-zero `Artifact`; all existing XTRG tests
  updated to use it.
- New `TestResume` (5 tests): a resumed run matches an uninterrupted one's thermodynamics
  to `rel_tol=1e-10` while only computing the remaining steps; missing or inconsistent
  checkpoints raise `FileNotFoundError`/`ValueError`; resuming past `n_steps` is a no-op.

### Documentation

- `xtrg_spinless`, `xtrg_spinful`, and the XTRG documentation and worked examples
  (free-fermion, Hubbard) updated to build the initial `Artifact` via `thermal_mpo`
  before calling `run()`. The module docstring and `index.md` gain a resumption example.

### Statistics

- **946 tests** across 29 test modules (up from 940 / 29 modules in v0.2.3).
- **9 commits** since v0.2.3.
- **8 files changed**, 295 insertions, 82 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** `xtrg.run()` no longer accepts `(H, spc, opts)`; its signature is
  now `run(state: Artifact, opts=None)`. Callers must build the initial state themselves,
  e.g. `xtrg.Artifact(rho=thermal_mpo(H, opts.tau_0, opts.taylor_order, spc), beta=opts.tau_0, step=0)`.
  `progress.ckpt` and `thermal.ckpt` from earlier versions remain loadable for resumption.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.3] - 2026-08-17

**XTRG: Convergence-Based Early Termination**

Lets XTRG's variational compression terminate before exhausting its sweep budget once it
has converged, via a new `Options.z_tol` tolerance on `‖C‖` between sweeps.
`Summary.converged` is renamed to `Summary.finished` to say what it actually means.
One breaking change to `Summary`'s fields.

### Early-Termination Compression Fit

- New `Options.z_tol` (default `1e-10`): `_fit_mpo` terminates sweeping once
  `|‖C‖ − ‖C_prev‖| < z_tol`, measured at the orthogonality center after each full
  sweep; `n_sweeps` becomes a maximum rather than a fixed count.
- The criterion is exact: at the least-squares optimum, `⟨C, A·B⟩ = ‖C‖²`, so
  `‖C − A·B‖²_F = ‖A·B‖² − ‖C‖²` with `‖A·B‖` fixed across sweeps — `‖C‖` convergence is
  equivalent to residual convergence, read from the already-isometric center tensor.
- `z_tol=0.0` restores the previous fixed-sweep-count behavior exactly. `run()`'s
  startup banner now logs the configured fit convergence threshold.

### `Summary.finished` Rename

- **Breaking:** `Summary.converged` is renamed to `Summary.finished`; the field was
  always about whether the summary came from a completed `run()` call, not fit
  convergence. `serialize()`/`deserialize()` updated; serialization version stays at 2.
- A v0.2.2 `thermal.ckpt` still loads, but reloads as `finished=True` unconditionally,
  since the old `converged` key is no longer consulted.

### Tests

- New `TestFitMpoZTol`: a loose `z_tol` stops early, `z_tol=0.0` always runs the full
  sweep budget, and an early-terminated fit reproduces the `log_z` of a full-sweep run.

### Statistics

- **940 tests** across 29 test modules (up from 937 / 29 modules in v0.2.2).
- **7 commits** since v0.2.2.
- **3 files changed**, 119 insertions, 20 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** `Summary.converged` is renamed to `Summary.finished`; a
  `thermal.ckpt` from v0.2.2 or earlier still loads but reloads as `finished=True`
  unconditionally.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.2] - 2026-08-14

**XTRG: Specific Heat and Cache Isolation**

Fixes the sign of XTRG's specific heat, which was reported negative for every physical
Hamiltonian, and isolates XTRG's environment disk cache per run so that concurrent jobs
sharing one `env_cache_dir` no longer overwrite each other's blocks. The remainder is a
consistency pass over the codebase: American English spelling, "index" in place of "leg",
lowercase builtin type annotations, and role-based einsum index names. No breaking
changes.

### Specific Heat Sign Fix

- **Bug fixed:** `_compute_observables` computed `c_V[n] = β_n Δu / ln 2`, dropping the
  minus sign in `c_V = ∂u/∂T = −β ∂u/∂(ln β)`. Since `u` decreases with β, the reported
  specific heat was negative wherever the true value is positive.
- Both the interior and the trailing one-sided stencil now carry the correct sign;
  `free_energies`, `energies`, and `entropies` are unaffected.

### XTRG Environment Cache Isolation

- `xtrg.run()` now creates a unique subdirectory (first 8 hex characters of a UUID4,
  e.g. `{env_cache_dir}/a1b2c3d4/`) inside `env_cache_dir` per invocation, reused by
  every squaring step and removed in a `finally` block on return or exception.
- Cache-directory resolution moved from `_fit_mpo` to `run()`; `_fit_mpo` gains a
  `cache_dir` parameter and no longer reads `opts.env_cache_dir`.
- The resolved cache path is logged in `run()`'s startup banner.

### Consistency Pass

- All comments, docstrings, and documentation pages converted to American English;
  "leg" replaced by "index" or an ordinal rank ("2nd-order" instead of "2-leg").
- XTRG drops `typing.Dict`/`typing.List` for builtin `dict`/`list`; `sweep` gains real
  `Options` annotations under `TYPE_CHECKING`.
- `_observe_mps`'s contraction renamed from `einsum('ace,abg,cdgh,efh->bdf', …)` to
  `einsum('aob,acr,oprs,bds->cpd', …)`, following the project's index-naming convention.

### Tests

- New `TestComputeObservables` (specific-heat sign on a two-level system) and
  `TestEnvCache` (per-run subdirectory creation and removal, distinct subdirectories
  across runs, cached run matching the in-memory run, TOML round trip).
- `test_scheme_1s`'s idempotency test now builds mixed-canonical environments in the
  compressed MPO's own canonical frame, where the 1-site update is a true variational
  optimum.

### Documentation

- New **Environment caching for large chains** section in the XTRG free-fermion example;
  the DMRG Hubbard example's caching note updated for the per-run subdirectory.

### Statistics

- **937 tests** across 29 test modules (up from 930 / 29 modules in v0.2.1).
- **44 commits** since v0.2.1.
- **48 files changed**, 528 insertions, 313 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.2.1. Checkpoints from
  v0.2.0/v0.2.1 still load, but their `specific_heats` entries must be negated.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.1] - 2026-08-07

**XTRG: Density-Matrix Artifacts**

Splits XTRG's density matrix out of `Summary` into a new `Artifact` dataclass, and
reworks checkpointing around that split: `thermal.ckpt` tracks the thermodynamic history
alone, `progress.ckpt` protects the latest density matrix against a mid-run crash, and an
optional `artifacts/` archive keeps a caller-chosen range of per-step density matrices on
disk. `xtrg.run()` now returns `(Summary, Artifact)`. Breaking change to `xtrg.run()`'s
return signature and `Summary`'s fields.

### `Artifact` and Checkpointing

- New `Artifact(AlgorithmSummary)` dataclass holding `rho` (`NormalMPO`), `beta`, and
  `step`; exported from `alice.algorithm.xtrg` alongside `Options`, `Summary`, and `run`.
- `Summary` drops its `rho` and `rho_log_scale` fields entirely; `serialize`/
  `deserialize` bump to version 2 and reject version-1 (rho-carrying) payloads with a
  `ValueError`.
- `thermal.ckpt` replaces `xtrg.ckpt` (rho-free `Summary`, written every step);
  `progress.ckpt` is new (latest `Artifact`, deleted on successful completion).
- New `Options.save_artifacts` (default `True`) and `Options.save_artifacts_since`
  (default `0`) archive per-step `Artifact` files under `artifacts/step_XX.ckpt`.
- `checkpoint_dir` now defaults to the current working directory instead of disabling
  checkpointing when unset.

### Examples

- `xtrg_spinless` and `xtrg_spinful` return `(Summary, Artifact)` and accept
  `save_artifacts` / `save_artifacts_since`; CLI gains `--no-save-artifacts` and
  `--save-artifacts-since K`.

### Documentation

- New `docs/algorithms/xtrg/artifact.md`; other XTRG algorithm pages updated for the
  `Summary`/`Artifact` split and the `(Summary, Artifact)` return tuple.

### Statistics

- **930 tests** across 29 test modules (up from 916 / 29 modules in v0.2.0).
- **19 commits** since v0.2.0.
- **14 files changed**, 545 insertions, 142 deletions.
- **26 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** `xtrg.run()` returns `(Summary, Artifact)` instead of a single
  `Summary`; `Summary` no longer has `rho`/`rho_log_scale` fields; `xtrg.ckpt` is
  replaced by `thermal.ckpt` and `progress.ckpt`.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.2.0] - 2026-08-01

**XTRG: Finite-Temperature Thermodynamics**

Introduces XTRG (eXponential Tensor Renormalization Group), Alice's second algorithm
alongside DMRG: a finite-temperature solver that computes `ρ(β) = e^{-βH}` by repeated
squaring, sharing the `1s`/`2s`/`1sp` update schemes with DMRG. `NormalMPO` is upgraded to
track its physical magnitude in log form, and `observe` computes thermal expectation-value
ratios in log-space, keeping both stable arbitrarily deep into a cooling run. One breaking
change to `NormalMPO.__init__`.

### XTRG Algorithm

- New `alice.algorithm.xtrg` package exporting `Options`, `Summary`, and `run`, mirroring
  the `alice.algorithm.dmrg` interface.
- `xtrg.run(H, spc, opts)` initializes `ρ(τ₀)` via `thermal_mpo`, then repeatedly squares
  it — `ρ(2β) ≈ compress(ρ(β) ⊗ ρ(β))` — to reach `β_max = 2^n_steps × τ₀`, sampling an
  exponentially spaced β grid.
- Three update schemes for the inner variational MPO-MPO compression `C ≈ A · B`:
  **1-site** (`1s`), **2-site** (`2s`) with SVD truncation, and **1-site-plus** (`1sp`)
  controlled bond expansion (CBE), adapted from DMRG's `'1sp'` scheme to the linear
  (non-eigenvalue) fitting problem.
- `Summary` reports `betas`, `log_z`, `free_energies`, `energies`, `specific_heats`, and
  `entropies` per cooling step; `u`, `c_V`, and `S` are derived from `log_z` via log-β
  finite differences for uniform accuracy across the exponential grid.
- `Options.checkpoint_dir` atomically checkpoints `ρ` and thermodynamic history after
  every cooling step, matching DMRG's checkpoint pattern.

### `NormalMPO`: Log-Scale Representation

- **Breaking:** `NormalMPO.__init__` keyword argument renamed from `scale` to
  `log_scale` (`log_scale = log(scale)`).
- New `log_trace()` method returns `(log|Tr[ρ]|, sign)` without ever materializing the
  raw trace, which can reach `~10^500` deep into an XTRG run; `trace()` is now a thin
  wrapper over it. New `scale_by(log_scale_delta)` for in-place log-magnitude updates.
- `__matmul__`, `__add__`, and `__mul__` combine `log_scale` by addition/subtraction
  instead of multiplying raw floats; the sign of the physical operator is folded into
  site 0's tensor data instead.

### `observe`: Log-Space Thermal Ratios

- `_observe_thermal` now computes `Tr[ρ O] / Tr[ρ]` via `log_trace()` on both numerator
  and denominator, combined as `(sign_num · sign_den) × exp(log_num − log_den)`, so the
  well-behaved O(1) ratio remains correct even when either trace overflows float64.

### Documentation

- New `docs/algorithms/` section (replacing algorithm pages formerly under `docs/api/`)
  with per-algorithm subdirectories (`algorithms/dmrg/`, `algorithms/xtrg/`).
- New `docs/examples/xtrg/` pages for free-fermion and Hubbard worked examples, validated
  against exact grand-canonical solutions.
- README's **Algorithms** section gains an XTRG subsection alongside the existing DMRG
  one, plus a new **Upcoming** list (tanTRG, TDVP, TaSK).

### Statistics

- **916 tests** across 29 test modules (up from 814 / 23 modules in v0.1.6).
- **66 commits** since v0.1.6.
- **49 files changed**, 6,172 insertions, 92 deletions.
- **28 source modules** in four subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`, `alice.algorithm.xtrg`.

### Compatibility

- **Breaking Changes:** `NormalMPO.__init__` keyword argument renamed from `scale` to
  `log_scale`; `NormalMPO.from_mpo()`, `thermal_mpo()`, and the `scale` read-only
  property are unaffected.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.1.6] - 2026-06-10

**MPS Initialization for Odd Chains**

Extends `init_mps` with a `target_qn` parameter for explicit quantum-number targeting,
fixes a silent zero-MPS bug in random initialization for odd-L chains, and replaces
`warnings.warn` with structured `logging` throughout `automps`. No breaking changes.

### `init_mps`: `target_qn` Parameter

- New `target_qn` keyword on `init_mps`: the desired right-boundary charge `Q[L]` (total
  quantum number of the chain). Defaults to `Q_vac` (half-filling), preserving behavior
  for even-L even-filling cases.
- When `target_qn` is given and the auto-config cannot reach it (e.g. `target_qn=0` for
  odd-L spin-½), `init_mps` raises a `ValueError` in both modes rather than silently
  producing an MPS in the wrong sector.

### Odd-L Random MPS Fix

- **Bug fixed:** `_random_mps` previously pinned both boundary indices to `Op['vac']`
  (charge `Q_vac`). For odd L, where no config can return to `Q_vac`, every charge block
  of the last tensor was forbidden and the MPS canonicalized to zero.
- The right boundary is now constructed with `Q[L]` (the actual charge reached by the
  auto-config path) when `target_qn` was not given explicitly. For even L this is still
  `Q_vac`; for odd L it is the correct non-vacuum charge.

### Warning System Upgrade

- `init_mps` no longer calls `warnings.warn(UserWarning)` for odd-L chains where
  `Q_vac` is unreachable. The message is now emitted via `logging.getLogger(__name__)`
  and recommends passing `target_qn` explicitly.

### `_auto_config` Generalization

- `_auto_config` now accepts `target_qn` and all three internal strategies (single-sector
  fill, period-2 alternation, greedy fallback) target `target_qn` instead of `Q_vac`.
- The function is now a pure helper with no side effects; the sole warning site is
  `init_mps`.

### Documentation

- `init-mps.md` gains a **Logic Overview** decision diagram, an **Odd-chain lengths**
  section with a spin-½ example, and a corrected **Bond Sectors in Random Mode** table
  note (BFS is seeded from `Q_c = Q[L//2]`, not from `Q_vac`).

### Statistics

- **814 tests** across 23 test modules (up from ~795 / 23 modules in v0.1.5).
- **5 commits** since v0.1.5.
- **4 files changed**, 423 insertions, 89 deletions.
- **22 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.5.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.1.5] - 2026-06-08

**Cache Isolation and Thermal MPO**

Fixes DMRG environment-block caching for concurrent runs and corrects the early-stopping
criterion in `thermal_mpo` to account for the operator norm of `H`. No breaking changes.

### DMRG Environment Cache Isolation

- `dmrg.run()` now creates a unique subdirectory (first 8 hex characters of a UUID4,
  e.g. `{env_cache_dir}/a1b2c3d4/`) inside `env_cache_dir` per invocation. Previously,
  concurrent runs sharing the same `env_cache_dir` wrote to the same `left/` and
  `right/` paths and could corrupt each other's cached blocks.
- The unique subdirectory is removed automatically in a `finally` block on return or
  exception, leaving no stale files behind.
- `Options.env_cache_dir` docstring updated to document the subdirectory scheme and
  automatic cleanup.

### `thermal_mpo` Early-Stopping Correction

- The early-stopping guard now tests `|β^n / n!| × ‖H^n‖_F < coeff_thresh` rather
  than the bare coefficient `|β^n / n!|` alone. For Hamiltonians with large operator
  norm, `‖H^n‖_F` can be much larger than 1, causing the old criterion to exit
  prematurely before the series had converged.
- The conditional that skipped the power-update step when the next coefficient was small
  has been removed; `H_pow` is now always advanced when `n < order`.
- `order` and `coeff_thresh` parameter docstrings updated to describe the corrected
  criterion.

### Statistics

- **~795 tests** across 23 test modules (up from ~740 / 23 modules in v0.1.4).
- **3 commits** since v0.1.4.
- **4 files changed**, 141 insertions, 19 deletions.
- **22 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.4.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.1.4] - 2026-05-31

**Thermal Density Matrix**

Introduces `NormalMPO`, an `MPO` subclass with a separately tracked scale factor, and
`thermal_mpo`, which approximates `ρ(β) = exp(−βH)` via a truncated Taylor series in MPO
arithmetic. The `observe` function is extended to accept `NormalMPO` as the state
argument, enabling finite-temperature expectation values within the existing workflow.
No breaking changes.

### `NormalMPO`

- New `NormalMPO` class in `alice.network.thermal`: represents an operator as
  `scale × mpo_unit`, keeping the internal MPO at unit Frobenius norm and carrying the
  physical magnitude in a separate `_scale` attribute.
- `*` (scalar multiply): only `_scale` is updated; site tensors are not modified.
  `@` (MPO product) and `+` (MPO sum) produce new site tensors with `_scale` set to
  the combined physical magnitude. `compact()` folds the extracted SVD norm into
  `_scale` after each compression sweep and restores standard bond-arrow directions
  via `capcup`.
- Class method `NormalMPO.from_mpo(mpo)` constructs a `NormalMPO` from any plain `MPO`.
- Exported from `alice.network` and the `alice` top-level namespace.

### `thermal_mpo`

- New `thermal_mpo(H, beta, order, spc)` function: computes
  `ρ(β) ≈ Σ (−β)^n/n! · H^n` via MPO arithmetic, calling `compact()` after every
  addition and power step to control bond growth. Early exit when the Taylor coefficient
  drops below `coeff_thresh` (default `1e-15`).
- The returned `NormalMPO` carries `_scale ≈ Tr[ρ]` (the partition function, up to the
  MPO norm).
- Exported from `alice.network` and the `alice` top-level namespace.

### `observe` Updated for Thermal States

- `observe` now accepts a `NormalMPO` as the state argument: forms the MPO product
  `ρ @ O`, compresses it with `compact()`, and returns `Tr[ρ O] / Tr[ρ]` — the
  normalized thermal expectation value.

### Documentation

- New API reference pages for `NormalMPO` and `thermal_mpo`; `alice.network` index and
  `observe` reference updated to document the new types.

### Statistics

- **~740 tests** across 23 test modules (up from ~700 / 22 modules in v0.1.3).
- **13 commits** since v0.1.3.
- **9 files changed**, 1,192 insertions, 10 deletions.
- **22 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.3.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.7.

---

## [0.1.3] - 2026-05-13

**MPS Init and DMRG Checkpointing**

Introduces `init_mps`, a universal MPS initializer that replaces the per-example
`_random_mps` helpers with a single particle-type–agnostic function. Adds atomic
per-sweep checkpointing to DMRG via `dmrg.Options.checkpoint_dir`, and exposes
`Network.bond_states` for counting physical states at each bond in non-Abelian
simulations. No breaking changes.

### Universal MPS Initializer

- New `alice.network.automps` module with `init_mps`: constructs an initial MPS for
  DMRG from any `(Spc, Op)` pair returned by `load_space`. Works for bosonic,
  fermionic, and conductor sites without a `particle_type=` argument.
- `bond_dim=1`: deterministic product state with exact charge targeting, one sector per
  bond; best paired with CBE (`scheme='1sp'`) or 2-site (`scheme='2s'`) DMRG.
- `bond_dim>1`: random MPS with group-derived bond sectors; reachable charges are
  selected by BFS from the center-bond charge to depth 2.
- Auto-balanced `config=None` heuristic covers all standard even-*L* half-filled cases:
  alternating high/low for 2-sector spaces, single neutral sector for 3-sector spaces,
  alternating neutral-pair for 4-sector spaces, SU(2) dimer path for pure-SU(2) spaces.
- Exported from `alice` top-level namespace and from `alice.network`.
- DMRG example scripts (`dmrg_heisenberg`, `dmrg_freefermion`, `dmrg_conductor`)
  updated to accept an `init` parameter (`'iter_diag'` or `'random'`) using `init_mps`;
  per-example `_random_mps` helpers removed.

### DMRG Checkpointing

- New `checkpoint_dir` option in `dmrg.Options`: after every completed sweep the
  current state is serialized to `dmrg.ckpt` via an atomic write (write to
  `dmrg_lock.ckpt`, then rename). On POSIX systems the rename is atomic, so a crash
  during serialization cannot corrupt the previous checkpoint.
- Checkpoint file is in PyTorch format, loadable via `dmrg.Summary.load`.
- Defaults to the current working directory at `run()` call time (matching `.logging`).

### `Network.bond_states`

- New `bond_states` property on `Network`: returns the number of physical states per
  internal bond (length `L - 1`). Equal to `bond_dims` for Abelian groups; larger for
  SU(2) due to multiplet degeneracy (*2j+1* states per multiplet of spin *j*).

### Documentation

- New API reference page for `init_mps` with parameter table, charge-convention notes,
  and runnable examples; `Network` reference updated with `bond_states`.
- Quick-start guide updated to use `init_mps` and the `Summary` checkpoint interface.

### Statistics

- **~700 tests** across 22 test modules (up from ~650 / 21 modules in v0.1.2).
- **16 commits** since v0.1.2.
- **17 files changed**, 1,319 insertions, 187 deletions.
- **21 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.2.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.6.

---

## [0.1.2] - 2026-05-11

**Geometry Expansion and Refactor**

Introduces Kagome lattice support, a full refactor of the geometry subsystem around a
`Geometry` dataclass, a unified traversal-order naming scheme, and ASCII text diagrams
for MPS and MPO chains. Also adds `Sp4`/`Sm4` operator templates to `build_bosonic` and
`build_conductor`. Several breaking changes to the geometry and `build_interaction` APIs;
the DMRG, Hamiltonian, and algorithm APIs are unchanged.

### Kagome Lattice

- New `alice.physics.kagome` module with `intrcmap_kagome`: nearest-neighbor bond
  generation for Kagome lattices, covering N2U (upward-triangle: A–B, A–C, B–C within
  each unit cell) and N2D (downward-triangle: bonds between adjacent unit cells), with
  OBC/PBC boundary conditions along both axes.
- Traversal orders `'sequential'` (column-major, default) and `'serpentine'`
  (column-major with alternating row direction) for Kagome lattices.
- `intrcmap_kagome` is exported from `alice.physics` alongside `intrcmap_1dchain`
  and `intrcmap_square`.

### Geometry Module Refactor

- New `Geometry` dataclass returned by `build_geometry`; carries `cfg`, `ord_map`,
  and `latt` as a single typed object with derived properties `lattice`, `traverse`,
  `lx`, `ly`, `L`, and helpers `to_1d`/`to_2d`.
- All `intrcmap_*` functions now take a `Geometry` as input (previously a plain dict)
  and continue to return `List[Interaction2Site]`.
- `build_interaction` now returns `(interactions, spc, geo)` — the third element
  changed from an `int` (site count) to the `Geometry` instance.
- New `build_intrcmap(geo)` dispatcher: reconstructs the bond list from any `Geometry`
  instance, decoupling lattice construction from bond enumeration.
- New `build_traversal` helpers in all three lattice modules (`chain.py`, `square.py`,
  `kagome.py`); 1D chain and square lattice geometries separated into their own modules.
- `ord_map` keys changed from integer flat indices to coordinate tuples: `(row, col)`
  for chain and square, `(row, col, u)` for Kagome (where `u` is the sublattice index
  0=A, 1=B, 2=C), making coordinate lookups explicit for all lattice types.

### Traversal Order Naming

- `'snake'` is renamed to `'serpentine'` (same behavior: columns alternate
  direction, even columns top→bottom, odd columns bottom → top).
- `'sequential'` is a new order (all columns top → bottom, no reversal) and is
  now the default for square and Kagome lattices.
- All documentation, example configs, and tests updated to the new naming.

### MPS/MPO Text Display

- New `alice.network.display` module with `network_summary`: Unicode ASCII-art chain
  diagrams showing tensor nodes, bond dimensions, center site, and summary info.
- `MPS.__repr__` and `MPO.__repr__` delegate to `network_summary`, giving readable
  representations in REPLs and notebooks.

### New Operator Templates

- `build_bosonic` and `build_conductor` gain `Sp4`/`Sp4dag`/`Sm4`/`Sm4dag` templates
  (U(1)-only), enabling AutoMPO construction for models where raising and lowering
  channels must be handled separately.

### Documentation

- New API reference pages for `intrcmap_kagome` and `build_intrcmap`; geometry and
  traversal pages updated to reflect the refactored two-stage construction pipeline.
- All documentation page headings standardized to title case.

### Statistics

- **~650 tests** across 21 test modules (up from 538 / 19 modules in v0.1.0).
- **119 commits** since v0.1.1.
- **65 files changed**, 3,380 insertions, 824 deletions.
- **20 source modules** in three subpackages: `alice.network`, `alice.physics`,
  `alice.algorithm.dmrg`.

### Compatibility

- **Breaking Changes:**
    - `build_geometry` return type changed from `List[Interaction2Site]` to `Geometry`.
    - All `intrcmap_*` functions now take a `Geometry` as input instead of a plain dict;
      return type `List[Interaction2Site]` is unchanged.
    - `build_interaction` third return value changed from `int` (site count) to `Geometry`.
    - `geometry_fn` callable override signature changed from
      `(geo: dict) -> List[Interaction2Site]` to `(geo_cfg: dict) -> Geometry`; a new
      `intrcmap_fn` override was added with signature
      `(geo: Geometry) -> List[Interaction2Site]`.
    - The public function `generate_snake_order` is removed; its traversal logic is now
      internal.
    - The TOML key `traverse: 'snake'` is deprecated; `'snake'` was renamed to
      `'serpentine'`, and `'sequential'` is the new default order.
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.6.

---

## [0.1.1] - 2026-05-06

**Documentation Website**

Introduces the complete Alice documentation site: Material-themed MkDocs with a full API
reference, worked DMRG examples, and getting-started guides. Also
renames the PyPI distribution to `alice-net`. No changes to the Alice project; fully
backward compatible with v0.1.0.

### Documentation Infrastructure

- Custom `alice` color scheme, three-tab navigation, dark/light/system theme switching, search with suggestions and shareable links
- Content features: code copy, inline annotations, tabbed content, in-page edit and view actions
- Custom home page template (`docs/overrides/home.html`) with hero image and feature cards; custom stylesheet (`docs/stylesheets/extra.css`)
- Build hook (`docs/hooks.py`) for table-cell bullet-list post-processing
- Plugins: **mkdocstrings** (NumPy-style API docs), **markdown-exec** (configured for future live execution), **git-revision-date-localized**, **git-committers**
- Extensions: MathJax via `arithmatex`, Mermaid via `superfences`, tabbed content, FontAwesome/Material emoji

### Getting Started (7 pages)

- **What is Alice**: philosophy, relationship with Nicole, supported symmetry groups and algorithms
- **Installation**: `pip install alice-net` / `uv add alice-net`, development setup, optional dependency groups
- **Core Concepts**: MPS/MPO block-sparse structure, symmetry sectors, DMRG sweep logic
- **Quick Start**: end-to-end Heisenberg DMRG example from site definition through energy output
- **Contributing**: branch model, coding conventions, test requirements
- **Git Control**: tagging, branching, and release workflow for the Alice project
- **Changelog**: version history beginning with v0.1.0

### API Reference (30 pages)

- **Network** — `Network`, `MPS`, `MPO`: construction, canonicalization, norm, serialization, SVD compression, norm redistribution; `observe`: expectation-value sweep with SU(2) Bridge weight support
- **Interaction** — `Interaction`, `Interaction1Site`, `Interaction2Site` dataclass references; `build_interaction`: TOML-configured builder with plugin section documentation
- **Geometry** — `generate_snake_order`, `intrcmap_1dchain`, `intrcmap_square`, `build_geometry`: function references with parameter tables and usage notes
- **Local Space** — `build_bosonic`, `build_fermionic`, `build_conductor`: site Hilbert space constructors with symmetry-mode tables
- **Hamiltonian** — `build_hamiltonian`: MPO assembler; `build_heisenberg`, `build_free_fermion`, `build_hubbard`: model-specific builders
- **DMRG** — `dmrg.Options`: parameter reference with TOML key mapping; `dmrg.Summary`: output fields and serialization; `dmrg.run`: sweep logic, update schemes (`1s`, `2s`, `1sp`), convergence criteria
- **Logging** — `configure_logging`: handler configuration, log levels, output file naming

### Examples (8 pages)

- **Heisenberg chain**: ground-state energy of a spin-1/2 chain, U(1) and SU(2) symmetry comparison
- **Free fermion**: tight-binding chain benchmark against exact diagonalization
- **Hubbard model**: charge and spin sector targeting in a single-band system
- **AutoMPO from TOML**: `[[interaction]]` tables, `[plugin]` sections for user-defined models, built-in presets
- **Custom geometry**: implementing a user-defined lattice traversal and registering it with `build_geometry`
- **Custom local space**: defining a new site Hilbert space outside the built-in presets
- **Custom model**: wrapping a user-defined Hamiltonian function as an Alice-compatible builder

### Packaging

- PyPI distribution renamed from `alice` to `alice-net`; `pyproject.toml` updated with wheel target and project URLs; README and logging banner updated
- **Install:** `pip install alice-net` or `uv add alice-net`; the import namespace `alice` is unchanged

### Statistics

- **45 documentation pages**; all public symbols documented
- **39 commits** since v0.1.0
- **112 files changed**, 3,161 insertions, 63 deletions

### Compatibility

- **Breaking Changes:** None — fully backward compatible with v0.1.0; import paths, function signatures, and TOML configuration formats are unchanged
- **Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Nicole ≥ 0.3.6

---

## [0.1.0] - 2026-05-04

Initial stable release of Alice.

### Tensor Network Infrastructure

- `MPS`, `MPO`, and `Network` classes: block-sparse matrix product states and operators inheriting Nicole's exact symmetry engine with full support for U(1), Z(2), SU(2), and product symmetry groups.
- `Network.canonical()`: left/right QR canonicalization with configurable bond truncation.
- `Network.norm()` and `normalize()`: Frobenius norm via the center tensor or a full transfer-matrix sweep.
- `Network.serialize()` / `Network.deserialize()`: `torch.save`-compatible serialization preserving all index metadata.
- `MPO.compact()` and `MPO.redistribute_norm()`: SVD compression and norm redistribution.
- `observe()`: expectation-value computation via MPS–MPO–MPS transfer-matrix sweep, with SU(2) Bridge weight support.

### AutoMPO Construction

- `build_interaction()`: TOML-configured interaction map builder with built-in Heisenberg, free-fermion, and Hubbard model presets and support for user-defined model functions via `[plugin]` sections.
- `build_hamiltonian()`: term-by-term MPO assembler using Nicole's `oplus`; compresses the result with two canonical sweeps; handles non-Abelian SU(2).
- `Interaction`, `Interaction1Site`, `Interaction2Site` dataclasses.
- Incremental `compact_every` parameter to limit bond dimension growth during large builds.

### Lattice Geometries

- `intrcmap_1dchain()`: nearest-neighbor 1D chain with optional PBC.
- `intrcmap_square()`: 2D square lattice with NN and NNN bonds, PBC support, and snake-order traversal.
- `generate_snake_order()`: boustrophedon traversal for rectangular lattices.
- `build_geometry()`: TOML dispatcher for lattice and traversal selection.

### Physical Spaces

- `build_bosonic()`: spin-`s` site; U(1) and SU(2) symmetry.
- `build_fermionic()`: spinless-fermion site; U(1) and Z(2) symmetry.
- `build_conductor()`: spinful-fermion (Band) site; U(1)×U(1), Z(2)×U(1), U(1)×SU(2), Z(2)×SU(2) symmetry.

### Physics Models

- `build_heisenberg()`: Heisenberg spin model with NN coupling `J` and optional NNN `J'`.
- `build_free_fermion()`: spinless tight-binding chain with hopping `t`, optional NNN `t'`, and chemical potential `µ`.
- `build_hubbard()`: Hubbard model with hopping `t`, on-site `U`, optional NNN `t'`, and `µ` at half-filling.

### DMRG Algorithm

- `dmrg.Options` dataclass: all run parameters with TOML loading via `Options.from_toml()`.
- `dmrg.Summary` dataclass: energy, optimized MPS, convergence info, and serialization.
- `dmrg.run()`: alternating sweep optimization with three update schemes (`1s`, `2s`, `1sp`).
- Davidson eigensolver with thick restart operating directly in the symmetry-block-sparse space.
- `Environment` class with optional disk-spilling (sliding window + async I/O).

### Logging

- `configure_logging()`: one-call setup of Alice's two-handler logging scheme (console + timestamped file). Emits startup banner, model specification summary, sweep-by-sweep diagnostics, and geometry ASCII diagrams.

### Statistics

- 538 tests across 19 test modules.
- 196 commits across 10+ feature branches.
- 64 files, ~16,000 lines of code.
- 16 source modules in three subpackages: `alice.network`, `alice.physics`, `alice.algorithm.dmrg`.

[0.3.2]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.3.2
[0.3.1]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.3.1
[0.3.0]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.3.0
[0.2.8]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.8
[0.2.7]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.7
[0.2.6]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.6
[0.2.5]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.5
[0.2.4]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.4
[0.2.3]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.3
[0.2.2]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.2
[0.2.1]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.1
[0.2.0]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.2.0
[0.1.6]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.6
[0.1.5]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.5
[0.1.4]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.4
[0.1.3]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.3
[0.1.2]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.2
[0.1.1]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.1
[0.1.0]: https://github.com/Ideogenesis-AI/Alice/releases/tag/v0.1.0
