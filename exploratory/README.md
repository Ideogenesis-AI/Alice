# exploratory/

Research paths kept for reference but **not** the supported integrator.
The supported two-site BUG is `src/alice/algorithm/two_site_bug/`; its
discarded-projector kernel is the one mirrored by `bond_update_bug!` in
BUG-Julia (verified to 4.27e-11 on the L=6 Heisenberg Sz profile).

- `global_sweep/` — the discarded-projector BUG as a single global sweep
  (`phi = H*psi` formed once, augmented bases spanning `range(psi) + range(H psi)`).
  Measured first-order in the state; superseded by the per-bond kernel.
  `global_sweep_tests/` holds its tests.

Neither directory is importable from `alice.algorithm` any more, and neither is
collected by the default pytest run (`--ignore=exploratory`).

`global_sweep_tests/` therefore **cannot run as-is** — it imports
`alice.algorithm.discarded_bug`, which no longer exists. It is a record, not a
working suite; restore the module to `src/alice/algorithm/` to run it. One of its
tests was already failing before it was parked
(`TestRankAdaptivity::test_max_bond_cap_respected`), which is visible in the
pre-cleanup baseline.
