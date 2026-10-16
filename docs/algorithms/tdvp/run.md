# run

Top-level TDVP entry point.

::: alice.algorithm.tdvp.run
    options:
      heading_level: 2

!!! note "`run()` evolves the state in place"

    The returned `Artifact` holds the *same* `MPS` object that was passed in,
    not a copy, so the caller's reference to the initial state is advanced
    along with it. Snapshot it first if the initial state is needed for
    comparison — cloning tensor by tensor rather than with `copy.deepcopy`,
    which cannot pickle the intertwiners of a non-Abelian tensor:

    ```python
    reference = MPS([state[i].clone() for i in range(state.L)],
                    center=state.center)
    reference.to(dtype=torch.complex128)
    ```

## See Also

- [Options](options.md) — configure the run.
- [Summary](summary.md) — evolution history.
- [Artifact](artifact.md) — initial and final state.
