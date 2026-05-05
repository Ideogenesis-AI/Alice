# Git Control

This page covers the branching strategy, workflow conventions, and day-to-day Git practices used in the Alice project. Whether you are an algorithm contributor maintaining your own branch or a core collaborator integrating changes, understanding these conventions will help you work cleanly and avoid common pitfalls.

---

## Branching Strategy

Alice uses a two-tier protected model with a clear separation of responsibilities.

### Protected branches

| Branch | Purpose | Who can push |
|--------|---------|--------------|
| `stable` | Released, production-ready code. Tagged with version numbers. | Project maintainer only |
| `develop` | Integration branch. All reviewed contributions land here before a release. | Project maintainer only |

!!! warning "Do not push directly to `stable` or `develop`"
    Direct commits to `stable` or `develop` are not permitted for external contributors. All changes reach `develop` exclusively through pull requests.

### Contributor branches

Contributors create their own branches off `develop` using a structured naming convention:

| Prefix | When to use |
|--------|-------------|
| `feature/<name>` | A new algorithm, module, or substantial capability |
| `update/<name>` | An improvement or extension to existing functionality |
| `fix/<name>` | A bug fix |
| `docs/<name>` | Documentation-only changes |

For example: `feature/tdvp-algorithm`, `fix/dmrg-trunc-threshold`, `docs/api-hamiltonian`.

Each contributor branch is owned by its author. Algorithm contributors may maintain long-lived branches for their respective algorithms and open pull requests to `develop` when a piece of work is ready for integration.

### Release flow

```
feature/... ──┐
update/...  ──┤──► develop ──► stable (tagged release)
fix/...     ──┘
```

`stable` is only updated from `develop` by the project maintainer at release time.

---

## Git Merging — Key Concepts

Understanding how Git integrates histories helps you avoid surprises when collaborating.

### Fast-forward merge

When your branch has no divergence from its base (i.e. the base has not moved since you branched off it), Git simply advances the base pointer forward:

```
Before:  develop ──A──B
                      └── feature (C──D)

After:   develop ──A──B──C──D   (no merge commit)
```

Fast-forward keeps the history linear but leaves no explicit record that a branch existed.

**When to use fast-forward:**
Fast-forward is the right choice when syncing your local `stable` or `develop` with the upstream primary repository. In that case, you are not integrating diverged work — you are simply advancing your local pointer to reflect what the maintainer has already published. A merge commit would be meaningless noise here.

**When not to use fast-forward:**
All merges of contributor branches *into* `develop` or `stable` must use `--no-ff`, so the history preserves the context of each contribution as a distinct unit. This is enforced by the maintainer at integration time.

### Three-way merge (no-ff)

When either branch has advanced independently, or when `--no-ff` is used explicitly, Git creates a merge commit that records both parents:

```
Before:  develop ──A──B──E
                      └── feature (C──D)

After:   develop ──A──B──E─────────────────M  (M has parents E and D)
                      └── feature (C──D) ──┘
```

This is the standard integration style in Alice. The merge commit serves as a clear integration record.

### Rebase

Rebase re-applies your commits on top of the current tip of the target branch, rewriting their hashes:

```
Before:  develop ──A──B──E
                      └── feature (C──D)

After rebase onto develop:
         develop ──A──B──E──C'──D'   (new commits, old ones gone)
```

Rebasing produces a clean, linear history. It has two distinct use cases in the Alice workflow:

1. **Bringing a contributor branch up to date with `develop`**

    When `develop` has moved forward while you have been working on your branch, you should rebase your branch onto the current tip of `develop` before opening a pull request. This re-anchors your commits to the latest shared state, resolves conflicts in your own workspace rather than the reviewer's, and keeps the resulting merge commit clean.

    ```bash
    # Make sure your local develop is current first (see "Keeping Your Fork Up to Date")
    git checkout feature/my-feature
    git rebase develop
    # Resolve any conflicts, then:
    git rebase --continue
    ```

2. **Tidying up a local branch before submitting**

    Before opening a pull request you can rebase interactively to squash small fixup commits into logical units, reword messages, or reorder changes for clarity.

    ```bash
    git rebase -i develop
    ```

**When not to rebase:**
Rebase **rewrites commit hashes**. If your contributor branch is actively shared with other collaborators who have already pulled it, rebasing forces them to reconcile diverged histories. In that situation, merging `develop` into your branch is the safer option — see [Updating your feature branch](#updating-your-feature-branch) for details.

---

## Keeping Your Fork Up to Date

Because `stable` and `develop` advance as new work is integrated, it is important to keep your local clone and your GitHub fork in sync with the primary repository. Falling behind causes unnecessary conflicts and makes reviews harder.

### Initial setup — add the upstream remote

If you cloned your fork, add the primary repository as a second remote called `upstream` (you only need to do this once):

```bash
git remote add upstream https://github.com/Ideogenesis-AI/Alice.git
```

Verify both remotes are present:

```bash
git remote -v
# origin    https://github.com/your-username/Alice.git (fetch)
# origin    https://github.com/your-username/Alice.git (push)
# upstream  https://github.com/Ideogenesis-AI/Alice.git (fetch)
# upstream  https://github.com/Ideogenesis-AI/Alice.git (push)
```

### Fetching vs. pulling

!!! tip "Prefer `fetch` over `pull` in collaborative projects"
    `git pull` is shorthand for `git fetch` followed by an automatic `git merge` (or `git rebase`, depending on configuration). In a shared project, this automatic merge step can silently introduce a merge commit into your local branch, cluttering history and making it harder to track what actually changed.

    Use `git fetch` instead: it downloads the latest references without touching any of your local branches, so you can inspect the changes before deciding how to integrate them.

### Step-by-step: syncing `develop`

```bash
# 1. Fetch all new history from the primary repository
git fetch upstream

# 2. Switch to your local develop branch
git checkout develop

# 3. Bring your local develop up to date with upstream/develop
#    (fast-forward only — if this fails, investigate before forcing)
git merge --ff-only upstream/develop

# 4. Push the updated develop to your GitHub fork
git push origin develop
```

### Step-by-step: syncing `stable`

The same pattern applies to `stable`:

```bash
git fetch upstream
git checkout stable
git merge --ff-only upstream/stable
git push origin stable
```

### Updating your feature branch

As work is merged into `develop` by the maintainer, your contributor branch will gradually fall behind. The recommended approach for keeping your branch current is **rebase**, because it replays your commits on top of the latest `develop` without embedding a merge commit in the middle of your own work — making the history graph cleaner and the eventual pull request easier to review.

```bash
# 1. Make sure your local develop is up to date (see above)
git checkout develop
git merge --ff-only upstream/develop

# 2. Rebase your branch onto the updated develop
git checkout feature/my-feature
git rebase develop
# If there are conflicts, resolve them file by file, then:
git rebase --continue
```

!!! note "When to use merge instead"
    Contributor branches are sometimes shared among multiple collaborators. In that case, rebasing rewrites commit hashes and forces every collaborator to reconcile diverged histories — which can be disruptive. If your branch is actively shared, it is acceptable to merge `develop` into it instead:

    ```bash
    git checkout feature/my-feature
    git merge develop
    ```

    This introduces one merge commit from `develop` into your branch, which is a minor history cost but avoids the coordination overhead of a rebase. Use your judgement: rebase when you are the sole author or when the branch has not yet been pushed; merge when multiple people are working on the same branch.

    Even when merging is necessary, try to **delay it as long as possible**. Only merge `develop` into your branch when divergence is actively causing problems — conflicts that block your work, or when a dependency you need has been introduced in `develop`. Merging eagerly on every upstream update accumulates unnecessary merge commits and clutters the history graph. The goal is still to keep the graph as clean as possible; merging from `develop` should be a deliberate, infrequent act rather than a routine sync step.

### Syncing from the GitHub web interface

If you prefer not to use the command line for routine syncing, GitHub's web interface offers a **"Sync fork"** button on your fork's main page. This updates `develop` and `stable` in your fork directly. You still need to `git fetch origin` and `git merge` locally afterwards to bring your local clone in line.

---

## Summary Checklist

Before opening a pull request, confirm the following:

- [ ] Work is on a correctly named branch (`feature/…`, `fix/…`, etc.) — not on `develop` or `stable`.
- [ ] Your local `develop` is up to date with `upstream/develop` (via `fetch` + `merge --ff-only`).
- [ ] Your feature branch is up to date with the latest `develop` (via rebase, or merge if the branch is shared).
- [ ] All tests pass (`pytest`).
- [ ] The pull request targets `develop`, not `stable`.
