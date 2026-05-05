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

Fast-forward keeps the history linear but gives no explicit record that a branch existed. Alice discourages fast-forwards for branch integration — all merges into `develop` and `stable` use `--no-ff` so the history preserves the context of each contribution.

### Three-way merge (no-ff)

When either branch has advanced independently, or when `--no-ff` is used explicitly, Git creates a merge commit that records both parents:

```
Before:  develop ──A──B──E
                    └── feature (C──D)

After:   develop ──A──B──E──M   (M has parents E and D)
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

Rebasing produces a clean, linear history, but it **rewrites commit hashes**. Because of this:

- **Never rebase a branch that has been pushed and shared with others** — it forces everyone else to reconcile diverged histories.
- Rebase is appropriate for tidying up a *local*, *unpushed* branch before opening a pull request (e.g. squashing fixup commits into a coherent unit).
- Do not rebase your branch onto `develop` as a substitute for a merge — open a pull request instead.

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

Once `develop` is up to date locally, you can merge it into your working branch to incorporate the latest changes:

```bash
git checkout feature/my-feature
git merge develop
```

This keeps your branch compatible with the current state of `develop` and reduces the risk of large conflicts when your pull request is eventually reviewed.

### Syncing from the GitHub web interface

If you prefer not to use the command line for routine syncing, GitHub's web interface offers a **"Sync fork"** button on your fork's main page. This updates `develop` and `stable` in your fork directly. You still need to `git fetch origin` and `git merge` locally afterwards to bring your local clone in line.

---

## Summary Checklist

Before opening a pull request, confirm the following:

- [ ] Work is on a correctly named branch (`feature/…`, `fix/…`, etc.) — not on `develop` or `stable`.
- [ ] Your local `develop` is up to date with `upstream/develop` (via `fetch` + `merge --ff-only`).
- [ ] Your feature branch has been merged with the latest `develop` to resolve conflicts locally.
- [ ] All tests pass (`pytest`).
- [ ] The pull request targets `develop`, not `stable`.
