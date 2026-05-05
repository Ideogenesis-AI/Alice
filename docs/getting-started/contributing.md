# Contributing

Alice welcomes contributions from researchers, students, and developers. This guide explains everything you need to get involved — from filing a simple bug report to implementing a new algorithm.

If you have never used GitHub before, don't worry: this page walks through every step from scratch.

## Ways to Contribute

- **Implement a new algorithm.** Each algorithm (XTRG, tanTRG, TDVP, TaSK) is an independent unit on top of the shared MPS/MPO core. If you have expertise in one of these methods, your involvement is especially valued. See [Algorithm contributions](#algorithm-contributions) below.
- **Fix a bug or improve an existing algorithm.**
- **Improve the documentation.** Clear, accurate docs are as important as correct code.
- **Report issues and request features** via [GitHub Issues](https://github.com/Ideogenesis-AI/Alice/issues).
- **Share benchmarks and use cases.**

---

## Setting Up Your Development Environment

### Step 1 — Create a GitHub account

If you don't have one, go to [github.com](https://github.com) and sign up for a free account.

### Step 2 — Fork the repository

1. Go to [https://github.com/Ideogenesis-AI/Alice](https://github.com/Ideogenesis-AI/Alice).
2. Click the **Fork** button near the top-right corner of the page.
3. GitHub creates a copy of the repository under your account, e.g. `https://github.com/your-username/Alice`.

A fork is your personal copy of the project. Changes you make there do not affect the original repository until you submit a pull request.

### Step 3 — Clone your fork locally

Open a terminal on your computer and run:

```bash
git clone https://github.com/your-username/Alice.git
cd Alice
```

This downloads the repository to a folder called `Alice` on your machine.

### Step 4 — Add the upstream remote

This lets you pull the latest changes from the original Alice repository into your fork:

```bash
git remote add upstream https://github.com/Ideogenesis-AI/Alice.git
```

Verify that it worked:

```bash
git remote -v
```

You should see both `origin` (your fork) and `upstream` (the original).

### Step 5 — Install in editable mode

```bash
pip install -e ".[test,lint,docs]"
```

This installs Alice in *editable* mode: any change you make to the source files under `src/` is immediately reflected without reinstalling.

---

## The Development Workflow

### Step 1 — Sync with upstream

Before starting new work, bring your fork up to date:

```bash
git fetch upstream
git checkout develop
git merge upstream/develop
```

### Step 2 — Create a feature branch

Never commit directly to `develop` or `stable`. Create a branch named after your change:

```bash
git checkout -b feature/my-new-feature
```

Use descriptive names, e.g. `feature/tdvp-algorithm`, `fix/dmrg-bond-dim`, `docs/contributing-guide`.

### Step 3 — Make your changes

Edit, add, or delete files as needed. Follow the guidelines in [Code style](#code-style) below.

### Step 4 — Run the tests

```bash
pytest
```

All existing tests must pass. Add new tests for any new functionality you introduce.

### Step 5 — Commit your changes

Stage the files you changed and write a clear commit message:

```bash
git add src/alice/algorithm/my_module.py tests/algorithm/test_my_module.py
git commit -m "Add initial TDVP implementation with unit tests"
```

Good commit messages are short (≤72 characters in the subject line), in the imperative mood ("Add …", "Fix …", "Update …"), and explain *what* changed, not *how*.

### Step 6 — Push to your fork

```bash
git push origin feature/my-new-feature
```

### Step 7 — Open a pull request

1. Go to your fork on GitHub: `https://github.com/your-username/Alice`.
2. GitHub will notice your recently pushed branch and show a **"Compare & pull request"** banner — click it.
3. On the pull request form:
   - Set the **base branch** to `develop` (the Alice `develop` branch, **not** `stable`).
   - Write a clear title and description of what the PR does and why.
4. Click **"Create pull request"**.

!!! warning "Always target `develop`"
    Pull requests must always target the `develop` branch. Direct PRs to `stable` will be closed. The `develop` branch is the integration point; `stable` only receives reviewed, tested changes promoted by the maintainers.

### Step 8 — Respond to review

A maintainer will review your pull request and may request changes. Respond to comments, push additional commits to the same branch, and the PR will update automatically. Once approved, your contribution will be merged into `develop`.

---

## Code Style

Alice enforces consistent code style with `ruff`. Before pushing, run:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

Additional conventions:

- **Type hints** on all public function signatures.
- **NumPy-style docstrings** for all public functions and classes (see existing code for examples).
- **No `result` as a variable name** (Alice convention — use a descriptive name).
- **`index` / `axis` / `edge`** terminology consistent with Nicole; never "leg" or "legs".
- Line length ≤ 100 characters.

---

## Testing

Alice uses `pytest`. Tests live under `tests/`, mirroring the `src/alice/` package structure:

| Package | Test directory |
|---------|---------------|
| `alice.network` | `tests/network/` |
| `alice.algorithm.dmrg` | `tests/algorithm/` |
| `alice.physics` | `tests/physics/` |

Run all tests:

```bash
pytest
```

Run a specific module:

```bash
pytest tests/network/test_mps.py -v
```

Run only fast tests (skip slow ones):

```bash
pytest -m "not slow"
```

New code must be accompanied by tests that cover both typical usage and edge cases.

---

## Documentation

Documentation lives under `docs/`. To build and preview locally:

```bash
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

User-facing changes must include documentation updates. This includes:

- New public functions or classes → add an API page under `docs/api/` and link it from `docs/api/index.md`.
- New behaviour in existing functions → update the relevant API page.
- New examples → add a page under `docs/examples/`.

---

## Algorithm Contributions

Alice is structured around a shared core (MPS, MPO, network infrastructure) on top of which each algorithm is an independent unit. This means:

- Individual algorithms can be contributed, extended, and maintained by different people.
- Each contributor's work is attributed accordingly. Algorithm ownership is tracked in the source file headers and the release notes.

If you want to implement one of the planned algorithms (XTRG, tanTRG, TDVP, TaSK) or take ownership of its maintenance, please [open an issue](https://github.com/Ideogenesis-AI/Alice/issues) first to discuss the design and coordinate with the maintainers. Early design discussion prevents duplicate work and ensures the implementation integrates cleanly with the shared core.

---

## Reporting Issues

Found a bug or have a feature request? Open an issue at [https://github.com/Ideogenesis-AI/Alice/issues](https://github.com/Ideogenesis-AI/Alice/issues).

Good bug reports include:

- Alice version (`pip show alice-net`).
- A minimal, self-contained code snippet that reproduces the problem.
- The full error traceback.
- What you expected to happen vs. what actually happened.

---

## Authors and Maintainers

Alice is created and maintained by [Changkai Zhang](https://chx-zh.cc) as part of the Ideogenesis-AI effort. The core infrastructure is maintained by the original author; individual algorithms have their own maintenance teams. For questions about contributing or collaboration opportunities, open an issue or contact the responsible team directly.
