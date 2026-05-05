# Installation

Alice is distributed on PyPI under the package name `alice-net`. The import name is `alice`.

## Requirements

- Python ≥ 3.11
- [PyTorch](https://pytorch.org/) ≥ 2.5
- [Nicole](https://ideogenesis-ai.github.io/Nicole/) ≥ 0.3.6 (installed automatically)
- NumPy ≥ 2.0 (installed automatically)

## Install from PyPI

=== "pip"

    ```bash
    pip install alice-net
    ```

=== "uv"

    ```bash
    uv add alice-net
    ```

=== "conda (via pip)"

    ```bash
    conda create -n alice python=3.11
    conda activate alice
    pip install alice-net
    ```

## Verify the Installation

```python
import alice
print(alice.__version__)
```

You should see `0.1.0` (or whichever version was installed).

## Optional Extras

### Development and Testing

```bash
pip install "alice-net[test]"
```

Installs `pytest`, `hypothesis`, and `pytest-benchmark` for running the test suite.

### Linting and Type Checking

```bash
pip install "alice-net[lint]"
```

Installs `ruff` and `mypy`.

### Documentation

```bash
pip install "alice-net[docs]"
```

Installs MkDocs, the Material theme, mkdocstrings, and all plugins needed to build and serve this documentation locally:

```bash
mkdocs serve
```

## GPU Support

Alice inherits PyTorch's device support from Nicole. To run on CUDA:

```python
import torch
import alice

# Check that a CUDA device is available
print(torch.cuda.is_available())
```

Pass `device="cuda"` when deserializing an MPS or Summary, or move tensors to GPU using standard Nicole/PyTorch `.to(device)` calls on individual site tensors.

## Development Install

To contribute to Alice or run the latest unreleased code, clone the repository and install in editable mode:

```bash
git clone https://github.com/Ideogenesis-AI/Alice.git
cd Alice
pip install -e ".[test,lint,docs]"
```

See the [Contributing](contributing.md) guide for the full development workflow.
