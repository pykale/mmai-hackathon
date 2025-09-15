# MultimodalAI'25 Hackathon Base Source Code

[![tests](https://github.com/pykale/mmai-hackathon/workflows/test/badge.svg)](https://github.com/pykale/mmai-hackathon/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/pykale/mmai-hackathon/branch/main/graph/badge.svg)](https://codecov.io/gh/pykale/mmai-hackathon)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/pykale/mmai-hackathon/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)

## Overview

This repository provides the base source code for the MultimodalAI'25 workshop Hackathon. It is designed to help participants get started quickly with a pre-configured Python environment and essential dependencies for development and testing.

## Requirements

- Python 3.10, 3.11, or 3.12
- Git

## Installation

The steps below are linear and work with `venv`, `conda`, or `uv`. Pick one method and follow it end‑to‑end.

### 1) Clone and create an environment

```bash
git clone https://github.com/pykale/mmai-hackathon.git
cd mmai-hackathon

# conda (recommended)
conda create -n mmai-hackathon python=3.11 -y
conda activate mmai-hackathon


# venv (alternative)
# python3 -m venv .venv && source .venv/bin/activate

# uv (alternative)
# uv venv .venv && source .venv/bin/activate
```

### 2) Install dependencies (with tests)

```bash

# Recommended for development and testing (includes pytest, coverage, linters)
pip install -e .[dev]

# If you only need runtime dependencies (not recommended for contributors):
# pip install -e .
```

By default, the dependencies declared in `pyproject.toml` are sufficient (they include `torch` and `torch-geometric`).

Only if you enable a data‑loading extension that requires extra PyG ops (e.g., `torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`), do you need to install those extras. The snippet below detects your Torch/CUDA and prints the correct wheel index to use for these optional packages:

```bash
# Inspect Torch / CUDA (optional)
python - <<'PYINFO'
import torch
print('Torch:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
PYINFO

# Compute the PyG wheel index matching your Torch/CUDA
PYG_INDEX=$(python - <<'PYG'
import torch
torch_ver = torch.__version__.split('+')[0]
cuda = torch.version.cuda
if cuda:
    cu_tag = f"cu{cuda.replace('.', '')}"
else:
    cu_tag = 'cpu'
print(f"https://data.pyg.org/whl/torch-{torch_ver}+{cu_tag}.html")
PYG
)
echo "Using PyG wheel index: $PYG_INDEX"

# Install only the extras you need, for example:
# pip install torch-scatter -f "$PYG_INDEX"
# pip install torch-sparse  -f "$PYG_INDEX"
# pip install torch-cluster -f "$PYG_INDEX"
# pip install torch-spline-conv -f "$PYG_INDEX"
```

More details: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

### 3) (Optional) Pre‑commit hooks

```bash
pre-commit install
```

### 4) Run tests

```bash
pytest
```

## Example data

Example data can be downloaded from [this Dropbox folder](https://www.dropbox.com/scl/fo/8xjlsri0zcov20v8xsyxa/AOMpquFinnQnp287lT5hxJM?rlkey=1h1sm3wxd4s1oeygludri6hr6&st=cg0qdhic&dl=0). No Dropbox account is required to access the files.

## Notes

- The project restricts Python versions to 3.10–3.12 as specified in `.python-version` and `pyproject.toml`.
- For more information about the dependencies, see `pyproject.toml`.

Tip: Integration tests optionally use real data. In CI, datasets are downloaded with `python -m tests.dropbox_download "/MMAI25Hackathon" "MMAI25Hackathon" --unzip` when a Dropbox token is configured.

## Authors

- Shuo Zhou (<shuo.zhou@sheffield.ac.uk>)
- Xianyuan Liu (<xianyuan.liu@sheffield.ac.uk>)
- Wenrui Fan (<wenrui.fan@sheffield.ac.uk>)
- Mohammod N. I. Suvon (<m.suvon@sheffield.ac.uk>)
- L. M. Riza Rizky (<l.m.rizky@sheffield.ac.uk>)
