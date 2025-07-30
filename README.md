# RoSE

[![CI/CD Pipeline](https://github.com/rhoadesScholar/RoSE/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rhoadesScholar/RoSE/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/rhoadesScholar/RoSE/branch/main/graph/badge.svg)](https://codecov.io/gh/rhoadesScholar/RoSE)
[![PyPI version](https://badge.fury.io/py/rose-spatial-embeddings.svg)](https://badge.fury.io/py/rose-spatial-embeddings)
[![Python versions](https://img.shields.io/pypi/pyversions/rose-spatial-embeddings.svg)](https://pypi.org/project/rose-spatial-embeddings/)

PyTorch implementation of Rotary Spatial Embeddings

Rotary Spatial Embeddings (RoSE) extends 2D [Rotary Position Embeddings (RoPE)](https://arxiv.org/abs/2403.13298) to incorporate spatial information in terms of real world coordinates into the embeddings. This is particularly useful for tasks that require understanding of spatial relationships across different scales, such as in microscopy.

## Installation

### From PyPI (recommended)

```bash
pip install rose-spatial-embeddings
```

### From source

```bash
pip install git+https://github.com/rhoadesScholar/RoSE.git
```

### Development installation

```bash
git clone https://github.com/rhoadesScholar/RoSE.git
cd RoSE
pip install -e ".[dev]"
```

## Usage

```python
import torch
from RoSE import RoSELayer, RoSEMultiheadSelfAttention

# Basic RoSE layer for applying rotary spatial embeddings to q and k
layer = RoSELayer(dim=128, num_heads=8, spatial_dims=3, learnable=True)

batch_size, seq_len = 2, 1000
q = torch.randn(batch_size, seq_len, 128)
k = torch.randn(batch_size, seq_len, 128)

# Define spatial grid properties
grid_shape = (10, 10, 10)  # 3D grid dimensions
voxel_size = (1.0, 1.0, 1.0)  # Physical size of each voxel

# Apply rotary spatial embeddings
q_rot, k_rot = layer(q, k, grid_shape, voxel_size)

# Complete multihead self-attention with RoSE
mha = RoSEMultiheadSelfAttention(dim=128, num_heads=8, spatial_dims=3)
x = torch.randn(batch_size, seq_len, 128)
output = mha(x, grid_shape, voxel_size)
```

## Development

### Setting up development environment

```bash
# Clone the repository
git clone https://github.com/rhoadesScholar/RoSE.git
cd RoSE

# Install in development mode with all dependencies
make dev-setup

# Or manually:
pip install -e ".[dev]"
pre-commit install
```

### Running tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run tests for specific Python versions using tox
tox -e py39,py310,py311
```

### Code quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run all quality checks
make check-all
```

### Building and releasing

The project uses automatic date-based versioning with the format `YYYY.MM.DD`. Versions are automatically generated based on git tags and commit dates.

```bash
# Build package
make build

# The version will be something like: 2025.01.29 (for today's date)
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.