# RoSE: Rotary Spatial Embeddings for PyTorch

RoSE is a Python library that provides PyTorch implementations of Rotary Spatial Embeddings, extending 2D Rotary Position Embeddings (RoPE) to incorporate spatial information with real-world coordinates.

**CRITICAL**: Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Without Full Installation

If package installation fails due to network issues, you can still work with the codebase:

### Syntax and Structure Validation
```bash
# Check Python syntax
python -m py_compile src/RoSE/rose.py
python -m py_compile src/RoSE/__init__.py

# Verify file structure
ls -la src/RoSE/           # Should show __init__.py (681 bytes) and rose.py (12706 bytes)
wc -l src/RoSE/*.py       # __init__.py: 26 lines, rose.py: 333 lines
```

### Code Analysis Without Dependencies
```bash
# Check imports and basic structure
grep -n "^import\|^from" src/RoSE/rose.py
grep -n "class\|def" src/RoSE/rose.py | head -10

# Examine test structure 
ls -la tests/              # Should show test_rose.py (67 lines), test_rose_numerical.py (1166 lines)
```

### Manual Code Review
- **Core implementation**: `src/RoSE/rose.py` contains `RotarySpatialEmbedding` class (lines 46-333)
- **Public API**: `src/RoSE/__init__.py` exports `RoSEMultiHeadAttention` and `RotarySpatialEmbedding`
- **Test coverage**: Comprehensive numerical tests in `tests/test_rose_numerical.py`

## Working Effectively

### Environment Setup
- **Python Requirements**: Python 3.10+ (tested with 3.10, 3.11, 3.12)
- **Primary Dependency**: PyTorch >= 1.9.0

### Installation Process
1. **Install package in development mode:**
   ```bash
   pip install -e .
   ```
   - **Expected time**: 30-60 seconds for core package
   - **Note**: May fail due to network/firewall limitations accessing PyPI

2. **Install with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```
   - **Expected time**: 2-5 minutes (depending on PyTorch installation)
   - **NEVER CANCEL**: PyTorch download can take 3-5 minutes. Set timeout to 10+ minutes.
   - **Alternative**: Use `make install-dev` (equivalent command)

3. **Network Issues Fallback:**
   ```bash
   # If pip install fails due to network timeouts:
   pip install --timeout 600 -e ".[dev]"
   # Or try installing components separately:
   pip install --timeout 600 torch
   pip install --timeout 600 pytest black isort flake8 mypy
   pip install -e .
   ```
   - **Known Issue**: PyPI connectivity may timeout in restricted environments
   - **Alternative**: Use conda or system package managers if pip fails repeatedly

### Build and Testing

1. **Clean build artifacts:**
   ```bash
   make clean
   ```
   - **Expected time**: < 1 second

2. **Build package:**
   ```bash
   make build
   ```
   - **Expected time**: 10-30 seconds
   - **NEVER CANCEL**: Set timeout to 2+ minutes for first build

3. **Run tests:**
   ```bash
   make test
   ```
   - **Expected time**: 5-15 seconds (fast tests)
   - **Command**: Equivalent to `pytest tests/`

4. **Run tests with coverage:**
   ```bash
   make test-cov
   ```
   - **Expected time**: 15-30 seconds
   - **NEVER CANCEL**: Coverage analysis can take time. Set timeout to 5+ minutes.
   - **Output**: Generates `htmlcov/`, `coverage.xml`, and terminal coverage report

5. **Run fast tests only:**
   ```bash
   make test-fast
   ```
   - **Expected time**: 3-10 seconds
   - **Command**: Equivalent to `pytest tests/ -x -v` (stops on first failure)

### Code Quality and Validation

1. **Linting:**
   ```bash
   make lint
   ```
   - **Expected time**: 2-5 seconds
   - **Command**: Runs `flake8 src tests`

2. **Code formatting:**
   ```bash
   make format
   ```
   - **Expected time**: 1-3 seconds
   - **Commands**: Runs `black src tests` and `isort src tests`

3. **Type checking:**
   ```bash
   make type-check
   ```
   - **Expected time**: 3-8 seconds
   - **Command**: Runs `mypy src`

4. **Run all checks (recommended before commits):**
   ```bash
   make check-all
   ```
   - **Expected time**: 20-45 seconds
   - **NEVER CANCEL**: Comprehensive check suite. Set timeout to 2+ minutes.
   - **Commands**: Runs lint, type-check, and test-cov in sequence

5. **Pre-commit hooks:**
   ```bash
   make pre-commit-install  # Install hooks
   make pre-commit         # Run all hooks
   ```
   - **Expected time**: 5-15 seconds for pre-commit run

## Validation Scenarios

**CRITICAL**: Always run these validation steps after making changes:

### 1. Basic Functionality Test
```python
# Test that imports work (requires package installation)
python -c "from RoSE import RoSEMultiHeadAttention, RotarySpatialEmbedding; print('Import successful')"
```
- **Expected output**: "Import successful"
- **Fallback if package not installed**: Check file structure with `ls -la src/RoSE/`

### 2. Core Mathematical Properties Test
```bash
# Run numerical validation tests
pytest tests/test_rose_numerical.py::TestRoSENumericalProperties::test_phase_conjugate_property -v
```
- **Expected time**: 2-5 seconds
- **Purpose**: Validates core mathematical properties of rotary embeddings

### 3. Multi-dimensional Spatial Test
```bash
# Test 2D and 3D spatial embeddings
pytest tests/test_rose.py::TestRoSEMultiHeadAttention::test_rose_layer_forward -v
```
- **Expected time**: 1-3 seconds
- **Purpose**: Ensures spatial embedding works correctly across dimensions

### 4. End-to-end Integration Test
```python
# Create and test a complete workflow
python -c "
import torch
from RoSE import RoSEMultiHeadAttention

# 2D spatial embedding test
layer = RoSEMultiHeadAttention(dim=64, num_heads=8, spatial_dims=2)
grid_shape = (10, 10)
voxel_size = (1.0, 1.0)
batch_size, seq_len = 2, 100

q = torch.randn(batch_size, seq_len, 64)
k = torch.randn(batch_size, seq_len, 64)

attn = layer(q, k, voxel_size, grid_shape)
print(f'Success: Output shape {attn.shape}')
assert attn.shape == (batch_size, 8, seq_len, seq_len)
"
```

## Repository Structure

### Key Files and Directories
```
├── src/RoSE/              # Main package source
│   ├── __init__.py        # Package exports (27 lines)
│   └── rose.py            # Core implementation (333 lines)
├── tests/                 # Test suite
│   ├── test_rose.py       # Basic functionality tests (67 lines)
│   └── test_rose_numerical.py  # Mathematical property tests (1166 lines)
├── .github/
│   └── workflows/
│       └── ci-cd.yml      # CI/CD pipeline
├── pyproject.toml         # Build configuration and dependencies
├── Makefile              # Development workflow commands
├── .pre-commit-config.yaml # Pre-commit hooks
└── tox.ini               # Multi-environment testing
```

### Important Configuration Files
- **pyproject.toml**: All project configuration (build, dependencies, tools)
- **setup.cfg**: flake8 configuration
- **Makefile**: Development workflow automation
- **CITATION.cff**: Academic citation information

## CI/CD Pipeline Compatibility

### GitHub Actions Workflow
The repository uses `.github/workflows/ci-cd.yml` which:
1. **Format job**: Runs linting and formatting checks
2. **Test job**: Multi-platform testing (Ubuntu, Windows, macOS) × Python (3.10, 3.11, 3.12)
3. **Tag-release job**: Automatic versioning and tagging
4. **Publish job**: PyPI publication

### Required CI Commands
Always run these before pushing to ensure CI success:
```bash
make lint      # Must pass
make format    # Must pass  
make test-cov  # Must pass
```

## Common Development Tasks

### Making Changes to Core Library
1. Edit `src/RoSE/rose.py` (main implementation)
2. Always run validation after changes:
   ```bash
   make check-all
   ```
3. Add tests in `tests/test_rose_numerical.py` for mathematical properties
4. Add basic tests in `tests/test_rose.py` for API functionality

### Adding New Features
1. Follow existing patterns in `rose.py`
2. Add comprehensive tests covering edge cases
3. Update `__init__.py` exports if adding public APIs
4. Run full validation suite:
   ```bash
   make clean && make check-all
   ```

### Debugging Test Failures
- **Fast iteration**: Use `make test-fast` to stop on first failure
- **Specific test**: Use `pytest tests/test_rose.py::TestClassName::test_method -v`
- **Coverage gaps**: Check `htmlcov/index.html` after `make test-cov`

## Known Issues and Limitations

### Known Issues and Limitations

### Network Connectivity
- **PyPI timeouts**: Package installation frequently fails due to read timeouts from pypi.org
- **Error pattern**: `ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out`
- **Workarounds**: 
  - Use `--timeout 600` flag with pip commands
  - Try installing dependencies individually
  - Use conda or system package managers as alternatives
- **Testing limitation**: Some validation scenarios require PyTorch installation
- **Fallback validation**: Use file structure checks and syntax validation when packages unavailable

### Platform-Specific Notes
- **Windows**: All commands work through Makefile
- **macOS**: Requires Xcode command line tools for some dependencies
- **Linux**: Most reliable platform for development

### Performance Considerations
- **Large tensors**: Memory usage scales with `batch_size × seq_len²` 
- **Gradient computation**: Enable only when training (`requires_grad=True`)
- **Multi-head attention**: Memory scales with `num_heads`

## Version Information
- **Current setup**: Date-based versioning (YYYY.M.D.HHMM format)
- **Version location**: Automatically managed in `CITATION.cff`
- **Pre-commit hook**: Updates version on commits to main branch

## Quick Reference Commands

```bash
# Essential workflow
make install-dev     # Setup environment
make check-all      # Full validation
make clean          # Clean artifacts

# Development iteration  
make format         # Fix formatting
make lint          # Check style
make test-fast     # Quick test

# Release preparation
make test-cov      # Full test with coverage
make build         # Build package
```

**Remember**: NEVER CANCEL long-running operations. Set appropriate timeouts (2+ minutes for builds, 5+ minutes for full test suites) and wait for completion.