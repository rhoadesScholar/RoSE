"""RoSE: Rotary Spatial Embeddings for PyTorch.

This package provides PyTorch implementations of Rotary Spatial Embeddings,
extending 2D Rotary Position Embeddings (RoPE) to incorporate spatial information
in terms of real world coordinates.
"""

from .rose import RoSELayer

__all__ = ["RoSELayer"]
