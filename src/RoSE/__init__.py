"""RoSE: Rotary Spatial Embeddings for PyTorch.

This package provides PyTorch implementations of Rotary Spatial Embeddings,
extending 2D Rotary Position Embeddings (RoPE) to incorporate spatial information
in terms of real world coordinates.
"""

from .rose import RoSELayer
from .rose_mha import RoSEMultiheadSelfAttention

__all__ = ["RoSELayer", "RoSEMultiheadSelfAttention"]
