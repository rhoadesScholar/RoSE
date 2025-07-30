"""RoSE: Rotary Spatial Embeddings for PyTorch.

This package provides PyTorch implementations of Rotary Spatial Embeddings,
extending 2D Rotary Position Embeddings (RoPE) to incorporate spatial information
in terms of real world coordinates.
"""

from .rose import (
    RoSELayer,
    apply_rotary_emb,
    reshape_for_broadcast,
    init_nd_freqs,
    init_t_nd,
)

__all__ = [
    "RoSELayer",
    "apply_rotary_emb",
    "reshape_for_broadcast",
    "init_nd_freqs",
    "init_t_nd",
]

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("rose-spatial-embeddings")
except PackageNotFoundError:
    __version__ = "0.0.0"
