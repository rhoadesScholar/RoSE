"""RoSE: Rotary Spatial Embeddings for PyTorch.

This package provides PyTorch implementations of Rotary Spatial Embeddings,
extending 2D Rotary Position Embeddings (RoPE) to incorporate spatial information
in terms of real world coordinates.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("RoSE")
except PackageNotFoundError:
    __version__ = "0.1.1"

__author__ = "Jeff Rhoades"
__email__ = "rhoadesj@hhmi.org"

from .rose import (
    RoSELayer,
    apply_rotary_emb,
    reshape_for_broadcast,
    init_nd_freqs,
    init_t_nd,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "RoSELayer",
    "apply_rotary_emb",
    "reshape_for_broadcast",
    "init_nd_freqs",
    "init_t_nd",
]
