import math
from typing import Tuple

import torch
from torch import Tensor, nn

from .rose import _RotationCache, _make_base_frequencies, _rope_apply

_ROSE_CACHE = _RotationCache()


class RoSEMultiheadSelfAttention(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention or ViT MSA block
    that applies **Rotary Spatial Embeddings** to q & k.

    Args
    ----
    dim : total embedding dim (must be divisible by num_heads)
    num_heads : number of attention heads
    spatial_dims : 1, 2, or 3
    learnable : if True, θ spectrum is learnable per head
    """

    def __init__(
        self, dim: int, num_heads: int, spatial_dims: int = 3, learnable: bool = True
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim_half = self.head_dim // 2
        self.spatial_dims = spatial_dims

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # θ frequencies  (H, D/2)
        self.theta = _make_base_frequencies(
            self.dim_half,
            learnable,
            num_heads,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def forward(
        self,
        x: Tensor,  # (B, N, C)
        grid_shape: Tuple[int, ...],
        voxel_size: Tuple[float, ...],
    ) -> Tensor:
        assert (
            len(grid_shape) == self.spatial_dims
        ), f"grid_shape must have {self.spatial_dims} dims"
        B, N, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)  # 3,B,H,N,D
        q, k, v = qkv[0], qkv[1], qkv[2]  # each B,H,N,D

        # ---  retrieve / build trig tables
        cos, sin = _ROSE_CACHE.get(
            grid_shape, voxel_size, self.theta, dtype=q.dtype
        )  # (N,1,1,D/2)

        # cos/sin: broadcast to (N, B, H, D/2) by unsqueezing properly
        cos = cos.to(q.device)
        sin = sin.to(q.device)

        q = _rope_apply(q, cos, sin)
        k = _rope_apply(k, cos, sin)

        # --- attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        attn = attn.softmax(dim=-1)

        out = attn @ v  # B, H, N, D
        out = out.transpose(1, 2).reshape(B, N, self.dim)
        return self.proj(out)
