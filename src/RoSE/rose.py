# Adapted from https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py
# by @rhoadesScholar 2025

from functools import lru_cache
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def _make_log_spaced_frequencies(
    num_planes: int,
    spatial_dims: int,
    base_theta: float = 1e4,
) -> torch.Tensor:
    """
    Roughly imitates the original RoPE frequency schedule:
        ω_k = 1 / (base_theta ** (2k / D))
    but returns a tensor shaped (num_planes, spatial_dims).
    Every spatial axis shares the same 1D schedule so that
    the dot-product 'coords ⋅ ω' has a clean log scale.
    """
    exponents = torch.arange(num_planes, dtype=torch.float32) * 2.0 / (num_planes * 2)
    freqs_1d = 1.0 / (base_theta**exponents)  # (num_planes,)
    freqs = freqs_1d.unsqueeze(-1).repeat(1, spatial_dims)  # (num_planes, d)
    return freqs


@lru_cache(maxsize=128)
@torch.no_grad()
def _init_p_nd(
    dims: Tuple[int, ...],
    spacing: Tuple[float, ...] = (1.0, 1.0),
    dtype=torch.float32,
    device="cpu",
) -> torch.Tensor:
    p = torch.arange(int(torch.prod(torch.tensor(dims))), device=device)
    idxs = torch.stack(torch.unravel_index(p, dims), dim=-1).to(
        dtype=dtype, device=device
    )  # (N, spatial_dims)
    return idxs * torch.tensor(
        spacing, device=device, dtype=dtype, requires_grad=False
    ).view(1, -1)


class RotarySpatialEmbedding(nn.Module):
    """
    Rotates input embeddings in 2-D sub-planes, independently per attention group (head).

    Given:
        x: Tensor of shape (B, N, D)   – input embeddings (D must be even)
    Returns:
        x_rot: Tensor of shape (B, N, D) – rotated embeddings

    Definitions:
      • H    = num_heads
      • P    = D // 2               – number of 2-D sub-planes
      • pos_n, pos_m ∈ ℝᵈ          – coordinates of tokens n and m
      • ω_{h,p} ∈ ℝᵈ              – frequency vector for head h, plane p
      • cis(θ) = cos(θ) + i·sin(θ)

    Angle computation for head h, plane p at position t:
        θ_{h,p}(t) = ⟨pos_t, ω_{h,p}⟩

    Attention matrix A[b,h,n,m]:
        A[b,h,n,m] = Re ⎡
            ∑_{p=0}^{P-1}
              q[b,h,n,p] · conj(k[b,h,m,p]) · cis( ⟨pos_n − pos_m, ω_{h,p}⟩ )
        ⎤
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        spatial_dims: int = 2,
        base_theta: float = 1e4,
        learnable: bool = True,
        init_jitter_std: float = 0.02,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.spatial_dims = spatial_dims
        self.rope_theta = base_theta ** (1 / spatial_dims)
        self.learnable = learnable

        num_planes = dim // 2
        freqs = _make_log_spaced_frequencies(
            num_planes, spatial_dims, self.rope_theta
        )  # (num_planes, spatial_dims)
        freqs = freqs.reshape(num_heads, -1, spatial_dims)  # (H, P, spatial_dims)

        if learnable:
            if init_jitter_std > 0:
                eps = torch.randn_like(freqs) * init_jitter_std
                freqs = torch.exp(freqs.log() + eps)
            self.freqs = nn.Parameter(freqs)  # learned per head/plane
        else:
            self.register_buffer("freqs", freqs, persistent=False)

    def _get_complex_split(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(
            *x.shape[:-1], self.num_heads, self.dim // (2 * self.num_heads), 2
        )  # (B,T,H,P,2)
        return torch.view_as_complex(x)  # (B,T,H,P)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, D)
        spacing: Tuple[float, ...],
        grid_shape: Optional[Tuple[int, ...]] = None,
        flatten: bool = True,
    ) -> torch.Tensor:
        # Get position tensor
        pos = _init_p_nd(
            grid_shape, spacing=spacing, dtype=x.dtype, device=x.device
        )  # (N, spatial_dims)

        # Get phase vectors
        ph_x = torch.einsum("td,hpd->thp", pos, self.freqs)  # (N, H, P)

        ph_x = ph_x.cos() + 1j * ph_x.sin()  # (N, H, P)

        # Split x into real and imaginary parts, per head/plane
        x = self._get_complex_split(x)  # (B, N, H, P)

        # Apply rotary embeddings
        x = x * ph_x.unsqueeze(0)  # (B, N, H, P)

        if flatten:
            # Reshape back to original dimensions
            x = torch.view_as_real(x).flatten(2)  # (B, N, D)

        return x


class RoSEMultiHeadAttention(nn.Module):
    """
    Rotates key and query embeddings in 2-D sub-planes, then computes multi-head attention.

    Given:
        q: Tensor of shape (B, N, D)   – query embeddings (D must be even)
        k: Tensor of shape (B, M, D)   – key embeddings (D must be even)
    Returns:
        attn: Tensor of shape (B, H, N, M) – attention scores

    Definitions:
      • H    = num_heads
      • P    = D // 2               – number of 2-D sub-planes
      • pos_n, pos_m ∈ ℝᵈ          – coordinates of tokens n and m
      • ω_{h,p} ∈ ℝᵈ              – frequency vector for head h, plane p
      • cis(θ) = cos(θ) + i·sin(θ)

    Angle computation for head h, plane p at position t:
        θ_{h,p}(t) = ⟨pos_t, ω_{h,p}⟩

    Attention matrix A[b,h,n,m]:
        A[b,h,n,m] = Re ⎡
            ∑_{p=0}^{P-1}
              q[b,h,n,p] · conj(k[b,h,m,p]) · cis( ⟨pos_n − pos_m, ω_{h,p}⟩ )
        ⎤
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        spatial_dims: int = 2,
        base_theta: float = 1e4,
        learnable: bool = True,
        init_jitter_std: float = 0.02,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.spatial_dims = spatial_dims
        self.rope_theta = base_theta ** (1 / spatial_dims)
        self.learnable = learnable
        self.init_jitter_std = init_jitter_std

        pe_kwargs = {
            "dim": dim,
            "num_heads": num_heads,
            "spatial_dims": spatial_dims,
            "base_theta": base_theta,
            "learnable": learnable,
            "init_jitter_std": init_jitter_std,
        }
        self.q_pe = RotarySpatialEmbedding(**pe_kwargs)
        self.k_pe = RotarySpatialEmbedding(**pe_kwargs)

    def forward(
        self,
        q: torch.Tensor,  # (B, N_q, D)
        k: torch.Tensor,  # (B, N_k, D)
        q_spacing: Tuple[float, ...],
        q_grid_shape: Tuple[int, ...],
        k_spacing: Optional[Tuple[float, ...]] = None,
        k_grid_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        if k_spacing is None:
            k_spacing = q_spacing
        if k_grid_shape is None:
            k_grid_shape = q_grid_shape

        # Rotate embeddings
        q = self.q_pe(q, q_spacing, q_grid_shape, flatten=False)  # (B, N_q, D)
        k = self.k_pe(k, k_spacing, k_grid_shape, flatten=False)  # (B, N_k, D)

        # Compute attention scores
        attn = torch.einsum("bnhp,bmhp->bhnm", q, k)  # (B, H, N_q, N_k)

        return attn


if __name__ == "__main__":
    # Example usage
    num_heads = 8
    spatial_dims = 3
    dim = 16 * 2 * spatial_dims * num_heads  # Ensure dim is divisible by num_heads
    layer = RoSEMultiHeadAttention(
        dim=dim, num_heads=num_heads, spatial_dims=spatial_dims
    )

    # create grid_shape and spacing based on spatial_dims
    grid_shape = tuple([10] * spatial_dims)
    spacing = tuple([1.0] * spatial_dims)
    batch_size, seq_len = 2, math.prod(grid_shape)

    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)

    attn = layer(q, k, spacing, grid_shape)
    assert attn.shape == (batch_size, num_heads, seq_len, seq_len)
