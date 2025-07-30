# Adapted from https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py
# by @rhoadesScholar 2025

from functools import lru_cache
import math
from typing import Tuple

import torch
import torch.nn as nn


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    leading_dims = x.ndim - freqs_cis.ndim
    # Only run this check in eager mode; skip during TorchScript
    if not torch.jit.is_scripting():
        if not all(xs == fs for xs, fs in zip(x.shape[leading_dims:], freqs_cis.shape)):
            raise ValueError(
                f"Cannot reshape frequency matrix of size {freqs_cis.shape} "
                f"to match token shape of {x.shape[leading_dims:]}"
            )
    shape = [1] * leading_dims + list(freqs_cis.shape)
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


@torch.no_grad()
def init_nd_freqs(
    dim: int,
    num_heads: int,
    spatial_dims: int,
    theta: float = 10.0,
    rotate: bool = True,
    device=None,
    dtype=torch.float32,
):
    """
    Generalized ND version for rotary embedding frequency initialization, using random orthonormal frame
    Returns  [spatial_dims, num_heads, dim // spatial_dims]
    """
    pair = 2 * spatial_dims
    if dim % pair:
        raise ValueError(f"dim must be divisible by 2×spatial_dims = {pair}")
    num_f = dim // pair  # magnitudes per axis
    mag = 1 / (theta ** (torch.arange(0, dim, pair)[:num_f].float() / dim))

    freqs = torch.empty(spatial_dims, num_heads, 2 * num_f, device=device, dtype=dtype)

    for h in range(num_heads):
        if rotate:
            # one orthonormal frame R ∈ SO(D)
            A, _ = torch.linalg.qr(
                torch.randn(spatial_dims, spatial_dims, device=device, dtype=dtype)
            )
            if torch.det(A) < 0:
                A[:, 0].neg_()
        else:
            A = torch.eye(spatial_dims, device=device, dtype=dtype)

        # first two columns of A give cos & sin directions
        freqs[:, h, :num_f] = mag * A[:, 0:1]  # real
        freqs[:, h, num_f:] = mag * A[:, 1:2]  # imag

    return freqs


@lru_cache(maxsize=128)
@torch.no_grad()
def init_t_nd(dims, spacing=(1.0, 1.0), dtype=torch.float32):
    t = torch.arange(int(torch.prod(torch.tensor(dims))), device="cpu")
    idxs = torch.unravel_index(t, dims)
    return tuple((i * s).to(dtype) for i, s in zip(idxs, spacing))


class RoSELayer(nn.Module):
    """Layer to apply RoSE (Rotary Spatial Embeddings) to query and key tensors."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        spatial_dims: int = 2,
        rope_theta=10.0,
        learnable: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.spatial_dims = spatial_dims
        self.rope_theta = rope_theta
        self.learnable = learnable

        freqs = init_nd_freqs(
            dim=self.dim // self.num_heads,
            num_heads=self.num_heads,
            spatial_dims=spatial_dims,
            theta=rope_theta,
            rotate=True,
        ).view(2, -1)
        self.freqs = nn.Parameter(freqs, requires_grad=learnable)

    def compute_cis(self, ts: Tuple[torch.Tensor, torch.Tensor]):
        N = ts[0].shape[0]  # Number of points in the grid
        # No float 16 for this range
        with torch.amp.autocast("cuda", enabled=False):
            freqs = torch.stack(
                [
                    (t.unsqueeze(-1) @ f.unsqueeze(-2))
                    .view(N, self.num_heads, -1)
                    .permute(1, 0, 2)
                    for t, f in zip(ts, self.freqs)
                ],
                dim=-1,
            ).sum(dim=-1)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        spacing: Tuple[float, ...],
        grid_shape: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reshape q and k for multi-head attention:
        # [B, N, C] -> [B, N, num_heads, C // num_heads]
        B, N, C = q.shape
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        ts = init_t_nd(grid_shape, spacing=spacing, dtype=q.dtype)
        ts = tuple(t.to(q.device) for t in ts)
        freqs_cis = self.compute_cis(ts)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        # Reshape back to [B, N, C]
        q = q.permute(0, 2, 1, 3).reshape(B, N, C)
        k = k.permute(0, 2, 1, 3).reshape(B, N, C)
        return q, k


if __name__ == "__main__":
    # Example usage
    spatial_dims = 2
    dim = 64 * 2 * spatial_dims
    layer = RoSELayer(dim=dim, num_heads=8, spatial_dims=spatial_dims)

    # create grid_shape and spacing based on spatial_dims
    grid_shape = tuple([10] * spatial_dims)
    spacing = tuple([1.0] * spatial_dims)

    batch_size, seq_len = 2, math.prod(grid_shape)
    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)

    q_out, k_out = layer(q, k, spacing, grid_shape)
    assert q_out.shape == (
        batch_size,
        seq_len,
        dim,
    ), f"Expected q_out shape {(batch_size, seq_len, dim)}, got {q_out.shape}"
    assert k_out.shape == (
        batch_size,
        seq_len,
        dim,
    ), f"Expected k_out shape {(batch_size, seq_len, dim)}, got {k_out.shape}"
