# Adapted from https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py
# by @rhoadesScholar 2025

from functools import lru_cache
from typing import Tuple

import torch
import torch.nn as nn


# def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
#     leading_dims = x.ndim - freqs_cis.ndim
#     shape = [1] * leading_dims + list(freqs_cis.shape)
#     return freqs_cis.view(*shape)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
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
        ts = init_t_nd(grid_shape, spacing=spacing, dtype=q.dtype)
        for t in ts:
            t = t.to(q.device)
        freqs_cis = self.compute_cis(ts)

        return apply_rotary_emb(q, k, freqs_cis=freqs_cis)
