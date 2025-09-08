# Adapted from https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py
# by @rhoadesScholar 2025

from functools import lru_cache
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .utils import to_tuple


@torch.no_grad()
def _make_log_spaced_frequencies(
    num_planes: int,
    spatial_dims: int,
    base_theta: float = 1e4,
) -> torch.Tensor:
    """
    Generate log-spaced frequencies ω_p in R^{spatial_dims} for p=0..num_planes-1.
    Frequencies only depend on plane index, so translation cancels out.
    """
    exponents = torch.arange(num_planes, dtype=torch.float32) * 2.0 / (num_planes * 2)
    log_freqs = -exponents * math.log(base_theta)
    freqs_1d = torch.exp(log_freqs)  # shape (num_planes,)
    freqs = freqs_1d.unsqueeze(-1).repeat(1, spatial_dims)
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
    Supports partial rotation where only a subset of the embedding dimension is rotated.

    Given:
        x: Tensor of shape (B, N, D)   – input embeddings (D must be even)
    Returns:
        x_rot: Tensor of shape (B, N, D) – rotated embeddings

    Definitions:
      • H    = num_heads
      • P    = rotary_dim // 2          – number of 2-D sub-planes for rotation
      • pos_n, pos_m ∈ ℝᵈ              – coordinates of tokens n and m
      • ω_{h,p} ∈ ℝᵈ                  – frequency vector for head h, plane p
      • cis(θ) = cos(θ) + i·sin(θ)

    Angle computation for head h, plane p at position t:
        θ_{h,p}(t) = ⟨pos_t, ω_{h,p}⟩

    Attention matrix A[b,h,n,m]:
        A[b,h,n,m] = Re ⎡
            ∑_{p=0}^{P-1}
              q[b,h,n,p] · conj(k[b,h,m,p]) · cis( ⟨pos_n − pos_m, ω_{h,p}⟩ )
        ⎤

    Args:
        feature_dims: Total embedding dimension (must be even and divisible by num_heads)
        num_heads: Number of attention heads
        spatial_dims: Number of spatial dimensions (2 for 2D, 3 for 3D, etc.)
        base_theta: Base frequency for rotary embeddings
        learnable: Whether frequencies should be learnable parameters
        init_jitter_std: Standard deviation for frequency initialization jitter
        rotary_ratio: Fraction of embedding dimension to apply rotation to (0.0 to 1.0)
                     When < 1.0, only the first rotary_ratio * dim dimensions are rotated,
                     the rest are passed through unchanged.
        frequency_scaling: Scaling strategy for frequencies ("none", "linear", "sqrt", "adaptive")
    """

    def __init__(
        self,
        feature_dims: int,
        num_heads: int,
        spatial_dims: int = 2,
        base_theta: float = 1e4,
        learnable: bool = True,
        init_jitter_std: float = 0.02,
        frequency_scaling: str = "sqrt",
        rotary_ratio: float = 1.0,
    ):
        super().__init__()
        assert feature_dims % num_heads == 0, "dim must be divisible by num_heads"
        assert (
            feature_dims // num_heads % 2 == 0
        ), "dims_per_head must be even for complex representation"
        assert 0.0 <= rotary_ratio <= 1.0, "rotary_ratio must be between 0.0 and 1.0"

        self.feature_dims = feature_dims
        self.num_heads = num_heads
        self.dims_per_head = feature_dims // num_heads
        self.spatial_dims = spatial_dims
        self.rotary_ratio = rotary_ratio
        self.learnable = learnable

        # Calculate dimensions for rotary embedding
        self.rotary_dim = int(self.feature_dims * rotary_ratio)

        # Ensure rotary_dim is even and divisible by num_heads
        self.rotary_dim = (self.rotary_dim // self.dims_per_head) * self.dims_per_head
        self.rotary_dim = (self.rotary_dim // 2) * 2

        # Ensure rotary_dim is divisible by num_heads
        self.non_rotary_dim = self.feature_dims - self.rotary_dim
        assert (
            self.non_rotary_dim % self.dims_per_head == 0
        ), "non_rotary_dim must be divisible by dims_per_head. suggest changing rotary_ratio"

        if self.rotary_dim > 0:
            # Calculate total rotary heads
            self.num_rotary_heads = int(self.rotary_dim // self.dims_per_head)

            # Only create frequencies for the rotary portion
            self.num_rotary_planes = self.dims_per_head // 2

            if frequency_scaling == "none":
                self.rose_theta = base_theta
            elif frequency_scaling == "linear":
                self.rose_theta = base_theta ** (1 / spatial_dims)
            elif frequency_scaling == "sqrt":
                self.rose_theta = base_theta ** (1 / math.sqrt(spatial_dims))
            elif frequency_scaling == "adaptive":
                self.rose_theta = base_theta ** (2.0 / (spatial_dims * feature_dims))

            freqs = _make_log_spaced_frequencies(
                self.num_rotary_planes * self.num_rotary_heads,
                spatial_dims,
                self.rose_theta,
            )  # (num_planes, spatial_dims)
            freqs = freqs.reshape(
                self.num_rotary_heads, -1, spatial_dims
            )  # (H, P, spatial_dims)

            if learnable:
                if init_jitter_std > 0:
                    eps = torch.randn_like(freqs) * init_jitter_std
                    freqs = torch.exp(torch.clamp(freqs.log() + eps, min=-10, max=10))
                self.freqs = nn.Parameter(
                    freqs, requires_grad=True
                )  # learned per head/plane
            else:
                self.register_buffer("freqs", freqs, persistent=False)

    def _get_complex_split(self, rotary_x: torch.Tensor) -> torch.Tensor:
        # rotary_x shape: (B, N, H, dims_per_head)
        rotary_x = rotary_x.view(
            *rotary_x.shape[:2],
            self.num_rotary_heads,
            self.num_rotary_planes,
            2,
        )  # (B,N,H,P,2)
        return torch.view_as_complex(rotary_x)  # (B,N,H,P)

    def ensure_input_shape(self, x: torch.Tensor):
        # If input is not already reshaped into heads, do it now
        if x.dim() == 3:
            B, N, D = x.shape
            assert (
                D == self.feature_dims
                or D == self.rotary_dim
                or D == self.non_rotary_dim
            )
            x = x.reshape(B, N, -1, self.dims_per_head)
        elif x.dim() == 4:
            B, N, H, D_head = x.shape
            if (
                H != self.num_heads
                and H != self.num_rotary_heads
                and H != (self.num_rotary_heads - self.num_heads)
            ):
                x = x.transpose(1, 2)
                B, N, H, D_head = x.shape
            assert (
                D_head == self.dims_per_head
            ), f"Expected {self.dims_per_head}, got {D_head}"
        return x

    def forward(
        self,
        x: torch.Tensor,  # (B, N, D)
        spacing: Tuple[float, ...],
        grid_shape: Optional[Tuple[int, ...]] = None,
        flatten: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the RoSE layer.

        Rotates the input tensor and applies the RoSE mechanism. By default, the input is reshaped and returned split by heads, as [B, N, H, dims_per_head]. Pass `flatten=True` to return the output as [B, N, D].
        """
        if self.rotary_dim == 0:
            x = self.ensure_input_shape(x)  # --> [B, N, H, dims_per_head]
            # No rotation, return input unchanged
            if flatten:
                return x.flatten(2)
            else:
                return x

        # Split input into rotary and non-rotary parts
        if x.dim() == 3:
            rotary_x = x[..., : self.rotary_dim]  # (B, N, rotary_dim)
            non_rotary_x = x[..., self.rotary_dim :]  # (B, N, non_rotary_dim)
        elif x.dim() == 4:
            # (B, N, rotary_heads, dims_per_head)
            rotary_x = x[..., : self.num_rotary_heads, :]
            # (B, N, non_rotary_heads, dims_per_head)
            non_rotary_x = x[..., self.num_rotary_heads :, :]
        else:
            raise ValueError(
                f"Unsupported input shape. Expected 3D or 4D tensor of shape [B, N, D] or [B, N, H, D_head], got {x.dim()}D."
            )

        # --> [B, N, rotary_heads, dims_per_head]
        rotary_x = self.ensure_input_shape(rotary_x)
        if self.rotary_dim < self.feature_dims:
            non_rotary_x = self.ensure_input_shape(non_rotary_x)
            # non_rotary_x shape: (B, N, non_rotary_heads, dims_per_head)

        # Split rotary part into real and imaginary parts, per head/plane
        rotary_x = self._get_complex_split(rotary_x)  # (B, N, rotary_heads, P)

        # Get position tensor
        # (N, spatial_dims)
        pos = _init_p_nd(grid_shape, spacing=spacing, dtype=x.dtype, device=x.device)

        # Get phase vectors
        ph_x = torch.einsum("td,hpd->thp", pos, self.freqs)  # (N, rotary_heads, P)

        ph_x = ph_x.cos() + 1j * ph_x.sin()  # (N, rotary_heads, P)

        # Apply rotary embeddings
        # (B, N, rotary_heads, P)
        rotary_x = rotary_x * ph_x.unsqueeze(0)

        # Get real part of rotated data
        # (B, N, rotary_heads, dims_per_head)
        rotary_x = torch.view_as_real(rotary_x).flatten(-2)

        if flatten:
            # Reshape rotary part back to original dimensions
            rotary_result = torch.flatten(rotary_x, 2)  # (B, N, total_rotary_dim)

            if self.non_rotary_dim > 0:
                non_rotary_x = torch.flatten(non_rotary_x, 2)

                # Concatenate rotary and non-rotary parts
                return torch.cat([rotary_result, non_rotary_x], dim=-1)  # (B, N, D)
            else:
                return rotary_result
        else:
            # Handle non-flattened case
            if self.non_rotary_dim > 0:
                # (B, N, H, dims_per_head)
                return torch.cat([rotary_x, non_rotary_x], dim=-2)
            else:
                return rotary_x


class RoSEMultiHeadCrossAttention(nn.Module):
    """
    Rotates key and query embeddings in 2-D sub-planes, then computes multi-head attention scores.
    Supports partial rotation where only a subset of the embedding dimension is rotated.

    Given:
        q: Tensor of shape (B, N, D)   – query embeddings (D must be even)
        k: Tensor of shape (B, M, D)   – key embeddings (D must be even)
    Returns:
        attn: Tensor of shape (B, H, N, M) – attention scores

    Definitions:
      • H    = num_heads
      • P    = rotary_dim // 2          – number of 2-D sub-planes for rotation
      • pos_n, pos_m ∈ ℝᵈ              – coordinates of tokens n and m
      • ω_{h,p} ∈ ℝᵈ                  – frequency vector for head h, plane p
      • cis(θ) = cos(θ) + i·sin(θ)

    Angle computation for head h, plane p at position t:
        θ_{h,p}(t) = ⟨pos_t, ω_{h,p}⟩

    Attention matrix A[b,h,n,m]:
        A[b,h,n,m] = Re ⎡
            ∑_{p=0}^{P-1}
              q[b,h,n,p] · conj(k[b,h,m,p]) · cis( ⟨pos_n − pos_m, ω_{h,p}⟩ )
        ⎤

    Args:
        feature_dims: Total embedding dimension (must be even and divisible by num_heads)
        num_heads: Number of attention heads
        spatial_dims: Number of spatial dimensions (2 for 2D, 3 for 3D, etc.)
        base_theta: Base frequency for rotary embeddings
        learnable: Whether frequencies should be learnable parameters
        init_jitter_std: Standard deviation for frequency initialization jitter
        rotary_ratio: Fraction of embedding dimension to apply rotation to (0.0 to 1.0)
        frequency_scaling: Frequency scaling method ("sqrt", "linear", etc.)
    """

    def __init__(
        self,
        feature_dims: int,
        num_heads: int,
        spatial_dims: int = 2,
        base_theta: float = 1e4,
        learnable: bool = True,
        init_jitter_std: float = 0.02,
        rotary_ratio: float = 1.0,
        frequency_scaling: str = "sqrt",
    ):
        super().__init__()
        assert feature_dims % num_heads == 0, "dim must be divisible by num_heads"
        assert feature_dims % 2 == 0, "dim must be even for complex representation"
        self.feature_dims = feature_dims
        self.num_heads = num_heads
        self.spatial_dims = spatial_dims
        self.rope_theta = base_theta ** (1 / spatial_dims)
        self.learnable = learnable
        self.init_jitter_std = init_jitter_std

        pe_kwargs = {
            "feature_dims": feature_dims,
            "num_heads": num_heads,
            "spatial_dims": spatial_dims,
            "base_theta": base_theta,
            "learnable": learnable,
            "init_jitter_std": init_jitter_std,
            "rotary_ratio": rotary_ratio,
            "frequency_scaling": frequency_scaling,
        }
        self.rose = RotarySpatialEmbedding(**pe_kwargs)

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
        q = self.rose(q, q_spacing, q_grid_shape)  # (B, N_q, H, dims_per_head)
        k = self.rose(k, k_spacing, k_grid_shape)  # (B, N_k, H, dims_per_head)

        # Compute attention scores
        attn = torch.einsum("bnhp,bmhp->bhnm", q, k)  # (B, H, N_q, N_k)

        return attn


class MultiRes_RoSE_Block(nn.Module):
    """
    Multi-Resolution RoSE Transformer Block

    A complete transformer block that applies RoSE attention at multiple resolutions,
    followed by feedforward layers. Includes modern components like:
    - Pre-normalization (PreNorm)
    - Residual connections
    - Attention dropout
    - Stochastic depth (DropPath)
    - Scaled dot-product attention with RoSE positional embeddings

    NOTE: Requires `timm` to be installed (`pip install timm`)
    """

    def __init__(
        self,
        feature_dims: int,
        num_heads: int,
        spatial_dims: int = 2,
        base_theta: float = 1e4,
        learnable: bool = True,
        init_jitter_std: float = 0.02,
        rotary_ratio: float = 1.0,
        frequency_scaling: str = "sqrt",
        mlp_ratio: float = 4.0,
        mlp_dropout: float = 0.0,
        mlp_bias: bool = True,
        mlp_activation: torch.nn.Module = torch.nn.GELU,  # type: ignore
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
        norm_eps: float = 1e-6,
    ):
        try:
            from timm.layers import DropPath, Mlp  # type: ignore
        except ImportError:
            raise ImportError(
                "timm is not installed. Please install it with `pip install timm`."
            )
        super().__init__()

        self.feature_dims = feature_dims
        self.num_heads = num_heads
        self.spatial_dims = spatial_dims
        self.head_dim = feature_dims // num_heads
        self.scale = self.head_dim**-0.5

        assert (
            feature_dims % num_heads == 0
        ), f"dim {feature_dims} must be divisible by num_heads {num_heads}"

        # RoSE positional embedding
        self.rose = RotarySpatialEmbedding(
            feature_dims=feature_dims,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            base_theta=base_theta,
            learnable=learnable,
            init_jitter_std=init_jitter_std,
            rotary_ratio=rotary_ratio,
            frequency_scaling=frequency_scaling,
        )

        # Attention components
        self.norm1 = norm_layer(feature_dims, eps=norm_eps)
        self.qkv = nn.Linear(feature_dims, feature_dims * 3, bias=qkv_bias)
        self.attn_dropout = attn_dropout
        self.proj = nn.Linear(feature_dims, feature_dims)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # MLP components
        self.norm2 = norm_layer(feature_dims, eps=norm_eps)
        self.mlp = Mlp(
            in_features=feature_dims,
            hidden_features=int(feature_dims * mlp_ratio),
            out_features=feature_dims,
            drop=mlp_dropout,
            bias=mlp_bias,
            act_layer=mlp_activation,  # type: ignore
        )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _prepare_inputs(
        self, x, input_spacing, input_grid_shape, leading_tokens
    ) -> Tuple[
        torch.Tensor, Sequence[Tuple[float, ...]], Sequence[Tuple[int, ...]], int
    ]:
        """Prepare inputs for multi-resolution processing."""
        if isinstance(x, torch.Tensor):
            x = [x]
        assert isinstance(
            x, Sequence
        ), "Input x must be a tensor or a sequence of tensors"
        x = list(x)  # Ensure x is a list of tensors
        num_ims = len(x) - leading_tokens
        if leading_tokens:
            leading_tokens = x[0].shape[1]

        input_spacing = to_tuple(input_spacing, self.spatial_dims)
        if not isinstance(input_spacing[0], Sequence):
            input_spacing = [input_spacing]

        input_grid_shape = to_tuple(input_grid_shape, self.spatial_dims)
        if not isinstance(input_grid_shape[0], Sequence):
            input_grid_shape = [input_grid_shape]

        assert (
            len(input_spacing) == num_ims
        ), "Spacing length must match number of input images"
        assert (
            len(input_grid_shape) == num_ims
        ), "Grid shape length must match number of input images"

        # Combine images across scales
        x = torch.cat(x, dim=1)

        return x, input_spacing, input_grid_shape, leading_tokens  # type: ignore

    def forward(
        self,
        x: torch.Tensor | Sequence[torch.Tensor],
        input_spacing: Tuple[float, ...] | Sequence[Tuple[float, ...]],
        input_grid_shape: Tuple[int, ...] | Sequence[Tuple[int, ...]],
        leading_tokens: bool = False,
    ) -> torch.Tensor | Sequence[torch.Tensor]:
        """Forward pass of Multi-Resolution RoSE Transformer Block.

        Args:
            x: Input tensor of shape [B, T, E] or sequence of tensors
            input_spacing: Spacing information for the input(s)
            input_grid_shape: Grid shape of the input(s)
            leading_tokens: Whether the first tensor in the sequence `x` are non-rotating tokens to be included in attention. If `True`, `x` must be a sequence of >= 2 tensors.

        Returns:
            Output tensor(s) with same shape as input
        """
        # Prepare inputs
        x, input_spacing, input_grid_shape, leading_tokens = self._prepare_inputs(
            x, input_spacing, input_grid_shape, leading_tokens  # type: ignore
        )

        # Store original input for residual connection
        identity = x

        # Multi-head self-attention with pre-normalization
        x = self.norm1(x)

        # Compute query, key, and value tensors
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Get dimensions for reshaping
        B, N, D = q.shape

        # Reshape for multi-head attention: (B, N, D) -> (B, N, H, D_head)
        # Also need to make stand-in
        _q = q.reshape(B, N, self.num_heads, self.head_dim).contiguous()
        _k = k.reshape(B, N, self.num_heads, self.head_dim).contiguous()
        v = v.reshape(B, N, self.num_heads, self.head_dim).contiguous()

        start = int(leading_tokens)
        for spacing, grid_shape in zip(input_spacing, input_grid_shape):
            length = math.prod(grid_shape)
            # Apply RoSE to queries and keys for this scale
            _q[:, start : start + length] = self.rose(
                q[:, start : start + length], spacing, grid_shape
            )
            _k[:, start : start + length] = self.rose(
                k[:, start : start + length], spacing, grid_shape
            )
            start += length

        # [B, N, H, D_head] -> [B, H, N, D_head]
        q = _q.transpose(1, 2)
        k = _k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Perform attention
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            scale=self.scale,
        )

        # Reshape back: (B, H, N, D_head) -> (B, N, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)

        # Projection and dropout
        attn_out = self.proj(attn_out)
        attn_out = self.proj_dropout(attn_out)

        # First residual connection with stochastic depth
        x = identity + self.drop_path(attn_out)

        # MLP with pre-normalization and residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # Split x back into the original multi-resolution sequence format
        if len(input_grid_shape) > 1 or leading_tokens:
            outputs = []
            if leading_tokens:
                outputs = [x[:, :leading_tokens, :]]  # type: ignore
            start = int(leading_tokens)
            for grid_shape in input_grid_shape:
                length = math.prod(grid_shape)
                outputs.append(x[:, start : start + length, :])  # type: ignore
                start += length
            return outputs
        return x


if __name__ == "__main__":
    # Example usage
    num_heads = 8
    spatial_dims = 3
    dim = 16 * 2 * spatial_dims * num_heads  # Ensure dim is divisible by num_heads
    layer = RoSEMultiHeadCrossAttention(
        feature_dims=dim, num_heads=num_heads, spatial_dims=spatial_dims
    )

    # create grid_shape and spacing based on spatial_dims
    grid_shape = tuple([10] * spatial_dims)
    spacing = tuple([1.0] * spatial_dims)
    batch_size, seq_len = 2, math.prod(grid_shape)

    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)

    attn = layer(q, k, spacing, grid_shape)
    assert attn.shape == (batch_size, num_heads, seq_len, seq_len)
