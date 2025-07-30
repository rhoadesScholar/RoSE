import functools
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor


# ----------  helpers ---------------------------------------------------------


def _make_base_frequencies(
    dim_half: int,
    learnable: bool,
    n_heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Parameter | nn.Buffer:
    """
    Return a (n_heads, dim_half) tensor of base frequencies θ.
    """
    freq = 1.0 / (
        10000 ** (torch.arange(0, dim_half, dtype=dtype, device=device) / dim_half)
    )
    freq = freq.unsqueeze(0).repeat(n_heads, 1)  # share across heads
    if learnable:
        return nn.Parameter(freq)
    else:
        # register_buffer keeps it on the right device & in state_dict (no grad)
        return nn.Buffer(freq)


def _rotate_half(x: Tensor) -> Tensor:
    # x (..., 2k) →  (..., 2k)  where even/odd pairs are rotated
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _rope_apply(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (B, H, N, D) ;  cos/sin: (N, 1, 1, D/2) after broadcasting
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    # expand cos/sin to (..., D/2)
    return torch.cat([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)


# ----------  cache -----------------------------------------------------------


class _RotationCache:
    """
    Memoises (cos, sin) tables for each (device, grid_shape, voxel_size, head_dim_half).
    The key is a tuple
         (device, grid_shape, voxel_size_tuple, dim_half, dtype)
    """

    def __init__(self) -> None:
        self._store: Dict[Tuple[Any, ...], Tuple[Tensor, Tensor]] = {}

    @staticmethod
    def _build_angles(
        phys_coords: Tensor, theta: Tensor  # (N, spatial_dims)  # (heads, dim_half)
    ) -> Tensor:  # → (N, heads, dim_half)
        # phys_coords = p · s  (already in physical units)
        # angle = p_phys ⋅ θ   (broadcasted over heads)
        return torch.einsum("nd,hd->nhd", phys_coords, theta)

    def get(
        self,
        grid_shape: Tuple[int, ...],
        voxel_size: Tuple[float, ...],
        theta: Tensor,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        device = theta.device
        dim_half = theta.shape[1]
        key = (device, grid_shape, voxel_size, dim_half, dtype)
        if key not in self._store:

            # 1. build physical-unit coordinates           (N, spatial_dims)
            coords = [torch.arange(n, device=device, dtype=dtype) for n in grid_shape]
            mesh = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)
            phys = mesh.reshape(-1, len(grid_shape)) * torch.tensor(
                voxel_size, device=device, dtype=dtype
            )

            # 2. angles and trig caches
            angle = self._build_angles(phys, theta.to(dtype))  # (N, H, D/2)
            cos = angle.cos()
            sin = angle.sin()

            # 3. reshape to   (N, 1, 1, D/2)   for cheap broadcasting later
            cos = cos.permute(0, 2, 1).reshape(-1, 1, 1, dim_half)
            sin = sin.permute(0, 2, 1).reshape(-1, 1, 1, dim_half)

            self._store[key] = (cos, sin)

        return self._store[key]


_ROSE_CACHE = _RotationCache()


class RoSELayer(nn.Module):
    """
    Applies **Rotary Spatial Embeddings** to q & k.

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

    @functools.lru_cache(maxsize=128)
    @torch.no_grad()
    def get_grid_centers(self, shape):
        """
        shape: (H, W) or (D, H, W)
        Returns: Centers of the grid of patches
            --> [N_patches, spatial_dims]
        """
        # Get the center of each patch
        grid_centers = [
            (
                torch.arange(
                    self.patch_size[i] / 2,
                    shape[i],
                    step=self.patch_size[i],
                )
                if shape[i] > 1
                else torch.tensor([0.0])
            )
            for i in range(self.spatial_dims)
        ]
        grid_centers = torch.stack(
            torch.meshgrid(*grid_centers, indexing="ij"), dim=-1
        ).reshape(-1, self.spatial_dims)
        # Center around the origin
        grid_centers -= torch.tensor(shape) / 2.0

        _param = next(self.mlp.parameters())
        return grid_centers.to(dtype=_param.dtype, device=_param.device)

    @functools.lru_cache(maxsize=128)
    @torch.no_grad()
    def get_scaled_centers(self, shape, scale: torch.Tensor):
        """
        shape: (H, W) or (D, H, W)
        Returns: [N_patches, *spatial_dims]
        """
        # Get the center of each patch in the grid
        grid_centers = self.get_grid_centers(shape)
        return grid_centers * scale.to(
            dtype=grid_centers.dtype, device=grid_centers.device
        )

    @functools.lru_cache(maxsize=128)
    @torch.no_grad()
    def get_real_centers(self, shape, scale, center=None):
        """
        shape: (H, W) or (D, H, W)
        scale: (s_y, s_x) or (s_z, s_y, s_x)
        center: (c_y, c_x) or (c_z, c_y, c_x), defaults to (0, 0) or (0, 0, 0)
        Returns: [N_patches, *spatial_dims]
        """
        if center is None:
            center = torch.zeros(self.spatial_dims)
        elif not isinstance(center, torch.Tensor):
            center = torch.tensor(center)

        # Get the center of each patch in the grid
        scaled_centers = self.get_scaled_centers(shape, scale)
        return scaled_centers + center.to(
            dtype=scaled_centers.dtype, device=scaled_centers.device
        )

    def forward(
        self,
        q: Tensor,  # (B, N, C)
        k: Tensor,  # (B, N, C)
        grid_shape: Tuple[int, ...],
        voxel_size: Tuple[float, ...],
    ) -> Tuple[Tensor, Tensor]:
        assert (
            len(grid_shape) == self.spatial_dims
        ), f"grid_shape must have {self.spatial_dims} dims"

        # ---  retrieve / build trig tables
        cos, sin = _ROSE_CACHE.get(
            grid_shape, voxel_size, self.theta, dtype=q.dtype
        )  # (N,1,1,D/2)

        # cos/sin: broadcast to (N, B, H, D/2) by unsqueezing properly
        cos = cos.to(q.device)
        sin = sin.to(q.device)

        q = _rope_apply(q, cos, sin)
        k = _rope_apply(k, cos, sin)

        return q, k
