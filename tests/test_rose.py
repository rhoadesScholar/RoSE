"""Test suite for RoSE package."""

import pytest
import torch

from RoSE import RoSEMultiHeadAttention


class TestRoSEMultiHeadAttention:
    """Test cases for RoSEMultiHeadAttention."""

    @pytest.mark.parametrize(
        "spatial_dims, learnable",
        [
            (2, True),
            (2, False),
            (3, True),
            (3, False),
        ],
    )
    def test_rose_layer_init(self, spatial_dims, learnable):
        """Test RoSELayer initialization for various spatial dimensions and learnability."""
        dim = 64 * 2 * spatial_dims  # Ensure dim is divisible by num_heads
        num_heads = 8
        layer = RoSEMultiHeadAttention(
            dim=dim, num_heads=num_heads, spatial_dims=spatial_dims, learnable=learnable
        )
        assert layer.dim == dim
        assert layer.num_heads == num_heads
        assert layer.spatial_dims == spatial_dims
        assert layer.learnable == learnable

    @pytest.mark.parametrize(
        "spatial_dims, learnable",
        [
            (2, True),
            (2, False),
            (3, True),
            (3, False),
        ],
    )
    def test_rose_layer_forward(self, spatial_dims, learnable):
        """Test RoSE forward pass for 2D/3D and learnable/non-learnable."""
        dim = 64 * 2 * spatial_dims  # Ensure dim is divisible by num_heads
        num_heads = 8
        layer = RoSEMultiHeadAttention(
            dim=dim, num_heads=num_heads, spatial_dims=spatial_dims, learnable=learnable
        )

        # create grid_shape and voxel_size based on spatial_dims
        grid_shape = tuple([10] * spatial_dims)
        voxel_size = tuple([1.0] * spatial_dims)

        # Example batch size and sequence length including CLS token
        batch_size, seq_len = 2, int(torch.prod(torch.tensor(grid_shape)))
        q = torch.randn(batch_size, seq_len, dim)
        k = torch.randn(batch_size, seq_len, dim)

        # Forward pass
        attn = layer(q, k, voxel_size, grid_shape)

        assert attn.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_dim_not_divisible_by_heads(self):
        """Test that initialization fails when dim is not divisible by num_heads."""
        with pytest.raises(AssertionError):
            RoSEMultiHeadAttention(dim=129, num_heads=8)
