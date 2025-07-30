"""Test suite for RoSE package."""

import pytest
import torch

from RoSE import RoSELayer, RoSEMultiheadSelfAttention


class TestRoSELayer:
    """Test cases for RoSELayer."""
    
    def test_rose_layer_init(self):
        """Test RoSELayer initialization."""
        layer = RoSELayer(dim=128, num_heads=8, spatial_dims=3, learnable=True)
        assert layer.dim == 128
        assert layer.num_heads == 8
        assert layer.head_dim == 16
        assert layer.spatial_dims == 3
    
    def test_rose_layer_forward(self):
        """Test RoSELayer forward pass."""
        layer = RoSELayer(dim=128, num_heads=8, spatial_dims=3, learnable=True)
        
        batch_size, seq_len = 2, 100
        q = torch.randn(batch_size, seq_len, 128)
        k = torch.randn(batch_size, seq_len, 128)
        
        grid_shape = (10, 10, 10)
        voxel_size = (1.0, 1.0, 1.0)
        
        q_out, k_out = layer(q, k, grid_shape, voxel_size)
        
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
    
    def test_dim_not_divisible_by_heads(self):
        """Test that initialization fails when dim is not divisible by num_heads."""
        with pytest.raises(AssertionError):
            RoSELayer(dim=129, num_heads=8)


class TestRoSEMultiheadSelfAttention:
    """Test cases for RoSEMultiheadSelfAttention."""
    
    def test_rose_mha_init(self):
        """Test RoSEMultiheadSelfAttention initialization."""
        mha = RoSEMultiheadSelfAttention(dim=128, num_heads=8, spatial_dims=3, learnable=True)
        assert mha.dim == 128
        assert mha.num_heads == 8
        assert mha.head_dim == 16
        assert mha.spatial_dims == 3
    
    def test_rose_mha_forward(self):
        """Test RoSEMultiheadSelfAttention forward pass."""
        mha = RoSEMultiheadSelfAttention(dim=128, num_heads=8, spatial_dims=3, learnable=True)
        
        batch_size, seq_len = 2, 100
        x = torch.randn(batch_size, seq_len, 128)
        
        grid_shape = (10, 10, 10)
        voxel_size = (1.0, 1.0, 1.0)
        
        output = mha(x, grid_shape, voxel_size)
        
        assert output.shape == x.shape
    
    def test_different_spatial_dims(self):
        """Test with different spatial dimensions."""
        # Test 2D
        mha_2d = RoSEMultiheadSelfAttention(dim=128, num_heads=8, spatial_dims=2)
        x = torch.randn(2, 100, 128)
        grid_shape = (10, 10)
        voxel_size = (1.0, 1.0)
        
        output = mha_2d(x, grid_shape, voxel_size)
        assert output.shape == x.shape
        
        # Test 1D
        mha_1d = RoSEMultiheadSelfAttention(dim=128, num_heads=8, spatial_dims=1)
        grid_shape = (100,)
        voxel_size = (1.0,)
        
        output = mha_1d(x, grid_shape, voxel_size)
        assert output.shape == x.shape


@pytest.mark.slow
class TestRoSEPerformance:
    """Performance and integration tests."""
    
    def test_large_batch_processing(self):
        """Test processing with larger batch sizes."""
        layer = RoSELayer(dim=256, num_heads=16, spatial_dims=3)
        
        batch_size, seq_len = 8, 1000
        q = torch.randn(batch_size, seq_len, 256)
        k = torch.randn(batch_size, seq_len, 256)
        
        grid_shape = (10, 10, 10)
        voxel_size = (0.5, 0.5, 0.5)
        
        q_out, k_out = layer(q, k, grid_shape, voxel_size)
        
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
