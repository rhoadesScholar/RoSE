"""Test suite for RoSE package."""

import math

import pytest
import torch

from RoSE import MultiRes_RoSE_Block, RoSEMultiHeadCrossAttention

torch.autograd.set_detect_anomaly(True)


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
        layer = RoSEMultiHeadCrossAttention(
            feature_dims=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=learnable,
        )
        assert layer.feature_dims == dim
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
        layer = RoSEMultiHeadCrossAttention(
            feature_dims=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=learnable,
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
            RoSEMultiHeadCrossAttention(feature_dims=129, num_heads=8)


class TestMultiResRoSETransformerBlock:
    """Test cases for MultiRes_RoSE_TransformerBlock."""

    @pytest.mark.parametrize(
        "spatial_dims, rotary_ratio",
        [
            (2, 1.0),
            (2, 0.5),
            (3, 1.0),
            (3, 0.25),
        ],
    )
    def test_transformer_block_init(self, spatial_dims, rotary_ratio):
        """Test that MultiRes_RoSE_TransformerBlock initializes correctly."""
        dim = 64
        num_heads = 8

        block = MultiRes_RoSE_Block(
            feature_dims=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            rotary_ratio=rotary_ratio,
            mlp_ratio=2.0,
            attn_dropout=0.1,
            proj_dropout=0.1,
            drop_path=0.1,
        )

        assert block.feature_dims == dim
        assert block.num_heads == num_heads
        assert block.spatial_dims == spatial_dims
        assert block.head_dim == dim // num_heads

    def test_single_tensor_forward(self):
        """Test forward pass with single tensor input."""
        batch_size = 2
        dim = 64
        num_heads = 8
        spatial_dims = 2
        grid_shape = (8, 8)
        seq_len = math.prod(grid_shape)
        spacing = (1.0, 1.0)

        block = MultiRes_RoSE_Block(
            feature_dims=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            rotary_ratio=1.0,
        )

        x = torch.randn(batch_size, seq_len, dim)

        output = block(x, spacing, grid_shape)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, seq_len, dim)

    def test_multi_resolution_forward(self):
        """Test forward pass with multi-resolution input."""
        batch_size = 2
        dim = 64
        num_heads = 8
        spatial_dims = 2

        # Two different resolutions
        grid_shape_1 = (4, 4)  # 16 tokens
        grid_shape_2 = (8, 8)  # 64 tokens
        spacing_1 = (1.0, 1.0)
        spacing_2 = (0.5, 0.5)

        seq_len_1 = math.prod(grid_shape_1)
        seq_len_2 = math.prod(grid_shape_2)

        block = MultiRes_RoSE_Block(
            feature_dims=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            rotary_ratio=0.5,
        )

        x1 = torch.randn(batch_size, seq_len_1, dim)
        x2 = torch.randn(batch_size, seq_len_2, dim)
        x_sequence = [x1, x2]

        output = block(
            x_sequence,
            input_spacing=[spacing_1, spacing_2],
            input_grid_shape=[grid_shape_1, grid_shape_2],
        )

        assert isinstance(output, list)
        assert len(output) == 2
        assert output[0].shape == (batch_size, seq_len_1, dim)
        assert output[1].shape == (batch_size, seq_len_2, dim)

    def test_zero_rotary_ratio(self):
        """Test with zero rotary ratio (no rotation)."""
        batch_size = 2
        dim = 64
        num_heads = 8
        spatial_dims = 2
        grid_shape = (4, 4)
        seq_len = math.prod(grid_shape)
        spacing = (1.0, 1.0)

        block = MultiRes_RoSE_Block(
            feature_dims=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            rotary_ratio=0.0,  # No rotation
        )

        x = torch.randn(batch_size, seq_len, dim)
        output = block(x, spacing, grid_shape)

        assert output.shape == x.shape

        # Also test the attention layer directly with zero rotary ratio
        attn_layer = RoSEMultiHeadCrossAttention(
            feature_dims=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            rotary_ratio=0.0,
        )

        q = torch.randn(batch_size, seq_len, dim)
        k = torch.randn(batch_size, seq_len, dim)

        # This should trigger the else branch in _reshape_qkv
        attn_output = attn_layer(q, k, spacing, grid_shape)
        assert attn_output.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_different_dropout_rates(self):
        """Test with different dropout configurations."""
        batch_size = 2
        dim = 64
        num_heads = 8
        spatial_dims = 2
        grid_shape = (4, 4)
        seq_len = math.prod(grid_shape)
        spacing = (1.0, 1.0)

        block = MultiRes_RoSE_Block(
            feature_dims=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            attn_dropout=0.2,
            proj_dropout=0.1,
            mlp_dropout=0.15,
            drop_path=0.05,
        )

        x = torch.randn(batch_size, seq_len, dim)

        # Test in training mode
        block.train()
        output_train = block(x, spacing, grid_shape)

        # Test in eval mode
        block.eval()
        output_eval = block(x, spacing, grid_shape)

        assert output_train.shape == x.shape
        assert output_eval.shape == x.shape

    def test_gradient_flow(self):
        """Test that gradients flow through the transformer block."""
        batch_size = 2
        dim = 64
        num_heads = 8
        spatial_dims = 2
        grid_shape = (4, 4)
        seq_len = math.prod(grid_shape)
        spacing = (1.0, 1.0)

        def block():
            return MultiRes_RoSE_Block(
                feature_dims=dim,
                num_heads=num_heads,
                spatial_dims=spatial_dims,
                learnable=True,  # Ensure learnable parameters
            )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = block()
                self.block2 = block()

            def forward(self, x, spacing, grid_shape):
                x = self.block1(x, spacing, grid_shape)
                x = self.block2(x, spacing, grid_shape)
                return x

        model = Model()

        x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        output = model(x, spacing, grid_shape)

        # Simple loss for gradient computation
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None

        # Make sure model has parameters:
        assert len(list(model.parameters())) > 0

        # Check that all model parameters have non-zero gradients
        for param in model.parameters():
            assert param.grad is not None and param.grad.abs().sum() != 0

    def test_different_mlp_ratios(self):
        """Test with different MLP expansion ratios."""
        for mlp_ratio in [1.0, 2.0, 4.0]:
            block = MultiRes_RoSE_Block(
                feature_dims=64,
                num_heads=8,
                spatial_dims=2,
                mlp_ratio=mlp_ratio,
            )

            expected_hidden_dim = int(64 * mlp_ratio)
            assert block.mlp.fc1.out_features == expected_hidden_dim

    def test_invalid_dimensions(self):
        """Test that invalid dimension configurations raise errors."""
        # dim not divisible by num_heads
        with pytest.raises(AssertionError):
            MultiRes_RoSE_Block(feature_dims=65, num_heads=8)

    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    def test_device_consistency(self, device):
        """Test that the transformer block works on different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size = 2
        dim = 64
        num_heads = 8
        spatial_dims = 2
        grid_shape = (4, 4)
        seq_len = math.prod(grid_shape)
        spacing = (1.0, 1.0)

        block = MultiRes_RoSE_Block(
            feature_dims=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
        ).to(device)

        x = torch.randn(batch_size, seq_len, dim, device=device)
        output = block(x, spacing, grid_shape)

        assert output.device == x.device
        assert output.shape == x.shape
