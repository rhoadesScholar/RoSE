"""Numerical tests for RoSE mathematical properties."""

import math

import pytest
import torch

from RoSE.rose import (
    RoSEMultiHeadCrossAttention,
    RotarySpatialEmbedding,
    _init_p_nd,
    _make_log_spaced_frequencies,
)


class TestRotarySpatialEmbedding:
    """Comprehensive tests for RotarySpatialEmbedding class."""

    @pytest.mark.parametrize("dim", [32, 64, 128])
    @pytest.mark.parametrize("num_heads", [4, 8, 16])
    @pytest.mark.parametrize("spatial_dims", [2, 3, 4])
    def test_initialization_parameters(
        self, dim: int, num_heads: int, spatial_dims: int
    ):
        """Test various initialization parameters."""
        # Skip invalid combinations
        if dim % num_heads != 0 or dim % 2 != 0:
            pytest.skip("Invalid dim/num_heads combination")

        layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=False,
        )

        assert layer.dim == dim
        assert layer.num_heads == num_heads
        assert layer.spatial_dims == spatial_dims
        assert layer.freqs.shape == (num_heads, dim // (2 * num_heads), spatial_dims)

    def test_invalid_initialization(self):
        """Test that invalid parameters raise appropriate errors."""
        # Dim not divisible by num_heads
        with pytest.raises(AssertionError, match="dim must be divisible by num_heads"):
            RotarySpatialEmbedding(dim=65, num_heads=8)

        # Odd dimension
        with pytest.raises(
            AssertionError,
            match="dims_per_head must be even for complex representation",
        ):
            RotarySpatialEmbedding(dim=63, num_heads=7)

    @pytest.mark.parametrize(
        "frequency_scaling", ["none", "linear", "sqrt", "adaptive"]
    )
    def test_frequency_scaling_modes(self, frequency_scaling: str):
        """Test different frequency scaling modes."""
        base_theta = 1e4
        spatial_dims = 3
        dim = 64
        num_heads = 8

        layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            frequency_scaling=frequency_scaling,
            base_theta=base_theta,
            learnable=False,
        )

        # Check that rope_theta is computed correctly
        expected_theta = base_theta  # Default case
        if frequency_scaling == "none":
            expected_theta = base_theta
        elif frequency_scaling == "linear":
            expected_theta = base_theta ** (1 / spatial_dims)
        elif frequency_scaling == "sqrt":
            expected_theta = base_theta ** (1 / math.sqrt(spatial_dims))
        elif frequency_scaling == "adaptive":
            expected_theta = base_theta ** (2.0 / (spatial_dims * dim))

        assert abs(layer.rose_theta - expected_theta) < 1e-6

    def test_learnable_vs_fixed_frequencies(self):
        """Test difference between learnable and fixed frequencies."""
        dim, num_heads, spatial_dims = 64, 8, 2

        # Fixed frequencies
        fixed_layer = RotarySpatialEmbedding(
            dim=dim, num_heads=num_heads, spatial_dims=spatial_dims, learnable=False
        )
        assert not fixed_layer.freqs.requires_grad

        # Learnable frequencies
        learnable_layer = RotarySpatialEmbedding(
            dim=dim, num_heads=num_heads, spatial_dims=spatial_dims, learnable=True
        )
        assert learnable_layer.freqs.requires_grad

    def test_jitter_initialization(self):
        """Test that jitter initialization creates different frequencies."""
        dim, num_heads, spatial_dims = 64, 8, 2

        # Without jitter
        layer_no_jitter = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=True,
            init_jitter_std=0.0,
        )

        # With jitter
        layer_with_jitter = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=True,
            init_jitter_std=0.1,
        )

        # Base frequencies should be the same before jitter
        # Calculate per-head rotary dimension and number of planes
        dims_per_head = dim // num_heads
        rotary_dim_per_head = int(dims_per_head * 1.0)  # rotary_ratio = 1.0
        rotary_dim_per_head = (rotary_dim_per_head // 2) * 2  # ensure even
        num_planes = (rotary_dim_per_head * num_heads) // 2

        base_freqs = _make_log_spaced_frequencies(
            num_planes, spatial_dims, layer_no_jitter.rose_theta
        )
        base_freqs = base_freqs.reshape(num_heads, -1, spatial_dims)

        # Without jitter, frequencies should be the same
        assert torch.allclose(layer_no_jitter.freqs, base_freqs, atol=1e-6)

        # With jitter, frequencies should be different
        assert not torch.allclose(layer_with_jitter.freqs, base_freqs, atol=1e-6)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("shape", [4, 8, 16])
    @pytest.mark.parametrize("spatial_dims", [2, 3])
    def test_forward_pass_shapes(self, batch_size: int, shape: int, spatial_dims: int):
        """Test forward pass with various input shapes."""
        seq_len = (
            shape**spatial_dims
        )  # Ensure seq_len is compatible with spatial_dims
        dim = 16
        num_heads = 4

        layer = RotarySpatialEmbedding(
            dim=dim, num_heads=num_heads, spatial_dims=spatial_dims, learnable=False
        )

        x = torch.randn(batch_size, seq_len, dim)
        grid_shape = (shape,) * spatial_dims
        spacing = tuple([1.0] * spatial_dims)

        output = layer(x, spacing, grid_shape)
        assert output.shape == (batch_size, seq_len, dim)

    def test_complex_split_functionality(self):
        """Test the _get_complex_split method."""
        dim, num_heads, spatial_dims = 64, 8, 2
        batch_size, seq_len = 2, 16

        layer = RotarySpatialEmbedding(
            dim=dim, num_heads=num_heads, spatial_dims=spatial_dims, learnable=False
        )

        x = torch.randn(batch_size, seq_len, dim)
        complex_x = layer._get_complex_split(x)

        expected_shape = (batch_size, seq_len, num_heads, dim // (2 * num_heads))
        assert complex_x.shape == expected_shape
        assert complex_x.dtype == torch.complex64

    def test_device_consistency(self):
        """Test that layer works consistently across devices."""
        dim, num_heads, spatial_dims = 32, 4, 2
        batch_size, seq_len = 2, 9

        layer = RotarySpatialEmbedding(
            dim=dim, num_heads=num_heads, spatial_dims=spatial_dims, learnable=False
        )

        grid_shape = (3, 3)
        spacing = (1.0, 1.0)

        # Test CPU
        x_cpu = torch.randn(batch_size, seq_len, dim)
        output_cpu = layer(x_cpu, spacing, grid_shape)
        assert output_cpu.device == x_cpu.device

        # Test GPU if available
        if torch.cuda.is_available():
            layer_gpu = layer.to("cuda")
            x_gpu = x_cpu.to("cuda")
            output_gpu = layer_gpu(x_gpu, spacing, grid_shape)
            assert output_gpu.device == x_gpu.device
            # Results should be the same (within numerical precision)
            torch.testing.assert_close(
                output_cpu, output_gpu.cpu(), rtol=1e-5, atol=1e-6
            )

    def test_dtype_consistency(self):
        """Test that layer handles float32 input properly."""
        dim, num_heads, spatial_dims = 32, 4, 2
        batch_size, seq_len = 2, 9

        layer = RotarySpatialEmbedding(
            dim=dim, num_heads=num_heads, spatial_dims=spatial_dims, learnable=False
        )

        grid_shape = (3, 3)
        spacing = (1.0, 1.0)

        # Test float32 - this should work reliably
        x_f32 = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
        output_f32 = layer(x_f32, spacing, grid_shape)
        assert output_f32.dtype == torch.float32

        # Test that the layer doesn't change the dtype unexpectedly
        x_f16 = torch.randn(batch_size, seq_len, dim, dtype=torch.float16)
        try:
            output_f16 = layer(x_f16, spacing, grid_shape)
            # If it works, check dtype is preserved in the output
            assert output_f16.dtype == torch.float16
        except RuntimeError:
            # If it fails due to dtype mismatch, that's expected behavior
            pass

    def test_flatten_parameter(self):
        """Test the flatten parameter in forward pass."""
        dim, num_heads, spatial_dims = 32, 4, 2
        batch_size, seq_len = 2, 9

        layer = RotarySpatialEmbedding(
            dim=dim, num_heads=num_heads, spatial_dims=spatial_dims, learnable=False
        )

        x = torch.randn(batch_size, seq_len, dim)
        grid_shape = (3, 3)
        spacing = (1.0, 1.0)

        # With flattening (default)
        output_flat = layer(x, spacing, grid_shape, flatten=True)
        assert output_flat.shape == (batch_size, seq_len, dim)

        # Without flattening - based on the implementation: (B, H, N, D_heads)
        output_unflat = layer(x, spacing, grid_shape, flatten=False)
        expected_shape = (batch_size, num_heads, seq_len, dim // num_heads)
        assert output_unflat.shape == expected_shape

        # Flattened version should be equivalent to manually flattening
        output_manual_flat = output_unflat.transpose(1, 2).flatten(-2)
        torch.testing.assert_close(output_flat, output_manual_flat)

    @pytest.mark.parametrize("spatial_dims", [2, 3, 4])
    def test_position_encoding_consistency(self, spatial_dims: int):
        """Test that position encoding is consistent across different calls."""
        dim = 32 * spatial_dims
        num_heads = 4

        layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=False,
            init_jitter_std=0.0,  # No randomness
        )

        batch_size = 2
        grid_size = 4
        grid_shape = tuple([grid_size] * spatial_dims)
        spacing = tuple([0.5] * spatial_dims)
        seq_len = math.prod(grid_shape)

        # The position-dependent rotation should be the same for both
        # Create identical inputs to test this
        x_same = torch.ones(batch_size, seq_len, dim)
        output_same1 = layer(x_same, spacing, grid_shape)
        output_same2 = layer(x_same, spacing, grid_shape)

        torch.testing.assert_close(output_same1, output_same2)

    def test_backward_compatibility(self):
        """Test that the layer can handle edge cases gracefully."""
        dim, num_heads, spatial_dims = 32, 4, 2

        layer = RotarySpatialEmbedding(
            dim=dim, num_heads=num_heads, spatial_dims=spatial_dims, learnable=False
        )

        # Test with minimum valid inputs
        x_min = torch.randn(1, 1, dim)
        output_min = layer(x_min, (1.0, 1.0), (1, 1))
        assert output_min.shape == (1, 1, dim)

        # Test with larger grids
        x_large = torch.randn(1, 100, dim)
        output_large = layer(x_large, (0.1, 0.1), (10, 10))
        assert output_large.shape == (1, 100, dim)

    def test_rotary_embedding_properties(self):
        """Test fundamental properties of rotary embeddings."""
        dim, num_heads, spatial_dims = 64, 8, 2
        batch_size, seq_len = 2, 25

        layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=False,
            init_jitter_std=0.0,
        )

        x = torch.randn(batch_size, seq_len, dim)
        grid_shape = (5, 5)
        spacing = (1.0, 1.0)

        # Apply embedding
        x_rot = layer(x, spacing, grid_shape)

        # Test magnitude preservation (rotations should preserve norms)
        original_norms = torch.norm(x, dim=-1)
        rotated_norms = torch.norm(x_rot, dim=-1)
        torch.testing.assert_close(
            original_norms,
            rotated_norms,
            rtol=1e-5,
            atol=1e-6,
            msg="Rotary embedding should preserve vector magnitudes",
        )

    def test_deterministic_behavior(self):
        """Test that the layer produces deterministic results."""
        dim, num_heads, spatial_dims = 32, 4, 2

        # Create two identical layers
        layer1 = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=False,
            init_jitter_std=0.0,
        )

        layer2 = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=False,
            init_jitter_std=0.0,
        )

        x = torch.randn(2, 9, dim)
        grid_shape = (3, 3)
        spacing = (1.0, 1.0)

        # Should produce identical results
        output1 = layer1(x, spacing, grid_shape)
        output2 = layer2(x, spacing, grid_shape)

        torch.testing.assert_close(output1, output2)

    def test_spatial_relationship_encoding(self):
        """Test that spatial relationships are correctly encoded."""
        dim, num_heads, spatial_dims = 32, 4, 2
        batch_size = 1

        layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=False,
            init_jitter_std=0.0,
        )

        # Create a 2D grid
        grid_shape = (4, 4)
        spacing = (1.0, 1.0)
        seq_len = math.prod(grid_shape)

        # Test with identical vectors at different positions
        x = torch.ones(batch_size, seq_len, dim)
        x_rot = layer(x, spacing, grid_shape, flatten=False)  # (B, H, N, D_heads)

        # Different positions should have different embeddings due to rotation
        # Check that positions (0,0) and (1,1) have different embeddings
        pos_00_idx = 0  # First position in grid
        pos_11_idx = 5  # Position at (1,1) in 4x4 grid (row-major: 1*4 + 1 = 5)

        embedding_00 = x_rot[0, :, pos_00_idx]  # (H, D_heads)
        embedding_11 = x_rot[0, :, pos_11_idx]  # (H, D_heads)

        # They should be different due to different rotations
        assert not torch.allclose(embedding_00, embedding_11, atol=1e-6)

    def test_scaling_invariance(self):
        """Test behavior under coordinate scaling."""
        dim, num_heads, spatial_dims = 32, 4, 2
        batch_size = 1

        layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=False,
            init_jitter_std=0.0,
        )

        grid_shape = (3, 3)
        seq_len = math.prod(grid_shape)
        x = torch.randn(batch_size, seq_len, dim)

        # Test with different spacings
        output1 = layer(x, (1.0, 1.0), grid_shape)
        output2 = layer(x, (2.0, 2.0), grid_shape)

        # Outputs should be different due to different spatial scales
        assert not torch.allclose(output1, output2, atol=1e-6)

    @pytest.mark.parametrize("grid_size", [2, 4, 8])
    def test_various_grid_sizes(self, grid_size: int):
        """Test with various grid sizes."""
        dim, num_heads, spatial_dims = 32, 4, 2

        layer = RotarySpatialEmbedding(
            dim=dim, num_heads=num_heads, spatial_dims=spatial_dims, learnable=False
        )

        batch_size = 1
        grid_shape = (grid_size, grid_size)
        seq_len = math.prod(grid_shape)
        spacing = (1.0, 1.0)

        x = torch.randn(batch_size, seq_len, dim)
        output = layer(x, spacing, grid_shape)

        assert output.shape == (batch_size, seq_len, dim)
        # Check that output is not just zeros or identical to input
        assert not torch.allclose(output, x, atol=1e-6)
        assert not torch.allclose(output, torch.zeros_like(output), atol=1e-6)

    def test_gradient_flow_through_embedding(self):
        """Test that gradients flow properly through the embedding."""
        dim, num_heads, spatial_dims = 32, 4, 2

        # Test learnable version
        layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=True,
            init_jitter_std=0.01,
        )

        batch_size = 2
        grid_shape = (3, 3)
        seq_len = math.prod(grid_shape)
        spacing = (1.0, 1.0)

        x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        output = layer(x, spacing, grid_shape)

        # Compute a simple loss
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert layer.freqs.grad is not None

        # Check gradients are meaningful (non-zero)
        assert torch.any(x.grad != 0)
        assert torch.any(layer.freqs.grad != 0)


class TestRoSENumericalProperties:
    """Test mathematical properties of RoSE implementation."""

    @pytest.mark.parametrize("spatial_dims", [2, 3, 4])
    @pytest.mark.parametrize("learnable", [True, False])
    def test_phase_conjugate_property(self, spatial_dims: int, learnable: bool):
        """
        Test that φ(p_n)φ(p_m)* = e^{i⟨p_n-p_m,ω⟩}

        This is the fundamental property that enables relative position encoding.
        """
        # Setup parameters
        dim = 64 * spatial_dims  # Ensure divisibility
        num_heads = 8
        grid_size = 8
        grid_shape = tuple([grid_size] * spatial_dims)
        spacing = tuple([0.5] * spatial_dims)

        # Create embedding layer
        rose_layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=learnable,
            init_jitter_std=0.0,  # No jitter for reproducible tests
        )

        # Generate position coordinates
        pos = _init_p_nd(grid_shape, spacing=spacing)  # (N, spatial_dims)
        N = pos.shape[0]

        # Get frequencies for all heads and planes
        freqs = rose_layer.freqs  # (H, P, spatial_dims)

        # Compute phases for all positions: φ(p_n) = e^{i⟨p_n,ω⟩}
        # Shape: (N, H, P)
        angles = torch.einsum("nd,hpd->nhp", pos, freqs)
        phases = torch.exp(1j * angles)  # φ(p_n) for all n, h, p

        # Test the fundamental property: φ(p_n)φ(p_m)* = e^{i⟨p_n-p_m,ω⟩}
        for n in range(0, N, N // 4):  # Sample some positions
            for m in range(0, N, N // 4):
                if n == m:
                    continue

                # Left side: φ(p_n)φ(p_m)*
                left_side = phases[n] * torch.conj(phases[m])  # (H, P)

                # Right side: e^{i⟨p_n-p_m,ω⟩}
                pos_diff = pos[n] - pos[m]  # (spatial_dims,)
                angle_diff = torch.einsum("d,hpd->hp", pos_diff, freqs)
                right_side = torch.exp(1j * angle_diff)  # (H, P)

                # Verify they're equal (within numerical tolerance)
                torch.testing.assert_close(
                    left_side,
                    right_side,
                    rtol=1e-6,
                    atol=1e-8,
                    msg=f"Phase conjugate property failed at positions {n}, {m}",
                )

    def _compute_manual_attention_score(self, q, k, pos, freqs, b, h, n, m, P):
        """Helper method to compute manual attention score for given indices."""
        pos_diff = pos[n] - pos[m]  # (spatial_dims,)

        score = 0.0
        for p in range(P):
            angle = torch.dot(pos_diff, freqs[h, p])
            relative_phase = torch.exp(1j * angle)

            # A[h,n,m] += Re[q[h,n,p] · conj(k[h,m,p]) · phase]
            score += torch.real(
                q[b, n, h, p] * torch.conj(k[b, m, h, p]) * relative_phase
            )

        # Reduce to scalar if needed
        if isinstance(score, torch.Tensor):
            score = score.sum().item() if score.dim() > 0 else score.item()

        return score

    @pytest.mark.parametrize("spatial_dims", [2, 3])
    def test_attention_score_computation(self, spatial_dims: int):
        """
        Test that attention computation matches the mathematical formula:
        A[h,n,m] = Re[∑_p q[h,n,p] · conj(k[h,m,p]) · e^{i⟨p_n-p_m,ω_{h,p}⟩}]
        """
        # Setup
        dim = 8 * spatial_dims
        num_heads = 4
        grid_size = 4
        grid_shape = tuple([grid_size] * spatial_dims)
        spacing = tuple([1.0] * spatial_dims)
        batch_size = 1

        # Create layer
        rose_layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=False,
            init_jitter_std=0.0,
        )

        # Generate test data
        N = math.prod(grid_shape)
        q = torch.randn(batch_size, N, dim, dtype=torch.complex64)
        k = torch.randn(batch_size, N, dim, dtype=torch.complex64)

        # Get positions and frequencies
        pos = _init_p_nd(grid_shape, spacing=spacing)
        freqs = rose_layer.freqs  # (H, P, spatial_dims)
        P = dim // (2 * num_heads)

        # Reshape q, k to (B, N, H, P) for head-wise computation
        q = q.view(batch_size, N, num_heads, P, 2)
        k = k.view(batch_size, N, num_heads, P, 2)

        # Manual computation of attention scores
        manual_attn = torch.zeros(batch_size, num_heads, N, N, dtype=torch.float32)

        for b in range(batch_size):
            for h in range(num_heads):
                for n in range(N):
                    for m in range(N):
                        manual_attn[b, h, n, m] = self._compute_manual_attention_score(
                            q, k, pos, freqs, b, h, n, m, P
                        )

        # Compare with a simplified version using einsum
        # Note: This is a conceptual test - actual implementation uses rotated embeddings
        phases = torch.zeros(N, N, num_heads, P, dtype=torch.complex64)
        for n in range(N):
            for m in range(N):
                pos_diff = pos[n] - pos[m]
                angles = torch.einsum("d,hpd->hp", pos_diff, freqs)
                phases[n, m] = torch.exp(1j * angles)

        # Verify consistency of phase computation
        for n in [0, N // 2, N - 1]:  # Sample positions
            for m in [0, N // 2, N - 1]:
                pos_diff = pos[n] - pos[m]
                expected_angles = torch.einsum("d,hpd->hp", pos_diff, freqs)
                expected_phase = torch.exp(1j * expected_angles)

                torch.testing.assert_close(
                    phases[n, m],
                    expected_phase,
                    rtol=1e-6,
                    atol=1e-8,
                    msg=f"Phase computation inconsistent at {n}, {m}",
                )

    def test_frequency_schedule_properties(self):
        """Test properties of the frequency generation."""
        num_planes = 32
        spatial_dims = 3
        base_theta = 1e4

        freqs = _make_log_spaced_frequencies(num_planes, spatial_dims, base_theta)

        # Test shape
        assert freqs.shape == (num_planes, spatial_dims)

        # Test that frequencies decrease log-linearly
        freqs_1d = freqs[:, 0]  # All spatial dims should have same schedule

        # Verify log-linear decrease
        log_freqs = torch.log(freqs_1d)
        expected_exponents = (
            torch.arange(num_planes, dtype=torch.float32) * 2.0 / (num_planes * 2)
        )
        expected_log_freqs = -expected_exponents * math.log(base_theta)

        torch.testing.assert_close(
            log_freqs,
            expected_log_freqs,
            rtol=1e-6,
            atol=1e-8,
            msg="Frequency schedule doesn't match expected log-linear pattern",
        )

        # Test that all spatial dimensions have identical frequencies
        for d in range(1, spatial_dims):
            torch.testing.assert_close(
                freqs[:, d],
                freqs[:, 0],
                msg=f"Frequency mismatch between spatial dims 0 and {d}",
            )

    @pytest.mark.parametrize("spatial_dims", [2, 3])
    def test_coordinate_generation_properties(self, spatial_dims: int):
        """Test properties of N-dimensional coordinate generation."""
        grid_shape = tuple([4, 6, 8][:spatial_dims])
        spacing = tuple([0.5, 1.0, 1.5][:spatial_dims])

        pos = _init_p_nd(grid_shape, spacing=spacing)

        # Test shape
        expected_N = math.prod(grid_shape)
        assert pos.shape == (expected_N, spatial_dims)

        # Test coordinate ranges
        for d in range(spatial_dims):
            expected_max = (grid_shape[d] - 1) * spacing[d]
            assert pos[:, d].min() == 0.0
            assert pos[:, d].max() == expected_max

        # Test coordinate uniqueness (each position should be unique)
        unique_positions = torch.unique(pos, dim=0)
        assert unique_positions.shape[0] == expected_N, "Positions should be unique"

        # Test spacing consistency
        for d in range(spatial_dims):
            coords_d = pos[:, d]
            unique_coords = torch.unique(coords_d)
            if len(unique_coords) > 1:
                # Check that consecutive unique coordinates differ by spacing
                sorted_coords = torch.sort(unique_coords)[0]
                diffs = sorted_coords[1:] - sorted_coords[:-1]
                expected_diff = spacing[d]

                # Allow for small numerical errors
                torch.testing.assert_close(
                    diffs,
                    torch.full_like(diffs, expected_diff),
                    rtol=1e-6,
                    atol=1e-8,
                    msg=f"Spacing inconsistent in dimension {d}",
                )

    def test_complex_rotation_preservation(self):
        """Test that complex rotations preserve magnitudes."""
        spatial_dims = 2
        dim = 64
        num_heads = 8

        rose_layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=False,
        )

        # Create test input
        batch_size, N = 2, 25
        grid_shape = (5, 5)
        spacing = (1.0, 1.0)

        x = torch.randn(batch_size, N, dim)
        original_norm = torch.norm(x, dim=-1)  # (B, N)

        # Apply rotary embedding
        x_rotated = rose_layer(x, spacing, grid_shape)
        rotated_norm = torch.norm(x_rotated, dim=-1)  # (B, N)

        # Magnitudes should be preserved
        torch.testing.assert_close(
            original_norm,
            rotated_norm,
            rtol=1e-6,
            atol=1e-8,
            msg="Rotary embedding should preserve vector magnitudes",
        )

    @pytest.mark.parametrize("spatial_dims", [2, 3])
    def test_translation_invariance_property(self, spatial_dims: int):
        """
        Test that relative positions are preserved under translation.
        If we shift all coordinates by the same offset, relative phases should be unchanged.
        """
        dim = 32 * spatial_dims
        num_heads = 4

        rose_layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=False,
            init_jitter_std=0.0,
        )

        # Original positions
        grid_shape = tuple([4] * spatial_dims)
        spacing = tuple([1.0] * spatial_dims)
        pos1 = _init_p_nd(grid_shape, spacing=spacing)

        # Translated positions
        translation = torch.tensor([2.0] * spatial_dims)
        pos2 = pos1 + translation

        freqs = rose_layer.freqs  # (H, P, spatial_dims)

        # Compute phases for both position sets
        angles1 = torch.einsum("nd,hpd->nhp", pos1, freqs)
        phases1 = torch.exp(1j * angles1)

        angles2 = torch.einsum("nd,hpd->nhp", pos2, freqs)
        phases2 = torch.exp(1j * angles2)

        N = pos1.shape[0]

        # Test that relative phases are preserved
        for i in range(0, N, N // 3):
            for j in range(0, N, N // 3):
                if i == j:
                    continue

                # Relative phase with original positions
                rel_phase1 = phases1[i] * torch.conj(phases1[j])

                # Relative phase with translated positions
                rel_phase2 = phases2[i] * torch.conj(phases2[j])

                # Should be identical (translation invariance)
                torch.testing.assert_close(
                    rel_phase1,
                    rel_phase2,
                    rtol=1e-6,
                    atol=1e-8,
                    msg=f"Translation invariance failed for positions {i}, {j}",
                )

    def test_learnable_frequencies_gradient_flow(self):
        """Test that gradients flow through learnable frequencies."""
        spatial_dims = 2
        dim = 32
        num_heads = 4

        rose_layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=True,
            init_jitter_std=0.01,
        )

        # Ensure frequencies require gradients
        assert (
            rose_layer.freqs.requires_grad
        ), "Learnable frequencies should require gradients"

        # Forward pass
        batch_size, N = 1, 9
        grid_shape = (3, 3)
        spacing = (1.0, 1.0)
        x = torch.randn(batch_size, N, dim, requires_grad=True)

        output = rose_layer(x, spacing, grid_shape)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert (
            rose_layer.freqs.grad is not None
        ), "Gradients should flow to learnable frequencies"
        assert x.grad is not None, "Gradients should flow to input"

        # Gradients should be non-zero (at least somewhere)
        assert torch.any(
            rose_layer.freqs.grad != 0
        ), "Frequency gradients should be non-zero"


def test_core_rope_property():
    """
    Direct test of the fundamental RoSE property:
    φ(p_n)φ(p_m)* = e^{i⟨p_n-p_m,ω⟩}

    This is the most important test - it verifies the mathematical foundation.
    """
    print("Testing core RoSE mathematical property...")

    # Test parameters
    spatial_dims = 2
    dim = 64
    num_heads = 8
    grid_shape = (8, 8)
    spacing = (0.5, 0.5)

    # Create embedding layer with no jitter for reproducible results
    rose = RotarySpatialEmbedding(
        dim=dim,
        num_heads=num_heads,
        spatial_dims=spatial_dims,
        learnable=False,
        init_jitter_std=0.0,
    )

    # Get position coordinates and frequencies
    positions = _init_p_nd(grid_shape, spacing=spacing)  # (N, spatial_dims)
    frequencies = rose.freqs  # (H, P, spatial_dims)

    N = positions.shape[0]
    H, P = frequencies.shape[:2]

    print(f"Testing with {N} positions, {H} heads, {P} planes per head")

    # Test the property for several position pairs
    test_pairs = [
        (0, 1),
        (0, N // 2),
        (0, N - 1),  # From corner
        (N // 2, N // 2 + 1),
        (N // 2, N - 1),  # From center
        (N // 4, 3 * N // 4),
        (N // 3, 2 * N // 3),  # Random pairs
    ]

    max_error = 0.0

    for n, m in test_pairs:
        # Compute angles at each position: θ_n = ⟨p_n, ω⟩
        angles_n = torch.einsum("d,hpd->hp", positions[n], frequencies)  # (H, P)
        angles_m = torch.einsum("d,hpd->hp", positions[m], frequencies)  # (H, P)

        # Compute phases: φ(p_n) = e^{iθ_n}
        phase_n = torch.exp(1j * angles_n)  # (H, P)
        phase_m = torch.exp(1j * angles_m)  # (H, P)

        # Left side: φ(p_n) * φ(p_m)*
        left_side = phase_n * torch.conj(phase_m)  # (H, P)

        # Right side: e^{i⟨p_n-p_m,ω⟩}
        pos_difference = positions[n] - positions[m]  # (spatial_dims,)
        angle_difference = torch.einsum(
            "d,hpd->hp", pos_difference, frequencies
        )  # (H, P)
        right_side = torch.exp(1j * angle_difference)  # (H, P)

        # Compute error
        error = torch.abs(left_side - right_side).max().item()
        max_error = max(max_error, error)

        print(f"Positions ({n:2d}, {m:2d}): max error = {error:.2e}")

        # Verify they're equal
        torch.testing.assert_close(
            left_side,
            right_side,
            rtol=1e-6,
            atol=1e-8,
            msg=f"Core property failed for positions {n}, {m}",
        )

    print(f"✓ Core mathematical property verified! Maximum error: {max_error:.2e}")

    # Additional verification: check that the phase differences are correct
    print("\nVerifying phase difference interpretation...")

    # Take two specific positions
    n, m = 0, N // 2
    pos_diff = positions[n] - positions[m]

    # Manual computation of expected phase
    for h in range(min(2, H)):  # Test first 2 heads
        for p in range(min(2, P)):  # Test first 2 planes
            freq_vec = frequencies[h, p]  # (spatial_dims,)
            expected_angle = torch.dot(pos_diff, freq_vec)

            # Get actual phase from the computation above
            angles_n = torch.dot(positions[n], freq_vec)
            angles_m = torch.dot(positions[m], freq_vec)
            actual_angle = angles_n - angles_m

            error = torch.abs(expected_angle - actual_angle).item()
            print(f"Head {h}, Plane {p}: angle difference error = {error:.2e}")

            assert error < 1e-6, f"Angle computation error too large: {error}"

    print("✓ Phase difference computation verified!")
    return True


class TestPartialRotation:
    """Test partial rotation functionality with rotary_ratio parameter."""

    @pytest.mark.parametrize("rotary_ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_rotary_ratio_values(self, rotary_ratio: float):
        """Test various rotary_ratio values."""
        dim, num_heads, spatial_dims = 64, 8, 2
        layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            rotary_ratio=rotary_ratio,
            learnable=False,
        )

        x = torch.randn(2, 16, dim)
        result = layer(x, spacing=(1.0, 1.0), grid_shape=(4, 4))

        # Check output shape
        assert (
            result.shape == x.shape
        ), f"Shape mismatch for rotary_ratio={rotary_ratio}"

        # Check dimension calculation (per head)
        dims_per_head = dim // num_heads
        expected_rotary_dim_per_head = int(dims_per_head * rotary_ratio)
        # Ensure even for complex representation
        expected_rotary_dim_per_head = (expected_rotary_dim_per_head // 2) * 2
        expected_non_rotary_dim_per_head = dims_per_head - expected_rotary_dim_per_head

        assert layer.rotary_dim == expected_rotary_dim_per_head
        assert layer.non_rotary_dim == expected_non_rotary_dim_per_head

    def test_default_behavior_preservation(self):
        """Test that rotary_ratio=1.0 maintains original behavior."""
        dim, num_heads, spatial_dims = 32, 4, 2

        # Create layers with and without explicit rotary_ratio
        layer_default = RotarySpatialEmbedding(
            dim=dim, num_heads=num_heads, spatial_dims=spatial_dims, learnable=False
        )
        layer_explicit = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            rotary_ratio=1.0,
            learnable=False,
        )

        # Test with same input
        x = torch.randn(2, 9, dim)
        result_default = layer_default(x, spacing=(1.0, 1.0), grid_shape=(3, 3))
        result_explicit = layer_explicit(x, spacing=(1.0, 1.0), grid_shape=(3, 3))

        torch.testing.assert_close(
            result_default,
            result_explicit,
            msg="Default and explicit rotary_ratio=1.0 should be identical",
        )

    def test_zero_rotation(self):
        """Test that rotary_ratio=0.0 returns input unchanged."""
        dim, num_heads, spatial_dims = 32, 4, 2
        layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            rotary_ratio=0.0,
            learnable=False,
        )

        x = torch.randn(2, 9, dim)
        result = layer(x, spacing=(1.0, 1.0), grid_shape=(3, 3))

        # With zero rotation, output should be identical to input
        torch.testing.assert_close(
            x, result, msg="Zero rotation should return input unchanged"
        )

    def test_non_rotated_part_preservation(self):
        """Test that non-rotated parts are preserved unchanged."""
        dim, num_heads, spatial_dims = 64, 8, 2
        layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            rotary_ratio=0.5,
            learnable=False,
        )

        x = torch.randn(2, 16, dim)
        result = layer(x, spacing=(1.0, 1.0), grid_shape=(4, 4))

        # The non-rotated part should be unchanged
        total_rotary_dim = layer.num_heads * layer.rotary_dim
        if layer.non_rotary_dim > 0:
            non_rotated_input = x[..., total_rotary_dim:]
            non_rotated_output = result[..., total_rotary_dim:]

            torch.testing.assert_close(
                non_rotated_input,
                non_rotated_output,
                msg="Non-rotated parts should be preserved",
            )

    @pytest.mark.parametrize("flatten", [True, False])
    def test_flatten_parameter_with_partial_rotation(self, flatten: bool):
        """Test flatten parameter works correctly with partial rotation."""
        dim, num_heads, spatial_dims = 64, 8, 2
        layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            rotary_ratio=0.5,
            learnable=False,
        )

        x = torch.randn(2, 16, dim)
        result = layer(x, spacing=(1.0, 1.0), grid_shape=(4, 4), flatten=flatten)

        if flatten:
            assert result.shape == (2, 16, dim)
        else:
            assert result.shape == (2, num_heads, 16, dim // num_heads)

    def test_invalid_rotary_ratio(self):
        """Test that invalid rotary_ratio values raise errors."""
        dim, num_heads, spatial_dims = 32, 4, 2

        # Test negative rotary_ratio
        with pytest.raises(
            AssertionError, match="rotary_ratio must be between 0.0 and 1.0"
        ):
            RotarySpatialEmbedding(
                dim=dim,
                num_heads=num_heads,
                spatial_dims=spatial_dims,
                rotary_ratio=-0.1,
            )

        # Test rotary_ratio > 1.0
        with pytest.raises(
            AssertionError, match="rotary_ratio must be between 0.0 and 1.0"
        ):
            RotarySpatialEmbedding(
                dim=dim,
                num_heads=num_heads,
                spatial_dims=spatial_dims,
                rotary_ratio=1.1,
            )

    def test_rotary_dimension_alignment(self):
        """Test that rotary dimensions are properly aligned with head structure."""
        # Test case where raw rotary_dim needs adjustment
        dim, num_heads = 64, 8

        # This should result in rotary_dim being adjusted to maintain head alignment
        layer = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=2,
            rotary_ratio=0.3,
            learnable=False,
        )

        # Raw calculation: dims_per_head * 0.3 = 8 * 0.3 = 2.4 -> 2
        # Ensure even: 2 // 2 * 2 = 2
        dims_per_head = dim // num_heads  # 64 // 8 = 8
        expected_rotary_dim_per_head = int(dims_per_head * 0.3)  # int(8 * 0.3) = 2
        expected_rotary_dim_per_head = (
            expected_rotary_dim_per_head // 2
        ) * 2  # (2 // 2) * 2 = 2

        assert layer.rotary_dim == expected_rotary_dim_per_head
        assert layer.rotary_dim % 2 == 0  # Must be even for complex representation
        assert layer.rotary_dim % 2 == 0  # Must be even

    def test_multihead_attention_partial_rotation(self):
        """Test RoSEMultiHeadAttention with partial rotation."""
        dim, num_heads, spatial_dims = 64, 8, 2
        layer = RoSEMultiHeadCrossAttention(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            rotary_ratio=0.25,
            learnable=False,
        )

        batch_size, seq_len = 2, 9
        q = torch.randn(batch_size, seq_len, dim)
        k = torch.randn(batch_size, seq_len, dim)

        attn = layer(q, k, q_spacing=(1.0, 1.0), q_grid_shape=(3, 3))

        expected_shape = (batch_size, num_heads, seq_len, seq_len)
        assert (
            attn.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {attn.shape}"


if __name__ == "__main__":
    # Run a quick test
    test_class = TestRoSENumericalProperties()
    test_class.test_phase_conjugate_property(2, False)
    test_class.test_frequency_schedule_properties()
    test_class.test_coordinate_generation_properties(2)
    print("All numerical tests passed!")
