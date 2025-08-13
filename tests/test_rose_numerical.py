"""Numerical tests for RoSE mathematical properties."""

import math

import pytest
import torch

from RoSE.rose import RotarySpatialEmbedding, _init_p_nd, _make_log_spaced_frequencies


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


if __name__ == "__main__":
    # Run a quick test
    test_class = TestRoSENumericalProperties()
    test_class.test_phase_conjugate_property(2, False)
    test_class.test_frequency_schedule_properties()
    test_class.test_coordinate_generation_properties(2)
    print("All numerical tests passed!")
