# RoSE N-dimensional Rotary Spatial Embeddings

## Original implementation of Rotary Spatial Embeddings (in PyTorch)

![GitHub - License](https://img.shields.io/github/license/rhoadesScholar/RoSE)
[![CI/CD Pipeline](https://github.com/rhoadesScholar/RoSE/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rhoadesScholar/RoSE/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/github/rhoadesScholar/RoSE/graph/badge.svg?token=PPT4ZNZZCJ)](https://codecov.io/github/rhoadesScholar/RoSE)
![PyPI - Version](https://img.shields.io/pypi/v/rotary-spatial-embeddings)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rotary-spatial-embeddings)


Rotary Spatial Embeddings (RoSE) extends [2D Rotary Position Embeddings (RoPE)](https://arxiv.org/abs/2403.13298) and the original [1D RoPE](https://arxiv.org/pdf/2104.09864) to incorporate into the embeddings spatial information in terms of N-dimensional real world coordinates. This is particularly useful for tasks that require understanding of spatial relationships across different scales, such as in microscopy.

## Explanation

### 1 Relative phase in 1-D RoPE

If you write the 1-D RoPE positional factor for token $t$ as a per-token complex phase

```math
\phi(t)=e^{\,i\,t\theta},\qquad t\in\mathbb Z .
```

After you attach that phase to query $q_t$ and key $k_t$,

```math
\tilde q_t = q_t\;\phi(t),\qquad
\tilde k_t = k_t\;\phi(t)^{*},
```

where $^*$ denotes complex conjugation, their dot-product inside attention becomes

```math
\tilde q_n\,\tilde k_m^{}
\;=\; q_n\,k_m^{}\,
\underbrace{\phi(n)\,\phi(m)^{*}}_{=\,e^{\,i\,(n-m)\theta}} .
```

⸻

### 2 Extending to N dimensions

Give every token a coordinate vector
$\mathbf{p}=(x,y,z,\dots)\in\mathbb R^{N}.$

Define its phase as

```math
\phi(\mathbf{p}) \;=\;e^{\,i\,\langle\mathbf{p},\,\boldsymbol\theta\rangle},
\qquad
\langle\mathbf{p},\boldsymbol\theta\rangle
=\sum_{a=1}^{N} p_a\,\theta_a .
```

Then

```math
\phi(\mathbf{p}_n)\,\phi(\mathbf{p}_m)^{*}
\;=\;
e^{\,i\,\langle\mathbf{p}_n-\mathbf{p}_m,\;\boldsymbol\theta\rangle},
```

which is the ND generalisation of the 1-D $e^{\,i\,(n-m)\theta}$.
You still get

```math
A_{nm}\;=\;\mathrm{Re}
\bigl[q_n k_m^{*}\;e^{\,i\,\langle\mathbf{p}_n-\mathbf{p}_m,
\boldsymbol\theta\rangle}\bigr],
```

while keeping the per-token encoding cost $O(LD)$.

**Partial Rotation**: RoSE also supports partial rotation via the `rotary_ratio` parameter, where only a fraction of the embedding dimensions are rotated while the rest are passed through unchanged. This provides a balance between spatial awareness and computational efficiency.

---

### 3 Embedding real-world coordinates

In many applications, such as microscopy or 3D point clouds, the coordinates are not just indices but represent real-world positions that may contain useful spatial information. RoSE allows for injecting these coordinates directly into the rotary embeddings by simply multiplying the coordinate vectors by the coordinate spacing (i.e. voxel size) before applying the rotary embedding.

---

## Installation

### From PyPI

```bash
pip install rotary-spatial-embeddings
```

### From source

```bash
pip install git+https://github.com/rhoadesScholar/RoSE.git
```

## Usage

### Basic Usage - Multi-Head Attention with Spatial Embeddings

```python
import torch
from RoSE import RoSEMultiHeadCrossAttention

# Create RoSE multi-head attention layer
layer = RoSEMultiHeadCrossAttention(
    feature_dims=128,
    num_heads=8,
    spatial_dims=3,
    learnable=True,
    base_theta=1e4,
    rotary_ratio=1.0,  # Apply rotation to all dimensions (default)
    learnable_scale=False  # Set to True for adaptive spatial scaling
)

batch_size, seq_len = 2, 1000
q = torch.randn(batch_size, seq_len, 128)
k = torch.randn(batch_size, seq_len, 128)

# Define spatial grid properties
grid_shape = (10, 10, 10)  # 3D grid dimensions
spacing = (1.0, 1.0, 1.0)  # Physical size of each voxel

# Compute attention scores with spatial embeddings
attn_scores = layer(q, k, spacing, grid_shape)  # Shape: (batch_size, num_heads, seq_len, seq_len)
```

### Partial Rotation with `rotary_ratio`

The `rotary_ratio` parameter allows you to apply rotary embeddings to only a fraction of the embedding dimensions, which can be beneficial for performance and model capacity:

```python
import torch
from RoSE import RotarySpatialEmbedding

# Apply rotation to only 50% of the embedding dimensions
embedding = RotarySpatialEmbedding(
    feature_dims=128,
    num_heads=8,
    spatial_dims=2,
    rotary_ratio=0.5,  # Only rotate first 50% of dimensions per head
    learnable=False
)

batch_size, seq_len = 2, 100
x = torch.randn(batch_size, seq_len, 128)

# The first 64 dimensions (50% of 128) will be rotated
# The last 64 dimensions will be passed through unchanged
x_embedded = embedding(x, spacing=(0.5, 0.5), grid_shape=(10, 10))
```

**Key benefits of partial rotation:**

- **Performance**: Reduces computational cost for large embeddings
- **Flexibility**: Allows some dimensions to encode non-spatial information
- **Stability**: Can improve training stability in some scenarios
- **Memory**: Lower memory usage for frequency parameters

### Using Just the Embedding Layer

```python
import torch
from RoSE import RotarySpatialEmbedding

# Create just the rotary spatial embedding layer
embedding = RotarySpatialEmbedding(
    feature_dims=128,
    num_heads=8,
    spatial_dims=2,
    learnable=False,
    frequency_scaling="sqrt",
    rotary_ratio=1.0,  # Apply rotation to all dimensions (default)
    learnable_scale=False  # Set to True for adaptive spatial scaling
)

batch_size, seq_len = 2, 100
x = torch.randn(batch_size, seq_len, 128)

# Define 2D grid
grid_shape = (10, 10)
spacing = (0.5, 0.5)

# Apply rotary spatial embeddings
x_embedded = embedding(x, spacing, grid_shape)  # Shape: (batch_size, seq_len, 128)
```

## Parameters

### Core Parameters

- **`feature_dims`**: Total embedding dimension (must be even and divisible by `num_heads`)
- **`num_heads`**: Number of attention heads
- **`spatial_dims`**: Number of spatial dimensions (2 for 2D, 3 for 3D, etc.)
- **`rotary_ratio`**: Fraction of embedding dimensions to apply rotation to (0.0 to 1.0, default: 1.0)
  - `1.0`: Apply rotation to all dimensions (full rotation)
  - `0.5`: Apply rotation to 50% of dimensions per head
  - `0.0`: No rotation applied (passthrough)

### Advanced Parameters

- **`base_theta`**: Base frequency for rotary embeddings (default: 10000.0)
- **`learnable`**: Whether frequencies should be learnable parameters (default: True)
- **`init_jitter_std`**: Standard deviation for frequency initialization jitter (default: 0.02)
- **`frequency_scaling`**: Scaling strategy for frequencies (default: "sqrt")
  - `"none"`: No frequency scaling
  - `"linear"`: Linear scaling with spatial dimensions
  - `"sqrt"`: Square root scaling with spatial dimensions
  - `"adaptive"`: Adaptive scaling based on spatial dims and embedding dim
- **`learnable_scale`**: Enable learnable spatial scale transformation (default: False)
  - When `True`, adds learnable parameters that transform spatial coordinates
  - Uses equation: `scale = a * scale ** b + c * log(scale / d)`
  - Helps handle rotations across different spatial scales automatically
- **`initial_scaling`**: Initial scaling transformation mode (default: None)
  - Works with both `learnable_scale=True` (learnable parameters) and `learnable_scale=False` (fixed parameters)
  - `"log"`: Initialize for pure logarithmic scaling (`a=0, b=c=d=1`)
  - `"rope"`: Initialize to nullify scaling for standard RoPE behavior (`a=b=c=0, d=1`)
  - `"identity"`, `"linear"`, `"power"`, or `None`: Initialize for identity/power scaling (`a=b=d=1, c=0`)
  - Useful for data with large coordinate ranges, standard RoPE compatibility, or specific scaling relationships

## Advanced Examples

### Working with 3D Medical Imaging Data

```python
import torch
from RoSE import RotarySpatialEmbedding

# Example: 3D CT scan with anisotropic voxel spacing
batch_size, seq_len = 1, 8000  # 20x20x20 volume flattened
embedding_dim = 256
num_heads = 8

# Create embedding layer for 3D medical data
embedding = RotarySpatialEmbedding(
    feature_dims=embedding_dim,
    num_heads=num_heads,
    spatial_dims=3,
    learnable=True,
    rotary_ratio=0.75,  # Rotate 75% of dimensions
    frequency_scaling="adaptive"
)

# Define anisotropic voxel spacing (common in medical imaging)
grid_shape = (20, 20, 20)
voxel_spacing = (0.5, 0.5, 2.0)  # 0.5mm x 0.5mm x 2mm

# Your input features (e.g., from a CNN backbone)
x = torch.randn(batch_size, seq_len, embedding_dim)

# Apply spatial embeddings
x_with_spatial = embedding(x, voxel_spacing, grid_shape)
print(f"Input shape: {x.shape}")
print(f"Output shape: {x_with_spatial.shape}")
```

### Multi-Scale Microscopy Analysis

```python
import torch
from RoSE import RoSEMultiHeadCrossAttention

# Example: Multi-scale microscopy with different zoom levels
def create_multiscale_attention():
    return RoSEMultiHeadCrossAttention(
        feature_dims=512,
        num_heads=16,
        spatial_dims=2,
        learnable=True,
        base_theta=1e4,
        rotary_ratio=1.0  # Full rotation for spatial awareness
    )

# Different scales: 10x, 40x, 100x magnification
scales_and_spacings = [
    ((100, 100), (1.0, 1.0)),      # 10x: 1μm/pixel
    ((200, 200), (0.25, 0.25)),    # 40x: 0.25μm/pixel
    ((400, 400), (0.1, 0.1)),      # 100x: 0.1μm/pixel
]

attention_layer = create_multiscale_attention()

for i, (grid_shape, spacing) in enumerate(scales_and_spacings):
    seq_len = grid_shape[0] * grid_shape[1]

    # Simulate features from different magnifications
    q = torch.randn(1, seq_len, 512)
    k = torch.randn(1, seq_len, 512)

    # Compute attention with spatial awareness
    attn_scores = attention_layer(q, k, spacing, grid_shape)

    print(f"Scale {i+1}: {grid_shape} grid, {spacing} spacing")
    print(f"Attention shape: {attn_scores.shape}\n")
```

### Custom Coordinate Systems

```python
import torch
from RoSE import RotarySpatialEmbedding

# Example: Geographic coordinate system (lat/lon/elevation)
class GeospatialEmbedding(torch.nn.Module):
    def __init__(self, feature_dims, num_heads):
        super().__init__()
        self.spatial_embedding = RotarySpatialEmbedding(
            feature_dims=feature_dims,
            num_heads=num_heads,
            spatial_dims=3,  # lat, lon, elevation
            learnable=True,
            frequency_scaling="adaptive"
        )

    def forward(self, x, coordinates):
        """
        Args:
            x: Features [B, N, D]
            coordinates: [B, N, 3] tensor with [lat, lon, elevation]
        """
        # Normalize coordinates to reasonable scales
        lat_scale, lon_scale, elev_scale = 1/90, 1/180, 1/1000
        normalized_coords = coordinates * torch.tensor([lat_scale, lon_scale, elev_scale])

        # Convert to grid format (this is a simplified example)
        # In practice, you'd need proper coordinate-to-grid mapping
        batch_size, seq_len, _ = coordinates.shape
        grid_size = int(seq_len ** (1/3)) if seq_len ** (1/3) == int(seq_len ** (1/3)) else 10
        grid_shape = (grid_size, grid_size, grid_size)
        spacing = (lat_scale, lon_scale, elev_scale)

        return self.spatial_embedding(x, spacing, grid_shape)

# Usage
geo_embedding = GeospatialEmbedding(feature_dims=256, num_heads=8)
features = torch.randn(2, 1000, 256)
coordinates = torch.randn(2, 1000, 3)  # Random lat/lon/elevation
result = geo_embedding(features, coordinates)
```

### Learnable Spatial Scale Adaptation

The `learnable_scale` parameter enables automatic adaptation to different spatial scales during training, which is particularly useful for multi-resolution data or when the optimal spatial scaling is unknown:

```python
import torch
from RoSE import RotarySpatialEmbedding

# Example: Multi-resolution medical imaging with learnable scaling
def create_adaptive_embedding():
    return RotarySpatialEmbedding(
        feature_dims=128,
        num_heads=8,
        spatial_dims=3,
        learnable=True,
        learnable_scale=True,  # Enable learnable spatial scaling
        initial_scaling=None,  # Use default initialization (identity transform)
        frequency_scaling="adaptive"
    )

# Create embeddings for different modalities/scales
ct_embedding = create_adaptive_embedding()  # CT scan data
mri_embedding = create_adaptive_embedding()  # MRI data

# Different spatial resolutions and voxel sizes
ct_data = torch.randn(1, 1000, 128)  # Lower resolution CT
mri_data = torch.randn(1, 8000, 128)  # Higher resolution MRI

ct_spacing = (1.0, 1.0, 3.0)    # 1mm x 1mm x 3mm
mri_spacing = (0.5, 0.5, 0.5)   # 0.5mm isotropic

ct_grid = (10, 10, 10)
mri_grid = (20, 20, 20)

# The learnable scaling will adapt to each modality's characteristics
ct_result = ct_embedding(ct_data, ct_spacing, ct_grid)
mri_result = mri_embedding(mri_data, mri_spacing, mri_grid)

print("Learnable parameters adapt automatically to spatial characteristics")
print(f"CT embedding scale parameters: a={ct_embedding.scale_a.data}, b={ct_embedding.scale_b.data}")
print(f"MRI embedding scale parameters: a={mri_embedding.scale_a.data}, b={mri_embedding.scale_b.data}")
```

### Logarithmic Scaling for Large Dynamic Ranges

For data with large spatial scale variations, you can use logarithmic scaling initialization:

```python
import torch
from RoSE import RotarySpatialEmbedding

# Example: Geographic data with large coordinate ranges
geo_embedding = RotarySpatialEmbedding(
    feature_dims=256,
    num_heads=16,
    spatial_dims=2,
    learnable=True,
    learnable_scale=True,  # Enable learnable scaling
    initial_scaling="log", # Initialize for logarithmic scaling
    frequency_scaling="adaptive"
)

# Geographic coordinates can span large ranges
batch_size, num_locations = 4, 2500
features = torch.randn(batch_size, num_locations, 256)

# Large coordinate ranges (e.g., global coordinates in meters)
grid_shape = (50, 50)
spacing = (1000.0, 1000.0)  # 1km spacing

# The log initialization helps handle large coordinate values
result = geo_embedding(features, spacing, grid_shape)

print("Log-scale initialization parameters:")
print(f"a={geo_embedding.scale_a.data} (should be ~0)")
print(f"b={geo_embedding.scale_b.data} (should be ~1)")
print(f"c={geo_embedding.scale_c.data} (should be ~1)")
print(f"d={geo_embedding.scale_d.data} (should be ~1)")
print("Equation: scale = a * scale^b + c * log(scale/d)")
```

### Fixed Logarithmic Scaling

You can also use logarithmic scaling with fixed (non-learnable) parameters, which is useful when you know logarithmic scaling is appropriate but don't want the computational overhead of learnable parameters:

```python
import torch
from RoSE import RotarySpatialEmbedding

# Example: Fixed log scaling for geographic data
fixed_log_embedding = RotarySpatialEmbedding(
    feature_dims=128,
    num_heads=8,
    spatial_dims=2,
    learnable=False,
    learnable_scale=False,  # Fixed (non-learnable) parameters
    initial_scaling="log", # But still apply log scaling transformation
    frequency_scaling="adaptive"
)

# The parameters are fixed at: a=0, b=c=d=1
# Equation becomes: scale = 0 * scale^1 + 1 * log(scale/1) = log(scale)

# Test with large coordinate values
batch_size, num_locations = 2, 1600
features = torch.randn(batch_size, num_locations, 128)

# Large coordinate ranges (e.g., geographic coordinates in meters)
grid_shape = (40, 40)
spacing = (1000.0, 1500.0)  # 1km x 1.5km spacing

# Fixed log scaling handles large values without learnable overhead
result = fixed_log_embedding(features, spacing, grid_shape)

print("Fixed log scaling - no learnable parameters, but applies log transformation")
print(f"Output shape: {result.shape}")

# Compare with normal fixed scaling
normal_embedding = RotarySpatialEmbedding(
    feature_dims=128, num_heads=8, spatial_dims=2,
    learnable_scale=False, initial_scaling=None
)

normal_result = normal_embedding(features, spacing, grid_shape)

# Results should be different due to log scaling
print(f"Results differ: {not torch.allclose(result, normal_result, atol=1e-6)}")
```

### Standard RoPE Compatibility Mode

For compatibility with standard RoPE implementations or when you want to disable spatial scaling entirely:

```python
import torch
from RoSE import RotarySpatialEmbedding

# RoPE compatibility mode - nullifies all spatial scaling
rope_embedding = RotarySpatialEmbedding(
    feature_dims=128,
    num_heads=8,
    spatial_dims=2,
    learnable=True,
    learnable_scale=True,
    initial_scaling="rope",  # Nullify scaling to reproduce standard RoPE
    frequency_scaling="sqrt"
)

# The scaling equation becomes: scale = 0 * scale^0 + 0 * log(scale/1) = 0
# This effectively disables spatial scaling, making it behave like standard RoPE

batch_size, seq_len = 2, 100
features = torch.randn(batch_size, seq_len, 128)
grid_shape = (10, 10)
spacing = (1.0, 1.0)

result = rope_embedding(features, spacing, grid_shape)

print(f"RoPE mode scaling parameters:")
print(f"a={rope_embedding.scale_a.data} (should be 0)")
print(f"b={rope_embedding.scale_b.data} (should be 0)")
print(f"c={rope_embedding.scale_c.data} (should be 0)")
print(f"d={rope_embedding.scale_d.data} (should be 1)")
print("Spatial scaling is effectively disabled")
```

### Comparing Fixed vs. Learnable Scaling

```python
import torch
from RoSE import RotarySpatialEmbedding

# Create embeddings with and without learnable scaling
fixed_embedding = RotarySpatialEmbedding(
    feature_dims=128, num_heads=8, spatial_dims=2,
    learnable_scale=False  # Traditional fixed scaling
)

adaptive_embedding = RotarySpatialEmbedding(
    feature_dims=128, num_heads=8, spatial_dims=2,
    learnable_scale=True   # Learnable adaptive scaling
)

# Test with challenging multi-scale data
x = torch.randn(2, 100, 128)
fine_spacing = (0.1, 0.1)      # Very fine spatial resolution
coarse_spacing = (10.0, 10.0)  # Very coarse spatial resolution
grid_shape = (10, 10)

# Fixed scaling may struggle with large scale differences
fine_fixed = fixed_embedding(x, fine_spacing, grid_shape)
coarse_fixed = fixed_embedding(x, coarse_spacing, grid_shape)

# Adaptive scaling learns to handle both scales
fine_adaptive = adaptive_embedding(x, fine_spacing, grid_shape)
coarse_adaptive = adaptive_embedding(x, coarse_spacing, grid_shape)

print("Fixed scaling treats all scales the same")
print("Adaptive scaling learns optimal transformation for each scale during training")
```

### Integration with Transformers

```python
import torch
import torch.nn as nn
from RoSE import RotarySpatialEmbedding

class SpatialTransformerBlock(nn.Module):
    """Transformer block with spatial awareness via RoSE."""

    def __init__(self, feature_dims, num_heads, spatial_dims=2):
        super().__init__()
        self.spatial_embedding = RotarySpatialEmbedding(
            feature_dims=feature_dims,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dims,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x, spacing, grid_shape):
        # Apply spatial embeddings
        x_spatial = self.spatial_embedding(x, spacing, grid_shape)

        # Self-attention with spatial embeddings
        attn_out, _ = self.attention(x_spatial, x_spatial, x_spatial)
        x = self.norm1(x + attn_out)

        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)

        return x

# Example usage
transformer = SpatialTransformerBlock(feature_dims=256, num_heads=8, spatial_dims=2)
x = torch.randn(4, 100, 256)  # Batch of sequences
result = transformer(x, spacing=(1.0, 1.0), grid_shape=(10, 10))
print(f"Transformer output shape: {result.shape}")
```

## Tips and Best Practices

1. **Voxel Spacing**: Always provide real-world spacing when available - it significantly improves spatial understanding
2. **Rotary Ratio**: Start with `rotary_ratio=1.0` for maximum spatial awareness, then experiment with lower values for efficiency
3. **Learnable Frequencies**: Set `learnable=True` for fine-tuning on your specific spatial domain
4. **Frequency Scaling**: Use `"adaptive"` scaling for most applications, `"sqrt"` for simpler cases
5. **Grid Shape**: Ensure your sequence length matches `prod(grid_shape)` for proper spatial mapping
6. **Learnable Scaling**: Enable `learnable_scale=True` when working with:
   - Multi-resolution or multi-scale data
   - Unknown optimal spatial scaling
   - Data with varying spatial characteristics
   - Large differences in coordinate ranges
7. **Initial Scaling**: Choose the appropriate mode:
   - `initial_scaling="log"` for geographic or astronomical data with large coordinate ranges
   - `initial_scaling="rope"` to reproduce standard RoPE behavior (nullifies spatial scaling)
   - `initial_scaling="identity"` or `None` for general-purpose identity/power scaling
   - `initial_scaling="linear"` or `"power"` for identity/power scaling (same as identity)
   - Can be used with either learnable (`learnable_scale=True`) or fixed (`learnable_scale=False`) parameters
8. **Performance**: Learnable scaling adds 4 parameters per spatial dimension but can significantly improve model performance on spatially diverse data

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
