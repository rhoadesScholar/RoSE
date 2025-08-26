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
    dim=128,
    num_heads=8,
    spatial_dims=3,
    learnable=True,
    base_theta=1e4,
    rotary_ratio=1.0  # Apply rotation to all dimensions (default)
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
    dim=128,
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
    dim=128,
    num_heads=8,
    spatial_dims=2,
    learnable=False,
    frequency_scaling="sqrt",
    rotary_ratio=1.0  # Apply rotation to all dimensions (default)
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

- **`dim`**: Total embedding dimension (must be even and divisible by `num_heads`)
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
    dim=embedding_dim,
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
        dim=512,
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
    def __init__(self, dim, num_heads):
        super().__init__()
        self.spatial_embedding = RotarySpatialEmbedding(
            dim=dim,
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
geo_embedding = GeospatialEmbedding(dim=256, num_heads=8)
features = torch.randn(2, 1000, 256)
coordinates = torch.randn(2, 1000, 3)  # Random lat/lon/elevation
result = geo_embedding(features, coordinates)
```

### Integration with Transformers

```python
import torch
import torch.nn as nn
from RoSE import RotarySpatialEmbedding

class SpatialTransformerBlock(nn.Module):
    """Transformer block with spatial awareness via RoSE."""

    def __init__(self, dim, num_heads, spatial_dims=2):
        super().__init__()
        self.spatial_embedding = RotarySpatialEmbedding(
            dim=dim,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            learnable=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
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
transformer = SpatialTransformerBlock(dim=256, num_heads=8, spatial_dims=2)
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

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
