# RoSE

## Original implementation of Rotary Spatial Embeddings (in PyTorch)

![License](https://img.shields.io/github/license/rhoadesScholar/RoSE)
[![CI/CD Pipeline](https://github.com/rhoadesScholar/RoSE/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rhoadesScholar/RoSE/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/rhoadesScholar/RoSE/branch/main/graph/badge.svg)](https://codecov.io/gh/rhoadesScholar/RoSE)
[![PyPI version](https://badge.fury.io/py/rose-spatial-embeddings.svg)](https://badge.fury.io/py/rose-spatial-embeddings)
[![Python versions](https://img.shields.io/pypi/pyversions/rose-spatial-embeddings.svg)](https://pypi.org/project/rose-spatial-embeddings/)


Rotary Spatial Embeddings (RoSE) extends 2D [Rotary Position Embeddings (RoPE)](https://arxiv.org/abs/2403.13298) to incorporate spatial information in terms of real world coordinates into the embeddings. This is particularly useful for tasks that require understanding of spatial relationships across different scales, such as in microscopy. Additionally, RoSE implements isotropic embeddings, providing rotation-equivariance across all dimensions.

# Rotation-equivariant Rotary Spatial Embeddings
### We extend 2-D Rotary Embeddings with a orthonormal frame.

### 1 Original 2-D RoPE

In two spatial dimensions the original Rotary Positional Embedding draws a single angle $ \theta \in [0, 2\pi) $ and forms the $2\times2$ rotation matrix

$$
R_\theta \text{ }=\text{ }
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta &  \cos\theta
\end{bmatrix}.
$$

For each exponentially-spaced magnitude $\mathrm{mag}_k$ it then stores, per axis,

$$
\bigl[f^{(x)}_k \text{ }\big|\text{ } f^{(y)}_k\bigr] \text{ }=\text{ }
\mathrm{mag}_k
\bigl[R_\theta^{\top}\bigr]_{0:2}
\text{ }=\text{ }
\mathrm{mag}_k
\text{ }[\text{ }\cos\theta,\text{ }-\sin\theta \text{ }\big|\text{ } \sin\theta,\text{ }\cos\theta\text{ }].
$$

When an $(x,y)$ coordinate is encountered at run time the phase for that frequency is

$$
\phi_k = x\text{ }f^{(x)}_k + y\text{ }f^{(y)}_k
       = \mathrm{mag}_k\bigl(x\cos\theta+y\sin\theta \text{ }\big|\text{ }
                                    -x\sin\theta+y\cos\theta\bigr),
$$

(i.e. the real/imaginary parts of $\mathrm{mag}_k\text{ }(x+iy)\text{ }e^{-i\theta}$).

Importantly, *no rotation is applied to the coordinates themselves; only the stored
frequency rows are pre-rotated at initialization with a uniformly distributed random angle*. These frequencies are then used to compute the phase at run time, and, optionally, can be learnable parameters.

---

### 2 Generalizing to **D** dimensions via a QR frame

In $D>2$ there is no single angle describing a rotation; instead we sample **one orthonormal matrix**

$$
R \text{ }\in\text{ } \mathrm{SO}(D)
\quad\text{(via QR decomposition, once per head).}
$$

Consider the first two columns

$$(v_0,\text{ }v_1) = (R_{\star,0},\text{ }R_{\star,1}).$$

They span a 2-D plane inside $\mathbb R^{D}$ and are orthonormal by construction,
perfectly mirroring the rôle of $(\cos\theta,\sin\theta)$ in the 2-D case.

For every spatial axis $i\in\{0,\dots,D-1\}$ we keep the *row* entries
$(v_{0,i},\text{ }v_{1,i})$:

$$
\text{real}_{i,k} \text{ }=\text{ } \mathrm{mag}_k\text{ }v_{0,i},
\quad
\text{imag}_{i,k} \text{ }=\text{ } \mathrm{mag}_k\text{ }v_{1,i}.
$$

The phase accumulated at run time is now

$$
\phi_k = \sum_{i=0}^{D-1} t_i
         \bigl(\text{real}_{i,k} \text{ }\big|\text{ } \text{imag}_{i,k}\bigr)
       = \mathrm{mag}_k
         \bigl(t\!\cdot\!v_0 \text{ }\big|\text{ } t\!\cdot\!v_1\bigr),
$$
with $t=(t_0,\dots,t_{D-1})$ the coordinate vector.  
Thus each frequency again represents the complex number  
$\mathrm{mag}_k\text{ }(t\cdot v_0 + i\text{ }t\cdot v_1)$ — **equivalent algebra** to the 2-D formula, just in a higher-dimensional plane.

At initialization, the orthonormal frame $R$ for each attention head is sampled from $\mathrm{SO}(D)$, which is a uniform distribution over all rotations in $D$ dimensions, similar to how the angle $\theta$ was sampled in the 2-D case. Again, the frequencies can be learnable parameters, allowing the model to adapt them during training.

---

### 3 Why use an orthonormal frame?

* **Isotropy without extra cost**  
  The axes of the data can be arbitrarily permuted or rotated; because the
  frequency lattice was sampled from $\mathrm{SO}(D)$, the attention mechanism
  sees no privileged “x-axis” or “y-axis”.  This removes an inductive bias that
  might otherwise hinder learning on
  volumetric data, point clouds, or molecular coordinates.

* **Head-level diversity**  
  Sampling a fresh $R$ for each attention head at initialization, and allowing the resulting frequencies to be learnable parameters, supplies
  *independent* 2-D sub-planes.
  Heads can therefore specialise in very different directional cues without
  any run-time overhead.

* **Exact backward-compatibility**  
  Setting `rotate=False` makes $R$ the identity; the recipe collapses to the
  classic axis-wise RoPE.

* **Still a single complex multiply**  
  Because every axis keeps just **two** frequency channels (real and imaginary),
  we retain the efficient `view_as_complex` implementation strategy—no need for
  extra tensor reshapes or larger hidden states.

---

### 4 Embedding real-world coordinates

In many applications, such as microscopy or 3D point clouds, the coordinates are not just indices but represent real-world positions that may contain useful spatial information. RoSE allows for injecting these coordinates directly into the rotary embeddings.

---

### 5 Conclusion

The QR‐based initialisation is a drop-in, mathematically faithful extension of the 2-D RoPE idea: one global plane per head, orthonormally embedded inside $\mathbb R^{D}$.  It keeps the computational footprint unchanged while gifting the model rotation-equivariance across all spatial dimensions. The additional ability to inject real-world coordinates makes RoSE particularly powerful for tasks that require understanding exact spatial relationships, such as in microscopy or 3D point clouds.

---

## Installation

### From PyPI

```bash
pip install rose-spatial-embeddings
```

### From source

```bash
pip install git+https://github.com/rhoadesScholar/RoSE.git
```

## Usage

```python
import torch
from RoSE import RoSELayer, RoSEMultiheadSelfAttention

# Basic RoSE layer for applying rotary spatial embeddings to q and k
layer = RoSELayer(dim=128, num_heads=8, spatial_dims=3, learnable=True)

batch_size, seq_len = 2, 1000
q = torch.randn(batch_size, seq_len, 128)
k = torch.randn(batch_size, seq_len, 128)

# Define spatial grid properties
grid_shape = (10, 10, 10)  # 3D grid dimensions
voxel_size = (1.0, 1.0, 1.0)  # Physical size of each voxel

# Apply rotary spatial embeddings
q_rot, k_rot = layer(q, k, grid_shape, voxel_size)

# Complete multihead self-attention with RoSE
mha = RoSEMultiheadSelfAttention(dim=128, num_heads=8, spatial_dims=3)
x = torch.randn(batch_size, seq_len, 128)
output = mha(x, grid_shape, voxel_size)
```


## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.