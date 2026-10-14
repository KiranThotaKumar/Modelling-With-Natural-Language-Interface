# Mathematical & Structural Architecture Specifications: Pure NumPy Seq2Seq Physics Intent Parser

This manual provides a mathematically precise specification of the custom, from-scratch NumPy Transformer model optimized for compiling raw human-language text strings into structured physics-intent representations without depending on high-level deep learning frameworks.

---

## Table of Contents

- [1. Global Architectural Geometry & Dimensions Blueprint](#1-global-architectural-geometry--dimensions-blueprint)
  - [1.1 Structural Hyperparameter Array Matrix](#11-structural-hyperparameter-array-matrix)
  - [1.2 Target Vocabulary Invariants & Dimension Spaces](#12-target-vocabulary-invariants--dimension-spaces)
- [2. Core Component Equations & Mathematical Mapping](#2-core-component-equations--mathematical-mapping)
  - [2.1 Static Geometric Waveform Positional Encodings](#21-static-geometric-waveform-positional-encodings-d_model--2--1-resilience-slicing)
  - [2.2 Multi-Dimensional Tensor Head Partitioning](#22-multi-dimensional-tensor-head-partitioning-_split_heads---_combine_heads-forward-transforms)
  - [2.3 Scaled Dot-Product Core Softmax Invariants & Numerical Stability Bounds](#23-scaled-dot-product-core-softmax-invariants--numerical-stability-bounds)
  - [2.4 Feature-Axis Layer Normalization Reduction Contractions](#24-feature-axis-layer-normalization-reduction-contractions)
- [3. Micro-Level Vectorized Backpropagation Analytical Derivatives](#3-micro-level-vectorized-backpropagation-analytical-derivatives)
  - [3.1 Layer Normalization Multi-Axis Error Transpositions](#31-layer-normalization-multi-axis-error-transpositions)
  - [3.2 Causal Mask Isolated Softmax Derivative Chain Rule](#32-causal-mask-isolated-softmax-derivative-chain-rule)
  - [3.3 Flattened 2D Indexing Shortcut for Masked Categorical Cross-Entropy Logit Gradients](#33-flattened-2d-indexing-shortcut-for-masked-categorical-cross-entropy-logit-gradients)
  - [3.4 Asymmetric Cross-Attention Path-Gradient Routing](#34-asymmetric-cross-attention-path-gradient-routing-anti-scrambling-controls)
  - [3.5 Atomic Shared Index Vector Accumulation](#35-atomic-shared-index-vector-accumulation-npaddat-invariants)
- [4. Decoupled First- and Second-Moment Adam Optimization Architecture](#4-decoupled-first--and-second-moment-adam-optimization-architecture)
  - [4.1 Running Averages & Variable Time-Step Clock Counters](#41-running-averages--variable-time-step-clock-counters)
  - [4.2 Initialization Moment Bias Neutralization Matrix Equations](#42-initialization-moment-bias-neutralization-matrix-equations)
  - [4.3 Warm-Up Step Cosine Learning Rate Decay Mechanics](#43-warm-up-step-cosine-learning-rate-decay-mechanics)
  - [4.4 Parameter Memory Map Isolation](#44-parameter-memory-map-isolation-isolated-encoder-decoder-and-unified-output-head-state-dictionaries)
- [5. Network Parameter Weight Invariant Serialization Layers](#5-network-parameter-weight-invariant-serialization-layers)
  - [5.1 Sequential Array Packaging & Suffix-Key Flattening Structures](#51-sequential-array-packaging--suffix-key-flattening-structures)
  - [5.2 Decoupled Native Array Manifold Restoration Mechanisms](#52-decoupled-native-array-manifold-restoration-mechanisms)
- [6. End-to-End System Tensor Invariant Geometry Flowchart](#6-end-to-end-system-tensor-invariant-geometry-flowchart)

---

## 1. Global Architectural Geometry & Dimensions Blueprint

### 1.1 Structural Hyperparameter Array Matrix

The architectural baseline of the Sequence-to-Sequence (Seq2Seq) Transformer is constructed completely using vectorized NumPy array allocations without dependence on high-level deep learning frameworks. Every tensor shape is explicitly encoded as a static structural variable.

| Parameter | Operational Definition | Static Field Coordinate Assignment |
| :--- | :--- | :--- |
| $d_{\text{model}}$ | Continuous hidden-layer coordinate representation dimension | **64** |
| $n_{\text{heads}}$ | Parallel multi-head attention processing streams | **4** |
| $d_{\text{depth}}$ | Individual head projection track slicing dimension ($d_{\text{model}} / n_{\text{heads}}$) | **16** |
| $d_{\text{ff}}$ | Position-wise feed-forward network hidden expansion width | **256** |
| $N_{\text{enc}}$ | Cascaded Transformer encoder layer blocks | **2** |
| $N_{\text{dec}}$ | Cascaded Transformer decoder layer blocks | **2** |
| $L_{\text{max}}$ | Hard sequence context mask and truncation boundary | **25** |

### 1.2 Target Vocabulary Invariants & Dimension Spaces

The mathematical projection layer links token-tracking maps across an engineered vocabulary domain:

- **Unique computational vocabulary size ($V_{\text{size}}$):** **877 clean structural words**
- **Global embedding matrix mapping boundary:** $(V_{\text{size}} \times d_{\text{model}}) \longrightarrow (877 \times 64)$

---

## 2. Core Component Equations & Mathematical Mapping

### 2.1 Static Geometric Waveform Positional Encodings (`d_model % 2 == 1` Resilience Slicing)

Spatial positions are mapped directly to coordinate wave vectors using alternating trigonometric frequencies. For token position index $\text{pos} \in [0, L_{\text{max}}-1]$ and channel axis index $i$, the positional encoding is defined as:

$$
PE_{(\text{pos}, 2i)} = \sin\left(\text{pos} \cdot \exp\left(-2i \cdot \frac{\ln(10000.0)}{d_{\text{model}}}\right)\right)
$$

$$
PE_{(\text{pos}, 2i+1)} = \cos\left(\text{pos} \cdot \exp\left(-2i \cdot \frac{\ln(10000.0)}{d_{\text{model}}}\right)\right)
$$

To handle cases where the embedding dimension is odd ($d_{\text{model}} \% 2 == 1$), the division array is dynamically clipped via matrix slicing:

```python
if d_model % 2 == 1:
    pe[:, 1::2] = np.cos(position * div_term[:, :-1] if div_term.shape[0] * 2 > d_model else position * div_term)
```

This preserves structural broadcasting integrity during matrix addition.

### 2.2 Multi-Dimensional Tensor Head Partitioning (`_split_heads` / `_combine_heads` Forward Transforms)

Before self-attention is applied, projected state matrices are distributed across independent processing tracks. The geometry transforms across axes according to the following layout:

```text
_split_heads:
(Batch, SeqLen, d_model) -> (Batch, SeqLen, n_heads, d_depth) -> (Batch, n_heads, SeqLen, d_depth)

_combine_heads:
(Batch, n_heads, SeqLen, d_depth) -> (Batch, SeqLen, n_heads, d_depth) -> (Batch, SeqLen, d_model)
```

### 2.3 Scaled Dot-Product Core Softmax Invariants & Numerical Stability Bounds

Given query, key, and value tracking manifolds assigned to 4D track blocks, the execution equation is:

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{depth}}}} + (M_{\text{causal}} \cdot -10^9)\right)V
$$

Where $M_{\text{causal}}$ is the upper-triangular look-ahead structural masking matrix penalizing future token coordinates. Row-wise numerical overflow is mitigated by isolating the maximum value per row before exponentiation:

$$
\text{Softmax}(Z_i) = \frac{\exp(Z_i - \max(Z))}{\sum_j \exp(Z_j - \max(Z))}
$$

### 2.4 Feature-Axis Layer Normalization Reduction Contractions

To limit shifting statistics across hidden representations, values are normalized across the final feature dimension vector:

$$
\text{LN}(x) = \gamma \odot \left(\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\right) + \beta
$$

$$
\mu = \frac{1}{d_{\text{model}}} \sum_{k=1}^{d_{\text{model}}} x_k,
\quad
\sigma^2 = \frac{1}{d_{\text{model}}} \sum_{k=1}^{d_{\text{model}}} (x_k - \mu)^2,
\quad
\epsilon = 10^{-6}
$$

---

## 3. Micro-Level Vectorized Backpropagation Analytical Derivatives

### 3.1 Layer Normalization Multi-Axis Error Transpositions

Let `x_norm = (x - mean) * inv_std`, where `inv_std = 1.0 / sqrt(variance + epsilon)`. The analytical mathematical gradients collapse non-feature tracking fields across multi-dimensional blocks while retaining the feature dimension for normalization.

```text
d_beta = sum(d_out, axis=reduce_axes)

d_gamma = sum(d_out * x_norm, axis=reduce_axes)

d_variance = sum(d_x_normalized * x_centered, axis=-1) * -0.5 * (inv_std ^ 3)

d_mean = sum(d_x_normalized * -inv_std, axis=-1) + d_variance * sum(-2.0 * x_centered, axis=-1) / d_model

dx = d_x_normalized * inv_std + d_variance * 2.0 * x_centered / d_model + d_mean / d_model
```

### 3.2 Causal Mask Isolated Softmax Derivative Chain Rule

When backpropagating through attention weights, the derivative is computed directly over the tracking matrix:

$$
\partial Z_{\text{logits}} = \left(\partial A_{\text{out}} \odot A_{\text{weights}}\right) - A_{\text{weights}} \odot \sum_{\text{row}} \left(\partial A_{\text{out}} \odot A_{\text{weights}}\right)
$$

Gradient leakage past causal masking coordinates is explicitly isolated by multiplying element-wise with an indicator tracking step:

$$
\partial Z_{\text{logits}} = \partial Z_{\text{logits}} \odot \mathbb{I}(M_{\text{causal}} < 0.5)
$$

### 3.3 Flattened 2D Indexing Shortcut for Masked Categorical Cross-Entropy Logit Gradients

To bypass expensive computational trees, the root sequence error derivative at the output classification layer is flattened directly into 2D memory arrays before backpropagation:

```text
dL/dZ = ((P - Y) ⊙ M) / N
```

Where:

- `P` represents the post-softmax prediction probabilities.
- `Y` is the ground-truth target label index representation (one-hot or properly broadcasted).
- `M` is the active sequence loss filtering padding mask (`loss_mask`): `1` for real tokens and `0` for padding.
- `N` represents the scalar tracking boundary count of non-padded entries (`active_tokens`) used to average the gradients.

A NumPy implementation note: if `probs` has shape `(B, L, V)` and `targets` are indices `(B, L)`, compute `one_hot = np.eye(V)[targets]` and then:

```python
grad = (probs - one_hot) * loss_mask[..., None] / max(active_tokens, 1)
```

### 3.4 Asymmetric Cross-Attention Path-Gradient Routing (Anti-Scrambling Controls)

Cross-attention gradients are routed with explicit asymmetry to preserve decoder-to-encoder dependency structure while preventing accidental scrambling of source-target attention paths.

### 3.5 Atomic Shared Index Vector Accumulation (`np.add.at` Invariants)

Gradient aggregation over repeated token indices must be performed with atomic updates to avoid silent overwrites during sparse accumulation. In NumPy this is implemented using `np.add.at`, which guarantees correct reduction semantics for shared indices.

---

## 4. Decoupled First- and Second-Moment Adam Optimization Architecture

### 4.1 Running Averages & Variable Time-Step Clock Counters

Parameter tracking maintains running momentum metrics across discrete optimization intervals $t$:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t,
\quad
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

With static hyperparameters:

- $\beta_1 = 0.9$
- $\beta_2 = 0.999$

### 4.2 Initialization Moment Bias Neutralization Matrix Equations

To neutralize first- and second-moment initialization bias during early training steps, tracking scales are corrected via:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t},
\quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

### 4.3 Warm-Up Step Cosine Learning Rate Decay Mechanics

The real-time operational learning rate $\alpha_t$ decays smoothly across training epochs following a cosine trajectory boundary:

$$
\alpha_t = \alpha_{\text{min}} + \frac{1}{2}(\alpha_{\text{base}} - \alpha_{\text{min}})\left(1 + \cos\left(\pi \cdot \frac{\text{step}_t}{\text{step}_{\text{total}}}\right)\right)
$$

With assignments:

- $\alpha_{\text{base}} = 0.002$
- $\alpha_{\text{min}} = 0.0001$

### 4.4 Parameter Memory Map Isolation (Isolated Encoder, Decoder, and Unified Output Head State Dictionaries)

The Adam optimizer contains isolated storage maps:

- `self.m_enc`, `self.v_enc`
- `self.m_dec`, `self.v_dec`
- `self.m_out`, `self.v_out`

These structures are keyed precisely by layer and parameter name, which keeps running statistics decoupled for encoder weights, decoder weights, and output head weights.

---

## 5. Network Parameter Weight Invariant Serialization Layers

### 5.1 Sequential Array Packaging & Suffix-Key Flattening Structures

Weight state tracking flattens multidimensional matrix entities into indexed dictionaries packed using custom layer string tags before exporting to disk.

- Encoder layers use keys like `f'enc_{layer_index}_Wq'`, `f'enc_{layer_index}_ffn_W1'`, etc.
- Decoder layers use keys like `f'dec_{layer_index}_Wq1'`, `f'dec_{layer_index}_Wq2'`, etc.
- Output prediction head parameters are isolated under `'W_vocab'` and `'b_vocab'`

### 5.2 Decoupled Native Array Manifold Restoration Mechanisms

During initialization recovery, `load_transformer_weights` streams the `.npz` archive directly into the model structures. This mirrors the shapes of the layer weight spaces and injects saved configuration into the active parameter map without requiring a framework-specific state object.

---

## 6. End-to-End System Tensor Invariant Geometry Flowchart

```text
       [Source Token Matrix (B, L)]            [Shifted Target Matrix (B, L)]
                    │                                         │
                    ▼                                         ▼
          [Embedding Row Lookup]                    [Embedding Row Lookup]
                    │                                         │
                    ▼                                         ▼
          [Wave PE Injection Matrix]                [Wave PE Injection Matrix]
                    │                                         │
                 (B, L, d_model)                         (B, L, d_model)
                    │                                         │
                    ▼                                         ▼
          ┌──────────────────────┐                  ┌──────────────────────┐
          │  Encoder Layer stack │                  │  Decoder Layer Stack │
          │   - Self-Attention   │                  │   - Self-Attention   │
          │   - Residual Add&Norm│                  │   - Cross-Attention  │◄──┐
          └──────────┬───────────┘                  │   - Residual Add&Norm│   │
                     │                              └──────────┬───────────┘   │
                     │                                         │               │
                     └─(Encoder Output Hidden Representations)─┘               │
                        Shape: (B, L, d_model) ────────────────────────────────┘
                                                               │
                                                               ▼
                                               [Flatten Transformation Matrix]
                                                   Shape: (B * L, d_model)
                                                               │
                                                               ▼
                                               [Unified Head Projection: W_vocab]
                                                   Shape: (B * L, V_size)
                                                               │
                                                               ▼
                                               [Reshape Array Unpacking Step]
                                                   Shape: (B, L, V_size)
                                                               │
                                                               ▼
                                               [Post-Softmax Vocabulary Probs]
```

---

This document serves as a reference specification for the implemented NumPy Transformer architecture, emphasizing tensor geometry, analytical gradients, and parameter bookkeeping in a framework-independent form.
