# Mathematical & Structural Architecture Specifications: Pure NumPy Seq2Seq Physics Intent Parser

This manual provides an exhaustive, mathematically precise specification of the custom, from-scratch NumPy Transformer model optimized for compiling raw human-language text strings into structured physics operational target syntax.

---

## Table of Contents
- [1. Global Architectural Geometry & Dimensions Blueprint](#1-global-architectural-geometry--dimensions-blueprint)
  - [1.1 Structural Hyperparameter Array Matrix](#11-structural-hyperparameter-array-matrix)
  - [1.2 Target Vocabulary Invariants & Dimension Spaces](#12-target-vocabulary-invariants--dimension-spaces)
- [2. Core Component Equations & Mathematical Mapping](#2-core-component-equations--mathematical-mapping)
  - [2.1 Static Geometric Waveform Positional Encodings (`d_model % 2 == 1` Resilience Slicing)](#21-static-geometric-waveform-positional-encodings-d_model--2--1-resilience-slicing)
  - [2.2 Multi-Dimensional Tensor Head Partitioning (`_split_heads` / `_combine_heads` Forward Transforms)](#22-multi-dimensional-tensor-head-partitioning-_split_heads---_combine_heads-forward-transforms)
  - [2.3 Scaled Dot-Product Core Softmax Invariants & Numerical Stability Bounds](#23-scaled-dot-product-core-softmax-invariants--numerical-stability-bounds)
  - [2.4 Feature-Axis Layer Normalization Reduction Contractions](#24-feature-axis-layer-normalization-reduction-contractions)
- [3. Micro-Level Vectorized Backpropagation Analytical Derivatives](#3-micro-level-vectorized-backpropagation-analytical-derivatives)
  - [3.1 Layer Normalization Multi-Axis Error Transpositions](#31-layer-normalization-multi-axis-error-transpositions)
  - [3.2 Causal Mask Isolated Softmax Derivative Chain Rule](#32-causal-mask-isolated-softmax-derivative-chain-rule)
  - [3.3 Flattened 2D Indexing Shortcut for Masked Categorical Cross-Entropy Logit Gradients](#33-flattened-2d-indexing-shortcut-for-masked-categorical-cross-entropy-logit-gradients)
  - [3.4 Asymmetric Cross-Attention Path-Gradient Routing (Anti-Scrambling Controls)](#34-asymmetric-cross-attention-path-gradient-routing-anti-scrambling-controls)
  - [3.5 Atomic Shared Index Vector Accumulation (`np.add.at` Invariants)](#35-atomic-shared-index-vector-accumulation-npaddat-invariants)
- [4. Decoupled First- and Second-Moment Adam Optimization Architecture](#4-decoupled-first--and-second-moment-adam-optimization-architecture)
  - [4.1 Running Averages & Variable Time-Step Clock Counters](#41-running-averages--variable-time-step-clock-counters)
  - [4.2 Initialization Moment Bias Neutralization Matrix Equations](#42-initialization-moment-bias-neutralization-matrix-equations)
  - [4.3 Warm-Up Step Cosine Learning Rate Decay Mechanics](#43-warm-up-step-cosine-learning-rate-decay-mechanics)
  - [4.4 Parameter Memory Map Isolation (Isolated Encoder, Decoder, and Unified Output Head State Dictionaries)](#44-parameter-memory-map-isolation-isolated-encoder-decoder-and-unified-output-head-state-dictionaries)
- [5. Network Parameter Weight Invariant Serialization Layers](#5-network-parameter-weight-invariant-serialization-layers)
  - [5.1 Sequential Array Packaging & Suffix-Key Flattening Structures](#51-sequential-array-packaging--suffix-key-flattening-structures)
  - [5.2 Decoupled Native Array Manifold Restoration Mechanisms](#52-decoupled-native-array-manifold-restoration-mechanisms)
- [6. End-to-End System Tensor Invariant Geometry Flowchart](#6-end-to-end-system-tensor-invariant-geometry-flowchart)

---

## 1. Global Architectural Geometry & Dimensions Blueprint

### 1.1 Structural Hyperparameter Array Matrix
The architectural baseline of the Sequence-to-Sequence (Seq2Seq) Transformer is constructed completely using vectorized NumPy array allocations without dependencies on high-level deep learning graph frameworks.


|$Parameter$ | Operational Definition | Static Field Coordinate Assignment |
| :--- | :--- | :--- |
| $d_{\text{model}}$ | Continuous Hidden Layer Coordinate Representation Dimension | **64** |
| $n_{\text{heads}}$ | Parallel Multi-Head Attention Processing Streams | **4** |
| $d_{\text{depth}}$ | Individual Head Projection Track Slicing Dimension ($d_{\text{model}} / n_{\text{heads}}$) | **16** |
| $d_{\text{ff}}$ | Position-Wise Feed-Forward Network Hidden Expansion Width | **256** |
| $N_{\text{enc}}$ | Cascaded Transformer Encoder Layer Blocks | **2** |
| $N_{\text{dec}}$ | Cascaded Transformer Decoder Layer Blocks | **2** |
| $L_{\text{max}}$ | Hard Sequence Context Mask and Truncation Boundary | **25** |

### 1.2 Target Vocabulary Invariants & Dimension Spaces
The mathematical projection layer links token tracking maps across an engineered vocabulary domain:
* **Unique Computational Vocabulary Size ($V_{\text{size}}$):** **877 clean structural words**.
* **Global Embedding Matrix Mapping Boundary:** $(V_{\text{size}} \times d_{\text{model}}) \longrightarrow (877 \times 64)$


---

## 2. Core Component Equations & Mathematical Mapping

### 2.1 Static Geometric Waveform Positional Encodings (`d_model % 2 == 1` Resilience Slicing)
Spatial positions are mapped directly onto coordinate wave vectors using alternating trigonometric frequencies. For token position index $\text{pos} \in [0, L_{\text{max}}-1]$ and channel axis index $i \in [0, d_{\text{model}}-1]$:

$$PE_{(\text{pos}, 2i)} = \sin\left(\text{pos} \cdot \exp\left(-2i \cdot \frac{\ln(10000.0)}{d_{\text{model}}}\right)\right)$$

$$PE_{(\text{pos}, 2i+1)} = \cos\left(\text{pos} \cdot \exp\left(-2i \cdot \frac{\ln(10000.0)}{d_{\text{model}}}\right)\right)$$


To handle variations where embedding dimensions might not be perfectly even (\(d_{\text{model}} \% 2 == 1\)), the division array term is dynamically clipped via matrix slices:
```python
if d_model % 2 == 1:
    pe[:, 1::2] = np.cos(position * div_term[:, :-1] if div_term.shape[0] * 2 > d_model else position * div_term)
```
This safeguards matrix addition maps from structural broadcasting errors.

### 2.2 Multi-Dimensional Tensor Head Partitioning (`_split_heads` / `_combine_heads` Forward Transforms)
Before applying self-attention mechanisms, projected state matrices are distributed across independent processing tracks. The geometry transforms continuously across axes following these layout tracks:

```text
_split_heads: 
(Batch, SeqLen, d_model) -> (Batch, SeqLen, n_heads, d_depth) -> (Batch, n_heads, SeqLen, d_depth)

_combine_heads: 
(Batch, n_heads, SeqLen, d_depth) -> (Batch, SeqLen, n_heads, d_depth) -> (Batch, SeqLen, d_model)
```


### 2.3 Scaled Dot-Product Core Softmax Invariants & Numerical Stability Bounds
Given query, key, and value tracking manifolds assigned to 4D track blocks, the execution equation is:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{depth}}}} + (M_{\text{causal}} \cdot -10^9)\right)V$$

Where $(M_{\text{causal}}$) is the upper-triangular look-ahead structural masking matrix penalizing future token coordinates. Row-wise numerical overflow profiles are mitigated by isolating maximum values:

$$\text{Softmax}(Z_{i}) = \frac{\exp(Z_i - \max(Z))}{\sum_{j} \exp(Z_j - \max(Z))}$$


### 2.4 Feature-Axis Layer Normalization Reduction Contractions
To limit shifting statistics across hidden representations, values are normalized across the final feature dimension vector:

$$\text{LN}(x) = \gamma \odot \left(\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\right) + \beta$$

$$\mu = \frac{1}{d_{\text{model}}} \sum_{k=1}^{d_{\text{model}}} x_k, \quad \sigma^2 = \frac{1}{d_{\text{model}}} \sum_{k=1}^{d_{\text{model}}} (x_k - \mu)^2, \quad \epsilon = 10^{-6}$$

---

## 3. Micro-Level Vectorized Backpropagation Analytical Derivatives

### 3.1 Layer Normalization Multi-Axis Error Transpositions
### 3.1 Layer Normalization Multi-Axis Error Transpositions
Let $x_{\text{norm}} = (x - \mu) \cdot \text{inv\_std}$, where $\text{inv\_std} = (\sigma^2 + \epsilon)^{-0.5}$. The mathematical gradient reductions collapse non-feature tracking fields across multi-dimensional blocks without manual looping arrays:

$$\delta_{\beta} = \sum_{\text{axes}} \partial L, \quad \delta_{\gamma} = \sum_{\text{axes}} \partial L \odot x_{\text{norm}}$$

$$\partial_{\sigma^2} = \sum_{k=1}^{d_{\text{model}}} \left( \partial L \odot \gamma \odot (x - \mu) \right) \cdot \left(-\frac{1}{2}\right) \cdot \text{inv\_std}^3$$

$$\partial_{\mu} = \sum_{k=1}^{d_{\text{model}}} \left( \partial L \odot \gamma \odot (-\text{inv\_std}) \right) + \partial_{\sigma^2} \cdot \frac{\sum -2(x - \mu)}{d_{\text{model}}}$$

$$\partial x = \partial L \odot \gamma \odot \text{inv\_std} + \left( \partial_{\sigma^2} \cdot \frac{2(x - \mu)}{d_{\text{model}}} \right) + \frac{\partial_{\mu}}{d_{\text{model}}}$$

### 3.2 Causal Mask Isolated Softmax Derivative Chain Rule
When backpropagating through attention weights, the derivative is computed directly over the tracking matrix:

$$\partial Z_{\text{logits}} = \left( \partial A_{\text{out}} \odot A_{\text{weights}} \right) - A_{\text{weights}} \odot \sum_{\text{row}} \left(\partial A_{\text{out}} \odot A_{\text{weights}}\right)$$

Gradient leakage past causal masking coordinates is explicitly isolated by multiplying element-wise with an indicator tracking step:

$$\partial Z_{\text{logits}} = \partial Z_{\text{logits}} \odot \mathbb{I}(M_{\text{causal}} < 0.5)$$

### 3.3 Flattened 2D Indexing Shortcut for Masked Categorical Cross-Entropy Logit Gradients
To bypass expensive computational trees, the root sequence error derivative at the output classification layer is flattened directly into 2D memory arrays before backpropagation:

$$\frac{\partial L}{\partial Z} = \frac{(P - Y) \odot M_{\text{loss}}}{N_{\text{active\_tokens}}}$$


## 4. Decoupled First- and Second-Moment Adam Optimization Architecture

### 4.1 Running Averages & Variable Time-Step Clock Counters
Parameter tracking maintains running momentum metrics across discrete optimization intervals $t$:

$$m_{t} = \beta_{1} m_{t-1} + (1 - \beta_{1}) g_{t}, \quad v_{t} = \beta_{2} v_{t-1} + (1 - \beta_{2}) g_{t}^{2}$$

Where static hyperparameter variables match values: $\beta_1 = 0.9$, $\beta_2 = 0.999$.

### 4.2 Initialization Moment Bias Neutralization Matrix Equations
To neutralize first- and second-moment initialization bias during initial steps, tracking scales are corrected via:

$$\hat{m}_{t} = \frac{m_{t}}{1 - \beta_{1}^{t}}, \quad \hat{v}_{t} = \frac{v_{t}}{1 - \beta_{2}^{t}}$$

### 4.3 Warm-Up Step Cosine Learning Rate Decay Mechanics
The real-time operational learning rate $\alpha_{t}$ drops smoothly down across tracking epochs following an absolute cosine trajectory boundary:

$$\alpha_{t} = \alpha_{\text{min}} + \frac{1}{2}(\alpha_{\text{base}} - \alpha_{\text{min}})\left(1 + \cos \left(\pi \cdot \frac{\text{step}_{t}}{\text{step}_{\text{total}}}\right)\right)$$

Where assignments trace configurations: $\alpha_{\text{base}} = 0.002$, $\alpha_{\text{min}} = 0.0001$.

### 4.4 Parameter Memory Map Isolation (Isolated Encoder, Decoder, and Unified Output Head State Dictionaries)
The Adam optimizer contains isolated storage maps (`self.m_enc`, `self.v_enc`, `self.m_dec`, `self.v_dec`, `self.m_out`, `self.v_out`) structured precisely to layer keys. This decouples running updates and limits variable bleeding across hidden layer blocks.

---

## 5. Network Parameter Weight Invariant Serialization Layers

### 5.1 Sequential Array Packaging & Suffix-Key Flattening Structures
Weight state tracking flattens multidimensional matrix entities into indexed dictionaries packed using custom layer string tags before exporting to disk:
* Encoder layers are mapped via: `f'enc_{layer_index}_Wq'`, `f'enc_{layer_index}_ffn_W1'`, etc.
* Decoder layers are mapped via: `f'dec_{layer_index}_Wq1'`, `f'dec_{layer_index}_Wq2'`, etc.
* The output prediction head parameters are locked under explicitly isolated tags: `'W_vocab'` and `'b_vocab'`.

### 5.2 Decoupled Native Array Manifold Restoration Mechanisms
During initialization recovery, `load_transformer_weights` streams the `.npz` archive directly into the model structures. This mirrors the shapes of the layer weight spaces and injects saved configurations back into the execution paths with 100% mathematical precision.

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
             (B, L, d_model)                           (B, L, d_model)
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
