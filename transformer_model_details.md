# Forward Modelling and Bayesian Inference Estimations for Atomic Spectra and Qubit States

### A Pure NumPy Seq2Seq Physics Intent Parser: Mathematical & Structural Architecture Specifications

This manual provides a mathematically precise specification of the custom, from-scratch NumPy Transformer model optimized for compiling raw human-language text strings into structured physics-intent representations without depending on high-level deep learning frameworks.

The primary objective of this architecture is to lower technical barriers for physicists by streamlining their interaction with complex computational simulation pipelines. Historically, setting up physical simulations required manual coding, rigid syntax scripts, or raw mathematical formulas.

This engine changes that workflow by allowing physicists to enter configuration commands using standard, intuitive natural language queries. The unstructured query is parsed by a sequence-to-sequence AI architecture to extract parameter tokens, which are then routed to downstream execution frameworks to automate both forward modeling tasks and inverse inference estimations for atomic spectral and qubit states.

Ultimately, the backend handles these tasks natively using Python and Markov Chain Monte Carlo (MCMC) libraries, yielding comprehensive output in the form of both precise statistical data tables and dynamic visualization graphs.

---
## Table of Contents

- [1. Architectural Overview & Domain Grounding](#1-architectural-overview--domain-grounding)
  - [1.1 Nature of the Physics Problem Domain](#11-nature-of-the-physics-problem-domain)
  - [1.2 Suitability of the Pure NumPy Transformer Design ("Attention is All You Need")](#12-suitability-of-the-pure-numpy-transformer-design-attention-is-all-you-need)
  - [1.3 Macro-Level Block Framework (Global Encoder-Decoder Pipeline)](#13-macro-level-block-framework-global-encoder-decoder-pipeline)
- [2. Global Hyperparameter & Dimensional Geometry Blueprint](#2-global-hyperparameter--dimensional-geometry-blueprint)
  - [2.1 Structural Hyperparameter Array Matrix](#21-structural-hyperparameter-array-matrix)
  - [2.2 Target Vocabulary Invariants & Dimension Spaces](#22-target-vocabulary-invariants--dimension-spaces)
- [3. Foundational Core Components & Micro-Layer Mechanics](#3-foundational-core-components--micro-layer-mechanics)
  - [3.1 Static Geometric Waveform Positional Encodings](#31-static-geometric-waveform-positional-encodings)
  - [3.2 Multi-Head Attention (MHA) Engine](#32-multi-head-attention-mha-engine)
    - [3.2.1 Parallel Multi-Dimensional Tensor Head Partitioning](#321-parallel-multi-dimensional-tensor-head-partitioning)
    - [3.2.2 Scaled Dot-Product Core, Softmax Invariants, and Numerical Stability Bounds](#322-scaled-dot-product-core-softmax-invariants-and-numerical-stability-bounds)
    - [3.2.3 Forward/Backward Inter-Layer Communication Paths (Upper & Lower Layer Interfaces)](#323-forwardbackward-inter-layer-communication-paths-upper--lower-layer-interfaces)
  - [3.3 Position-Wise Feed-Forward Network (FFN) Blocks](#33-position-wise-feed-forward-network-ffn-blocks)
    - [3.3.1 Linear Transformations & ReLU Activation Gates](#331-linear-transformations--relu-activation-gates)
    - [3.3.2 Forward/Backward Inter-Layer Communication Paths](#332-forwardbackward-inter-layer-communication-paths)
  - [3.4 Feature-Axis Layer Normalization (LN) Contractions](#34-feature-axis-layer-normalization-ln-contractions)
    - [3.4.1 Reduction Contractions & Statistical Invariants](#341-reduction-contractions--statistical-invariants)
    - [3.4.2 Forward/Backward Inter-Layer Communication Paths](#342-forwardbackward-inter-layer-communication-paths)
  - [3.5 Dropout & Bernoulli Operational Mask Scaling Multipliers](#35-dropout--bernoulli-operational-mask-scaling-multipliers)
- [4. Structural Stack Architectures](#4-structural-stack-architectures)
  - [4.1 Transformer Encoder Stack Layer Pipelines](#41-transformer-encoder-stack-layer-pipelines)
    - [4.1.1 Macro Layer Assembly & Self-Attention Pipelines](#411-macro-layer-assembly--self-attention-pipelines)
    - [4.1.2 Forward/Backward Inter-Layer Communication Paths](#412-forwardbackward-inter-layer-communication-paths)
  - [4.2 Transformer Decoder Stack Layer Pipelines](#42-transformer-decoder-stack-layer-pipelines)
    - [4.2.1 Causal Mask Self-Attention & Asymmetric Cross-Attention Blocks](#421-causal-mask-self-attention--asymmetric-cross-attention-blocks)
    - [4.2.2 Forward/Backward Inter-Layer Communication Paths](#422-forwardbackward-inter-layer-communication-paths)
- [5. Micro-Level Vectorized Backpropagation Analytical Derivatives](#5-micro-level-vectorized-backpropagation-analytical-derivatives)
  - [5.1 Layer Normalization Multi-Axis Error Transpositions](#51-layer-normalization-multi-axis-error-transpositions)
  - [5.2 Causal Mask Isolated Softmax Derivative Chain Rule](#52-causal-mask-isolated-softmax-derivative-chain-rule)
  - [5.3 Asymmetric Cross-Attention Path-Gradient Routing (Anti-Scrambling Controls)](#53-asymmetric-cross-attention-path-gradient-routing-anti-scrambling-controls)
  - [5.4 Atomic Shared Index Vector Embedding Accumulation (`np.add.at` Invariants)](#54-atomic-shared-index-vector-embedding-accumulation-npaddat-invariants)
- [6. Optimization Framework & Macro Training Loop Pipeline](#6-optimization-framework--macro-training-loop-pipeline)
  - [6.1 Masked Target Token Dataset Ingestion Formatting](#61-masked-target-token-dataset-ingestion-formatting)
  - [6.2 Flattened 2D Indexing Shortcut for Masked Categorical Cross-Entropy Logit Gradients](#62-flattened-2d-indexing-shortcut-for-masked-categorical-cross-entropy-logit-gradients)
  - [6.3 Decoupled First- and Second-Moment Adam Optimization Engine](#63-decoupled-first--and-second-moment-adam-optimization-engine)
    - [6.3.1 Running Averages & Variable Time-Step Clock Counters](#631-running-averages--variable-time-step-clock-counters)
    - [6.3.2 Initialization Moment Bias Neutralization Matrix Equations](#632-initialization-moment-bias-neutralization-matrix-equations)
    - [6.3.3 Warm-Up Step Cosine Learning Rate Decay Mechanics](#633-warm-up-step-cosine-learning-rate-decay-mechanics)
    - [6.3.4 Parameter Memory Map Isolation (Isolated Encoder, Decoder, Output Head Dictionaries)](#634-parameter-memory-map-isolation-isolated-encoder-decoder-output-head-dictionaries)
- [7. Network Parameter Weight Invariant Serialization Layers](#7-network-parameter-weight-invariant-serialization-layers)
  - [7.1 Sequential Array Packaging & Suffix-Key Flattening Structures](#71-sequential-array-packaging--suffix-key-flattening-structures)
  - [7.2 Decoupled Native Array Manifold Restoration Mechanisms](#72-decoupled-native-array-manifold-restoration-mechanisms)
- [8. Autoregressive Predictive Inference Engine](#8-autoregressive-predictive-inference-engine)
  - [8.1 Autoregressive Generation Logic & Shifted Look-Ahead Mask Execution](#81-autoregressive-generation-logic--shifted-look-ahead-mask-execution)
  - [8.2 Post-Softmax Live Token-Step Confidence Extraction](#82-post-softmax-live-token-step-confidence-extraction)
- [9. End-to-End System Tensor Invariant Geometry Flowchart](#9-end-to-end-system-tensor-invariant-geometry-flowchart)

---

## 1. Architectural Overview & Domain Grounding

### 1.1 Nature of the Physics Problem Domain
---
Downstream quantum computing hardware and atomic simulation rigs require highly precise parameter inputs to execute correctly. Users submit these variables using flexible, unstructured natural language queries rather than rigid programming scripts. These incoming prompts embed a complex mix of alphanumeric parameter keys alongside gridded numerical values and case-sensitive file targets.

Consequently, the semantic validity of any given slot value is tied directly to the specific execution domain declared at the start of the text. Traditional rule-based regex pipelines quickly fracture under these conditions due to token ordering variances, while standard recurrent networks fail because they tend to drop context over long-range multi-variable dependencies.

### 1.2 Suitability of the Pure NumPy Transformer Design ("Attention is All You Need")
The original "Attention is All You Need" blueprint perfectly solves this mapping problem by replacing recurrence entirely with simultaneous dot-product attention operations. This structural choice allows the network to link distant token keys to their exact numerical values instantly, rendering spatial distance across the prompt irrelevant.

Furthermore, writing this transformer architecture from scratch in pure NumPy grants us total, granular control over the tensor geometry and the analytical backpropagation paths. By implementing custom derivatives natively, we can explicitly force padding token gradients to zero and insulate our weight space from training artifacts. This matrix foundation feeds an autoregressive decoder that uses causal masking to build correct plain-text syntax step by step, which our symbolic fallback layer then refines with absolute precision.

---

### 1.3 Macro-Level Block Framework (Global Encoder-Decoder Pipeline)

At the highest execution boundary, the parser operates as an asymmetrical sequence-to-sequence framework split into three interconnected zones. The pipeline begins in the source space, where the encoder ingests raw text tokens, projects them into continuous embedding spaces, injects positional waveform frequencies, and passes them through cascaded encoder layers to generate a stable hidden representation of the query.

Meanwhile, the target generative space drives the decoder pipeline by taking right-shifted target strings and processing them through causal self-attention layers to maintain strict syntactic order. These streams are then combined through an asymmetric cross-attention block that dynamically extracts contextual features directly from the encoder's hidden representations. Finally, the vocabulary projection space unifies the entire network by flattening the 3D hidden tensors into optimized 2D views, executing efficient dot-product projections against the vocabulary weights, and applying a stable softmax layer to yield real-time token probabilities and confidence analytics.

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
