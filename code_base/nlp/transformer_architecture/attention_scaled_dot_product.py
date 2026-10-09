#Modelling\nlp\transformer_architecture\attention_scaled_dot_product.py

import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    # Cache parameters safely for multi-dimensional batch handling
    cache = (Q, K, V, mask)
    
    # Calculate matrix dot products: Q @ K^T
    matmul_qk = np.matmul(Q, K.swapaxes(-1, -2))
    
    d_k = Q.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)

    # Apply masking logic: 1.0 elements get heavily penalized
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Numerically stable Softmax execution across trailing dimensions
    attention_weights = np.exp(scaled_attention_logits - np.max(scaled_attention_logits, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)

    output = np.matmul(attention_weights, V)
    return output, attention_weights, cache

def scaled_dot_product_attention_backward(d_output, d_attention_weights, cache):
    Q, K, V, mask = cache
    d_k = Q.shape[-1]
    
    # Reconstruct attention weights matrix matching forward pass state
    matmul_qk = np.matmul(Q, K.swapaxes(-1, -2))
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = np.exp(scaled_attention_logits - np.max(scaled_attention_logits, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)

    # 1. Evaluate derivatives for Value Tensor (V)
    dV = np.matmul(attention_weights.swapaxes(-1, -2), d_output)

    # 2. Accumulate derivatives for weight metrics
    if d_attention_weights is None:
        d_attention_weights = np.zeros_like(attention_weights)
    d_attention_weights_combined = d_attention_weights + np.matmul(d_output, V.swapaxes(-1, -2))

    # 3. Softmax layer backward chain rule application
    d_scaled_attention_logits = d_attention_weights_combined * attention_weights
    sum_d_scaled_attention_logits = np.sum(d_scaled_attention_logits, axis=-1, keepdims=True)
    d_scaled_attention_logits -= attention_weights * sum_d_scaled_attention_logits

    # FIXED: Replaced raw boolean slicing with standard element multiplication broadcasting
    # Math: If mask == 1.0 (padding field), scale gradient to 0.0, else preserve original dZ values
    if mask is not None:
        d_scaled_attention_logits = d_scaled_attention_logits * (mask < 0.5)

    # 4. Direct scaling adjustments and matrix projections for Q and K
    d_matmul_qk = d_scaled_attention_logits / np.sqrt(d_k)
    dQ = np.matmul(d_matmul_qk, K)
    dK = np.matmul(d_matmul_qk.swapaxes(-1, -2), Q)

    return dQ, dK, dV

print("Attention matrices configured successfully!")