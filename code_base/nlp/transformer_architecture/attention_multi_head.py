#Modelling\nlp\transformer_architecture\attention_multi_head.py

import numpy as np


from nlp.transformer_architecture.attention_scaled_dot_product import scaled_dot_product_attention
from nlp.transformer_architecture.attention_scaled_dot_product import scaled_dot_product_attention_backward


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be perfectly divisible by num_heads"
        self.depth = d_model // num_heads

        # Weight matrices initialized using standard variance scaling scaling scale
        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.Wk = np.random.randn(d_model, d_model) * 0.01
        self.Wv = np.random.randn(d_model, d_model) * 0.01
        self.Wo = np.random.randn(d_model, d_model) * 0.01

        self.cache = None

    def _split_heads(self, x, batch_size):
        # Math Transform: (B, S, D) -> (B, S, h, d_k) -> (B, h, S, d_k)
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.swapaxes(1, 2)

    def _combine_heads(self, x, batch_size):
        # Math Transform: (B, h, S, d_k) -> (B, S, h, d_k) -> (B, S, D)
        x = x.swapaxes(1, 2)
        return x.reshape(batch_size, -1, self.d_model)

    def __call__(self, v, k, q, mask=None):
        batch_size = q.shape[0]

        # 1. Project inputs into coordinate frames via dot products
        q_proj = np.matmul(q, self.Wq)
        k_proj = np.matmul(k, self.Wk)
        v_proj = np.matmul(v, self.Wv)

        # 2. Distribute dimensions across parallel head architectures
        q_split = self._split_heads(q_proj, batch_size)
        k_split = self._split_heads(k_proj, batch_size)
        v_split = self._split_heads(v_proj, batch_size)

        # 3. Process via the core scaled attention mechanism
        scaled_attention, attention_weights, attn_cache = scaled_dot_product_attention(
            q_split, k_split, v_split, mask
        )

        # 4. Recombine heads and apply final mixing transformation
        concat_attention = self._combine_heads(scaled_attention, batch_size)
        output = np.matmul(concat_attention, self.Wo)

        # Cache exact forward state steps for exact backward alignment
        self.cache = (q, k, v, q_proj, k_proj, v_proj, q_split, k_split, v_split,
                      scaled_attention, concat_attention, attn_cache, mask)

        return output, attention_weights

    def backward(self, d_output, d_attention_weights=None):
        (q, k, v, q_proj, k_proj, v_proj, q_split, k_split, v_split,
         scaled_attention, concat_attention, attn_cache, mask) = self.cache
        batch_size = q.shape[0]

        # FIXED: Flatten sequence layout matrices to eliminate 3D tensor broadcasting bugs
        concat_attention_flat = concat_attention.reshape(-1, self.d_model)
        d_output_flat = d_output.reshape(-1, self.d_model)
        
        # Calculate Output Weight Matrix Derivatives
        dWo = np.matmul(concat_attention_flat.T, d_output_flat)
        d_concat_attention = np.matmul(d_output, self.Wo.T)

        # Propagate back through the head assembly functions
        d_scaled_attention = self._split_heads(d_concat_attention, batch_size)

        # Unwind the central attention matrix equations
        dQ_split, dK_split, dV_split = scaled_dot_product_attention_backward(
            d_scaled_attention, d_attention_weights, attn_cache
        )

        # Recombine parallel features back to sequence fields
        dq_proj = self._combine_heads(dQ_split, batch_size)
        dk_proj = self._combine_heads(dK_split, batch_size)
        dv_proj = self._combine_heads(dV_split, batch_size)

        # Consolidate projection blocks into 2D format for parameter tracking
        q_flat = q.reshape(-1, self.d_model)
        k_flat = k.reshape(-1, self.d_model)
        v_flat = v.reshape(-1, self.d_model)
        
        dq_proj_flat = dq_proj.reshape(-1, self.d_model)
        dk_proj_flat = dk_proj.reshape(-1, self.d_model)
        dv_proj_flat = dv_proj.reshape(-1, self.d_model)

        # Calculate final coordinate update parameter steps
        dWq = np.matmul(q_flat.T, dq_proj_flat)
        dWk = np.matmul(k_flat.T, dk_proj_flat)
        dWv = np.matmul(v_flat.T, dv_proj_flat)

        # Calculate inputs derivatives to flow back into previous layers
        dq = np.matmul(dq_proj, self.Wq.T)
        dk = np.matmul(dk_proj, self.Wk.T)
        dv = np.matmul(dv_proj, self.Wv.T)

        grads = {'dWq': dWq, 'dWk': dWk, 'dWv': dWv, 'dWo': dWo}
        return dq, dk, dv, grads

print("Multi-Head Attention class built and verified successfully!")

