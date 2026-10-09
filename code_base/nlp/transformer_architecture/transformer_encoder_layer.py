#Modelling\nlp\transformer_architecture\transformer_encoder_layer.py

import numpy as np

from nlp.transformer_architecture.attention_multi_head import MultiHeadAttention
from nlp.transformer_architecture.feed_forward_network import FeedForwardNetwork
from nlp.transformer_architecture.layer_normalization import LayerNormalization
from nlp.transformer_architecture.dropout import Dropout

class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, rate=0.1):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.cache = None

    def __call__(self, x, mask, training=True):
        # 1. Self-Attention Sub-Layer Pipeline
        attn_output, attn_weights = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # Math Connection: Residual Add & Norm -> out1 = LayerNorm(x + Attention)
        out1 = self.layernorm1(x + attn_output)

        # 2. Position-Wise Feed-Forward Sub-Layer Pipeline
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Math Connection: Residual Add & Norm -> out2 = LayerNorm(out1 + FFN)
        out2 = self.layernorm2(out1 + ffn_output)

        self.cache = (x, attn_output, out1, ffn_output)
        return out2

    def backward(self, d_out):
        x, attn_output, out1, ffn_output = self.cache

        # Backward through Add & Norm 2
        d_out1_plus_ffn_output, d_gamma2, d_beta2 = self.layernorm2.backward(d_out)
        
        # Split gradients along the residual summation junction
        d_out1_from_ffn_res = d_out1_plus_ffn_output
        d_ffn_output_from_res = d_out1_plus_ffn_output

        # Backward through Dropout2 and Feed-Forward Network
        d_ffn_output_pre_dropout = self.dropout2.backward(d_ffn_output_from_res)
        dx_ffn, ffn_grads = self.ffn.backward(d_ffn_output_pre_dropout)

        # Accumulate total gradients hitting the central out1 boundary
        d_out1 = d_out1_from_ffn_res + dx_ffn

        # Backward through Add & Norm 1
        d_x_plus_attn_output, d_gamma1, d_beta1 = self.layernorm1.backward(d_out1)
        
        # Split gradients along the initial residual summation junction
        dx_from_attn_res = d_x_plus_attn_output
        d_attn_output_from_res = d_x_plus_attn_output

        # Backward through Dropout1 and Multi-Head Attention
        d_attn_output_pre_dropout = self.dropout1.backward(d_attn_output_from_res)
        dq, dk, dv, mha_grads = self.mha.backward(d_attn_output_pre_dropout, d_attention_weights=None)

        # Math: Summing identical coordinate paths since Q, K, V all stem from input x
        dx = dx_from_attn_res + dq + dk + dv

        grads = {
            'd_gamma1': d_gamma1, 'd_beta1': d_beta1,
            'd_gamma2': d_gamma2, 'd_beta2': d_beta2,
            'mha_grads': mha_grads, 'ffn_grads': ffn_grads
        }
        return dx, grads