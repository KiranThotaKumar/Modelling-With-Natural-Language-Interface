#\Modelling\nlp\transformer_architecture\transformer_decoder_layer.py

import numpy as np

from nlp.transformer_architecture.positional_encoding import PositionalEncoding
from nlp.transformer_architecture.attention_multi_head import MultiHeadAttention
from nlp.transformer_architecture.feed_forward_network import FeedForwardNetwork
from nlp.transformer_architecture.layer_normalization import LayerNormalization
from nlp.transformer_architecture.dropout import Dropout


class TransformerDecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, rate=0.1):
        self.mha1 = MultiHeadAttention(d_model, num_heads)  # Masked Causal Self-Attention
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # Cross-Attention Block
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)
        self.cache = None

    def __call__(self, x, enc_output, look_ahead_mask, padding_mask, training=True):
        # 1. Masked Causal Self-Attention Pipeline
        attn1_output, attn_w1 = self.mha1(x, x, x, look_ahead_mask)
        attn1_output_dropped = self.dropout1(attn1_output, training=training)
        out1 = self.layernorm1(x + attn1_output_dropped)

        # 2. Cross-Attention Pipeline (Math: Q from Decoder out1, K and V from Encoder Output)
        attn2_output, attn_w2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2_output_dropped = self.dropout2(attn2_output, training=training)
        out2 = self.layernorm2(out1 + attn2_output_dropped)

        # 3. Position-Wise Feed-Forward Pipeline
        ffn_output = self.ffn(out2)
        ffn_output_dropped = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output_dropped)

        self.cache = (x, enc_output, attn1_output_dropped, attn2_output_dropped, out1, out2, ffn_output, training)
        return out3, attn_w1, attn_w2

    def backward(self, d_out):
        x, enc_output, attn1_output_dropped, attn2_output_dropped, out1, out2, ffn_output, training = self.cache

        # Backward through Add & Norm 3
        d_out2_plus_ffn, d_gamma3, d_beta3 = self.layernorm3.backward(d_out)
        d_ffn_dropped = self.dropout3.backward(d_out2_plus_ffn)
        d_ffn_out, ffn_grads = self.ffn.backward(d_ffn_dropped)
        d_out2 = d_out2_plus_ffn + d_ffn_out

        # Backward through Add & Norm 2 (Cross-Attention Block)
        d_out1_plus_attn2, d_gamma2, d_beta2 = self.layernorm2.backward(d_out2)
        d_attn2_dropped = self.dropout2.backward(d_out1_plus_attn2)
        
        # FIXED unpacking sequence: aligned to (dq, dk, dv) to prevent cross-coordinate scrambled fields
        dq2, dk2, dv2, mha2_grads = self.mha2.backward(d_attn2_dropped, d_attention_weights=None)
        
        # Distribute gradients: Q maps back to Decoder out1, while K and V map to the Encoder Output
        d_out1 = d_out1_plus_attn2 + dq2
        d_enc_output = dk2 + dv2

        # Backward through Add & Norm 1 (Causal Self-Attention Block)
        d_x_plus_attn1, d_gamma1, d_beta1 = self.layernorm1.backward(d_out1)
        d_attn1_dropped = self.dropout1.backward(d_x_plus_attn1)
        dq1, dk1, dv1, mha1_grads = self.mha1.backward(d_attn1_dropped, d_attention_weights=None)
        
        dx = d_x_plus_attn1 + dq1 + dk1 + dv1

        grads = {
            'd_gamma1': d_gamma1, 'd_beta1': d_beta1,
            'd_gamma2': d_gamma2, 'd_beta2': d_beta2,
            'd_gamma3': d_gamma3, 'd_beta3': d_beta3,
            'mha1_grads': mha1_grads, 'mha2_grads': mha2_grads,
            'ffn_grads': ffn_grads
        }
        return dx, d_enc_output, grads
