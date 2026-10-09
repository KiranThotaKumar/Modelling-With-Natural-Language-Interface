#\ModelWithNLP\attention_architecture\transformer_decoder.py

import numpy as np

from nlp.transformer_architecture.positional_encoding import PositionalEncoding
from nlp.transformer_architecture.transformer_decoder_layer import TransformerDecoderLayer

class TransformerDecoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_seq_len, rate=0.1):
        self.num_layers = num_layers
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dec_layers = [
            TransformerDecoderLayer(d_model, num_heads, d_ff, rate)
            for _ in range(num_layers)
        ]
        self.cache = None

    def __call__(self, x, enc_output, look_ahead_mask, padding_mask, training=True):
        seq_len = x.shape[1]
        x_initial = x

        # Math Layer: Add geometric coordinate wave fields
        pos_enc_output = self.pos_encoding(seq_len)
        x = x + pos_enc_output

        attention_weights = {}
        for i in range(self.num_layers):
            x, attn_w1, attn_w2 = self.dec_layers[i](
                x, enc_output, look_ahead_mask, padding_mask, training=training
            )
            attention_weights[f'decoder_layer_{i+1}_self_attn'] = attn_w1
            attention_weights[f'decoder_layer_{i+1}_cross_attn'] = attn_w2

        self.cache = (x_initial, enc_output, training)
        return x, attention_weights

    def backward(self, d_output):
        # Backpropagate in reverse chronological order through the decoder layers
        d_x = d_output
        d_enc_output = np.zeros_like(self.cache[1])
        all_grads = {'decoder_layers_grads': []}

        for i in reversed(range(self.num_layers)):
            dx_from_layer, d_enc_output_from_layer, layer_grads = self.dec_layers[i].backward(d_x)
            d_enc_output += d_enc_output_from_layer
            d_x = dx_from_layer
            all_grads['decoder_layers_grads'].insert(0, layer_grads)

        d_input_embeddings = d_x
        return d_input_embeddings, d_enc_output, all_grads
