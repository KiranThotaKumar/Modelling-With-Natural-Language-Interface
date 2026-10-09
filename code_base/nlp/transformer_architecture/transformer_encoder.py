#\Modelling\nlp\transformer_architecture\transformer_encoder.py

import numpy as np
from nlp.transformer_architecture.positional_encoding import PositionalEncoding
from nlp.transformer_architecture.transformer_encoder_layer import TransformerEncoderLayer


class TransformerEncoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_seq_len, rate=0.1):
        self.num_layers = num_layers
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.enc_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff, rate)
            for _ in range(num_layers)
        ]
        self.cache = None

    def __call__(self, x, mask, training=True):
        seq_len = x.shape[1]
        x_initial = x

        # Math Layer: Add geometric coordinate wave fields
        pos_enc_output = self.pos_encoding(seq_len)
        x = x + pos_enc_output

        # Loop through the multi-layer stack pipelines sequentially
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training=training)

        self.cache = (x_initial, pos_enc_output, mask, training)
        return x

    def backward(self, d_output):
        # Backpropagate in reverse chronological order through the encoder layers
        d_x = d_output
        all_grads = {'encoder_layers_grads': []}

        for i in reversed(range(self.num_layers)):
            d_x, layer_grads = self.enc_layers[i].backward(d_x)
            all_grads['encoder_layers_grads'].insert(0, layer_grads)

        # Math Update: PE has no learnable params, its input derivative dx maps to x_initial
        d_input_embeddings = d_x
        return d_input_embeddings, all_grads

