#\ModelWithNLP\positional_encoding.py
#Modelling\nlp\transformer_architecture\positional_encoding.py
import numpy as np

class PositionalEncoding:
    def __init__(self, d_model, max_seq_len=2048):
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Initialize position encoding matrix shape: (max_seq_len, d_model)
        pe = np.zeros((max_seq_len, d_model))
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        
        # Calculate division term based on channel steps
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        # Assign alternating sine and cosine frequencies
        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:, :-1] if div_term.shape[0] * 2 > d_model else position * div_term)
        else:
            pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe

    def __call__(self, sequence_length):
        if sequence_length > self.max_seq_len:
            raise ValueError(f"Sequence length {sequence_length} exceeds max_seq_len {self.max_seq_len}.")
        return self.pe[:sequence_length, :]

    def backward(self, d_output):
        # Natively passes gradients through since there are no learnable params
        return d_output

print("Positional Encodings configured successfully!")