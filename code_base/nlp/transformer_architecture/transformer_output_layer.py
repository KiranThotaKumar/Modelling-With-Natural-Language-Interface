#\Modelling\nlp\transformer_architecture\transformer_output_layer.py

import numpy as np
from nlp.transformer_architecture.utility_functions import softmax



class TransformerOutputLayer:
    def __init__(self, d_model, vocab_size):
        # A single unified matrix projecting hidden dimensions onto vocabulary classes
        self.W_vocab = np.random.randn(d_model, vocab_size) * 0.01
        self.b_vocab = np.zeros((1, vocab_size))
        self.cache = None

    def __call__(self, dec_output):
        # dec_output shape: (Batch_Size, Seq_Len, D_Model)
        batch_size, seq_len, d_model = dec_output.shape
        
        # Reshape to 2D matrix to compute dot product efficiently
        flat_dec = dec_output.reshape(-1, d_model)
        
        # Compute logits across vocabulary space: (Batch_Size * Seq_Len, Vocab_Size)
        logits = np.dot(flat_dec, self.W_vocab) + self.b_vocab
        probs = softmax(logits)
        
        # Reshape back to full sequence structure: (Batch_Size, Seq_Len, Vocab_Size)
        probs = probs.reshape(batch_size, seq_len, -1)
        
        self.cache = (dec_output, probs)
        return probs

    def backward(self, d_probs):
        # d_probs input shape matches sequence space: (Batch_Size, Seq_Len, Vocab_Size)
        dec_output, probs = self.cache
        batch_size, seq_len, d_model = dec_output.shape
        vocab_size = self.W_vocab.shape[1]
        
        # Flatten gradients and inputs to evaluate linear derivative matrices
        d_logits_flat = d_probs.reshape(-1, vocab_size)
        flat_dec = dec_output.reshape(-1, d_model)
        
        # Evaluate parameter gradients hitting the vocabulary weights matrix
        dW_vocab = np.dot(flat_dec.T, d_logits_flat)
        db_vocab = np.sum(d_logits_flat, axis=0, keepdims=True)
        
        # Propagate error array back to the decoder matrix boundary
        d_flat_dec = np.dot(d_logits_flat, self.W_vocab.T)
        d_dec_output = d_flat_dec.reshape(batch_size, seq_len, d_model)
        
        grads = {'dW_vocab': dW_vocab, 'db_vocab': db_vocab}
        return d_dec_output, grads


