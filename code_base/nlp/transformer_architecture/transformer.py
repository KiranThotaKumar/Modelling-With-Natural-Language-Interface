#\Modelling\nlp\transformer_architecture\transformer.py


import numpy as np


from nlp.transformer_architecture.transformer_encoder import TransformerEncoder
from nlp.transformer_architecture.transformer_decoder import TransformerDecoder
from nlp.transformer_architecture.transformer_output_layer import TransformerOutputLayer
from nlp.transformer_architecture.embeddings import get_embeddings
from nlp.transformer_architecture.process_data_for_training import train_data



class Transformer:
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, d_ff,
                 max_seq_len_encoder, max_seq_len_decoder, vocab_size, rate=0.1):
        # 1. Initialize structural encoder and decoder sub-components
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, max_seq_len_encoder, rate)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, max_seq_len_decoder, rate)
        
        # 2. FIXED: Instantiate the new unified sequence vocabulary prediction output layer
        self.output_layer = TransformerOutputLayer(d_model, vocab_size)
        self.cache = None


    def __call__(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask, embedding_matrix, training=True):
        # 1. Source and target space lookup conversions
        inp_embeddings = get_embeddings(inp, embedding_matrix)
        tar_embeddings = get_embeddings(tar, embedding_matrix)

        # 2. Complete sequence macro block execution passes
        enc_output = self.encoder(inp_embeddings, enc_padding_mask, training=training)
        # dec_output retains its clear token sequence spatial resolution shape: (B, S, D)
        dec_output, attention_weights = self.decoder(tar_embeddings, enc_output, look_ahead_mask, dec_padding_mask, training=training)

        # 3. Connect sequence dimensions straight to the vocabulary probability head
        vocab_probs = self.output_layer(dec_output)

        self.cache = (inp, tar, dec_output, embedding_matrix)
        return vocab_probs, attention_weights

    def backward(self, d_vocab_probs):
        # 1. Retrieve raw sequence data arrays cached during the forward execution pass
        inp, tar, dec_output, embedding_matrix_cached = self.cache
        d_embedding_matrix = np.zeros_like(embedding_matrix_cached)

        # 2. Backpropagate error sequences natively down the unified vocabulary output head
        # Returns d_dec_output_full_seq matching exactly (Batch_Size, Seq_Len, D_Model)
        d_dec_output_full_seq, output_layer_grads = self.output_layer.backward(d_vocab_probs)

        # 3. Backpropagate error streams through the Transformer Decoder stack
        # Returns full sequence coordinates backprop paths
        d_tar_embeddings, d_enc_output, decoder_grads = self.decoder.backward(d_dec_output_full_seq)

        # 4. Backpropagate error streams through the Transformer Encoder stack
        d_inp_embeddings, encoder_grads = self.encoder.backward(d_enc_output)

        # 5. C-Compiled Vectorised index accumulation over spatial embedding boundaries
        np.add.at(d_embedding_matrix, tar, d_tar_embeddings)
        np.add.at(d_embedding_matrix, inp, d_inp_embeddings)
        
        # Zero out padding token gradients to maintain architectural containment boundaries
        d_embedding_matrix[train_data.word_to_idx['<pad>']] = 0.0  

        all_grads = {
            'output_layer_grads': output_layer_grads,
            'decoder_grads': decoder_grads,
            'encoder_grads': encoder_grads,
            'd_embedding_matrix': d_embedding_matrix
        }
        return all_grads
