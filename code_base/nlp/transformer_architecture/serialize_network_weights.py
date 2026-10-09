#\Modelling\nlp\transformer_architecture\serialize_network_weights.py

import numpy as np

def save_transformer_weights(model, embedding_matrix, weights_filename="transformer_physics_weights.npz"):
    weights_dict = {
        'embedding_matrix': embedding_matrix,
        # FIXED: Save unified sequence vocabulary tracking arrays
        'W_vocab': model.output_layer.W_vocab,
        'b_vocab': model.output_layer.b_vocab
    }
    
    # Pack Encoder weights sequentially
    for i, layer in enumerate(model.encoder.enc_layers):
        weights_dict[f'enc_{i}_Wq'] = layer.mha.Wq
        weights_dict[f'enc_{i}_Wk'] = layer.mha.Wk
        weights_dict[f'enc_{i}_Wv'] = layer.mha.Wv
        weights_dict[f'enc_{i}_Wo'] = layer.mha.Wo
        weights_dict[f'enc_{i}_W1'] = layer.ffn.W1
        weights_dict[f'enc_{i}_b1'] = layer.ffn.b1
        weights_dict[f'enc_{i}_W2'] = layer.ffn.W2
        weights_dict[f'enc_{i}_b2'] = layer.ffn.b2
        weights_dict[f'enc_{i}_ln1_g'] = layer.layernorm1.gamma
        weights_dict[f'enc_{i}_ln1_b'] = layer.layernorm1.beta
        weights_dict[f'enc_{i}_ln2_g'] = layer.layernorm2.gamma
        weights_dict[f'enc_{i}_ln2_b'] = layer.layernorm2.beta

    # Pack Decoder weights sequentially
    for i, layer in enumerate(model.decoder.dec_layers):
        weights_dict[f'dec_{i}_Wq1'] = layer.mha1.Wq
        weights_dict[f'dec_{i}_Wk1'] = layer.mha1.Wk
        weights_dict[f'dec_{i}_Wv1'] = layer.mha1.Wv
        weights_dict[f'dec_{i}_Wo1'] = layer.mha1.Wo
        weights_dict[f'dec_{i}_Wq2'] = layer.mha2.Wq
        weights_dict[f'dec_{i}_Wk2'] = layer.mha2.Wk
        weights_dict[f'dec_{i}_Wv2'] = layer.mha2.Wv
        weights_dict[f'dec_{i}_Wo2'] = layer.mha2.Wo
        weights_dict[f'dec_{i}_W1'] = layer.ffn.W1
        weights_dict[f'dec_{i}_b1'] = layer.ffn.b1
        weights_dict[f'dec_{i}_W2'] = layer.ffn.W2
        weights_dict[f'dec_{i}_b2'] = layer.ffn.b2
        weights_dict[f'dec_{i}_ln1_g'] = layer.layernorm1.gamma
        weights_dict[f'dec_{i}_ln1_b'] = layer.layernorm1.beta
        weights_dict[f'dec_{i}_ln2_g'] = layer.layernorm2.gamma
        weights_dict[f'dec_{i}_ln2_b'] = layer.layernorm2.beta
        weights_dict[f'dec_{i}_ln3_g'] = layer.layernorm3.gamma
        weights_dict[f'dec_{i}_ln3_b'] = layer.layernorm3.beta

    np.savez(weights_filename, **weights_dict)
    print(f"Successfully serialized and saved all network weights to '{weights_filename}'!")


def load_transformer_weights(model, weights_filename="transformer_physics_weights.npz"):
    data = np.load(weights_filename)
    
    # FIXED: Unpack unified sequence vocabulary tracking arrays
    model.output_layer.W_vocab = data['W_vocab']
    model.output_layer.b_vocab = data['b_vocab']

    # Restore Encoder parameter manifolds
    for i, layer in enumerate(model.encoder.enc_layers):
        layer.mha.Wq = data[f'enc_{i}_Wq']
        layer.mha.Wk = data[f'enc_{i}_Wk']
        layer.mha.Wv = data[f'enc_{i}_Wv']
        layer.mha.Wo = data[f'enc_{i}_Wo']
        layer.ffn.W1 = data[f'enc_{i}_W1']
        layer.ffn.b1 = data[f'enc_{i}_b1']
        layer.ffn.W2 = data[f'enc_{i}_W2']
        layer.ffn.b2 = data[f'enc_{i}_b2']
        layer.layernorm1.gamma = data[f'enc_{i}_ln1_g']
        layer.layernorm1.beta = data[f'enc_{i}_ln1_b']
        layer.layernorm2.gamma = data[f'enc_{i}_ln2_g']
        layer.layernorm2.beta = data[f'enc_{i}_ln2_b']

    # Restore Decoder parameter manifolds
    for i, layer in enumerate(model.decoder.dec_layers):
        layer.mha1.Wq = data[f'dec_{i}_Wq1']
        layer.mha1.Wk = data[f'dec_{i}_Wk1']
        layer.mha1.Wv = data[f'dec_{i}_Wv1']
        layer.mha1.Wo = data[f'dec_{i}_Wo1']
        layer.mha2.Wq = data[f'dec_{i}_Wq2']
        layer.mha2.Wk = data[f'dec_{i}_Wk2']
        layer.mha2.Wv = data[f'dec_{i}_Wv2']
        layer.mha2.Wo = data[f'dec_{i}_Wo2']
        layer.ffn.W1 = data[f'dec_{i}_W1']
        layer.ffn.b1 = data[f'dec_{i}_b1']
        layer.ffn.W2 = data[f'dec_{i}_W2']
        layer.ffn.b2 = data[f'dec_{i}_b2']
        layer.layernorm1.gamma = data[f'dec_{i}_ln1_g']
        layer.layernorm1.beta = data[f'dec_{i}_ln1_b']
        layer.layernorm2.gamma = data[f'dec_{i}_ln2_g']
        layer.layernorm2.beta = data[f'dec_{i}_ln2_b']
        layer.layernorm3.gamma = data[f'dec_{i}_ln3_g']
        layer.layernorm3.beta = data[f'dec_{i}_ln3_b']

    return data['embedding_matrix']
