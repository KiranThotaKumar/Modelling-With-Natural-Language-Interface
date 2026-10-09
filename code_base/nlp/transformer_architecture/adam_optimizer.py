#\Modelling\nlp\transformer_architecture\adam_optimizer.py

import numpy as np


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Global time-step tracker clock counter
        
        # Isolated master history state dictionaries
        self.m_emb = None
        self.v_emb = None
        self.m_enc = {}
        self.v_enc = {}
        self.m_dec = {}
        self.v_dec = {}
        self.m_out = None
        self.v_out = None

    def update_param(self, param, grad, m_state, v_state):
        # Core time-step Adam matrix update calculus logic
        # Math: m = beta1 * m + (1 - beta1) * grad
        m_state *= self.beta1
        m_state += (1.0 - self.beta1) * grad
        
        # Math: v = beta2 * v + (1 - beta2) * grad^2
        v_state *= self.beta2
        v_state += (1.0 - self.beta2) * (grad ** 2)
        
        # Math: Bias corrections over early training iterations
        m_hat = m_state / (1.0 - self.beta1 ** self.t)
        v_hat = v_state / (1.0 - self.beta2 ** self.t)
        
        # Math Parameter Update: Theta = Theta - alpha * m_hat / (sqrt(v_hat) + eps)
        param -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param

    def step(self, model, embedding_matrix, grads):
        self.t += 1  # Increment training iteration time step

        # 1. Update Embedding Matrix Weights Layer
        if self.m_emb is None:
            self.m_emb = np.zeros_like(embedding_matrix)
            self.v_emb = np.zeros_like(embedding_matrix)
        embedding_matrix = self.update_param(embedding_matrix, grads['d_embedding_matrix'], self.m_emb, self.v_emb)

        # 2. Update Encoder Layers Parameters Sequentially
        for i, enc_layer_grads in enumerate(grads['encoder_grads']['encoder_layers_grads']):
            enc_layer = model.encoder.enc_layers[i]
            
            if i not in self.m_enc:
                self.m_enc[i] = {
                    'Wq': np.zeros_like(enc_layer.mha.Wq), 'Wk': np.zeros_like(enc_layer.mha.Wk),
                    'Wv': np.zeros_like(enc_layer.mha.Wv), 'Wo': np.zeros_like(enc_layer.mha.Wo),
                    'W1': np.zeros_like(enc_layer.ffn.W1), 'b1': np.zeros_like(enc_layer.ffn.b1),
                    'W2': np.zeros_like(enc_layer.ffn.W2), 'b2': np.zeros_like(enc_layer.ffn.b2),
                    'g1': np.zeros_like(enc_layer.layernorm1.gamma), 'beta1': np.zeros_like(enc_layer.layernorm1.beta),
                    'g2': np.zeros_like(enc_layer.layernorm2.gamma), 'beta2': np.zeros_like(enc_layer.layernorm2.beta)
                }
                self.v_enc[i] = {k: np.zeros_like(v) for k, v in self.m_enc[i].items()}

            em, ev = self.m_enc[i], self.v_enc[i]
            
            enc_layer.mha.Wq = self.update_param(enc_layer.mha.Wq, enc_layer_grads['mha_grads']['dWq'], em['Wq'], ev['Wq'])
            enc_layer.mha.Wk = self.update_param(enc_layer.mha.Wk, enc_layer_grads['mha_grads']['dWk'], em['Wk'], ev['Wk'])
            enc_layer.mha.Wv = self.update_param(enc_layer.mha.Wv, enc_layer_grads['mha_grads']['dWv'], em['Wv'], ev['Wv'])
            enc_layer.mha.Wo = self.update_param(enc_layer.mha.Wo, enc_layer_grads['mha_grads']['dWo'], em['Wo'], ev['Wo'])
            enc_layer.ffn.W1 = self.update_param(enc_layer.ffn.W1, enc_layer_grads['ffn_grads']['dW1'], em['W1'], ev['W1'])
            enc_layer.ffn.b1 = self.update_param(enc_layer.ffn.b1, enc_layer_grads['ffn_grads']['db1'], em['b1'], ev['b1'])
            enc_layer.ffn.W2 = self.update_param(enc_layer.ffn.W2, enc_layer_grads['ffn_grads']['dW2'], em['W2'], ev['W2'])
            enc_layer.ffn.b2 = self.update_param(enc_layer.ffn.b2, enc_layer_grads['ffn_grads']['db2'], em['b2'], ev['b2'])
            enc_layer.layernorm1.gamma = self.update_param(enc_layer.layernorm1.gamma, enc_layer_grads['d_gamma1'].flatten(), em['g1'], ev['g1'])
            enc_layer.layernorm1.beta = self.update_param(enc_layer.layernorm1.beta, enc_layer_grads['d_beta1'].flatten(), em['beta1'], ev['beta1'])
            enc_layer.layernorm2.gamma = self.update_param(enc_layer.layernorm2.gamma, enc_layer_grads['d_gamma2'].flatten(), em['g2'], ev['g2'])
            enc_layer.layernorm2.beta = self.update_param(enc_layer.layernorm2.beta, enc_layer_grads['d_beta2'].flatten(), em['beta2'], ev['beta2'])

        # 3. Update Decoder Layers Parameters Sequentially
        for i, dec_layer_grads in enumerate(grads['decoder_grads']['decoder_layers_grads']):
            dec_layer = model.decoder.dec_layers[i]
            
            if i not in self.m_dec:
                self.m_dec[i] = {
                    'Wq1': np.zeros_like(dec_layer.mha1.Wq), 'Wk1': np.zeros_like(dec_layer.mha1.Wk), 'Wv1': np.zeros_like(dec_layer.mha1.Wv), 'Wo1': np.zeros_like(dec_layer.mha1.Wo),
                    'Wq2': np.zeros_like(dec_layer.mha2.Wq), 'Wk2': np.zeros_like(dec_layer.mha2.Wk), 'Wv2': np.zeros_like(dec_layer.mha2.Wv), 'Wo2': np.zeros_like(dec_layer.mha2.Wo),
                    'W1': np.zeros_like(dec_layer.ffn.W1), 'b1': np.zeros_like(dec_layer.ffn.b1), 'W2': np.zeros_like(dec_layer.ffn.W2), 'b2': np.zeros_like(dec_layer.ffn.b2),
                    'g1': np.zeros_like(dec_layer.layernorm1.gamma), 'beta1': np.zeros_like(dec_layer.layernorm1.beta),
                    'g2': np.zeros_like(dec_layer.layernorm2.gamma), 'beta2': np.zeros_like(dec_layer.layernorm2.beta),
                    'g3': np.zeros_like(dec_layer.layernorm3.gamma), 'beta3': np.zeros_like(dec_layer.layernorm3.beta)
                }
                self.v_dec[i] = {k: np.zeros_like(v) for k, v in self.m_dec[i].items()}

            dm, dv = self.m_dec[i], self.v_dec[i]
            
            dec_layer.mha1.Wq = self.update_param(dec_layer.mha1.Wq, dec_layer_grads['mha1_grads']['dWq'], dm['Wq1'], dv['Wq1'])
            dec_layer.mha1.Wk = self.update_param(dec_layer.mha1.Wk, dec_layer_grads['mha1_grads']['dWk'], dm['Wk1'], dv['Wk1'])
            dec_layer.mha1.Wv = self.update_param(dec_layer.mha1.Wv, dec_layer_grads['mha1_grads']['dWv'], dm['Wv1'], dv['Wv1'])
            dec_layer.mha1.Wo = self.update_param(dec_layer.mha1.Wo, dec_layer_grads['mha1_grads']['dWo'], dm['Wo1'], dv['Wo1'])
            dec_layer.mha2.Wq = self.update_param(dec_layer.mha2.Wq, dec_layer_grads['mha2_grads']['dWq'], dm['Wq2'], dv['Wq2'])
            dec_layer.mha2.Wk = self.update_param(dec_layer.mha2.Wk, dec_layer_grads['mha2_grads']['dWk'], dm['Wk2'], dv['Wk2'])
            dec_layer.mha2.Wv = self.update_param(dec_layer.mha2.Wv, dec_layer_grads['mha2_grads']['dWv'], dm['Wv2'], dv['Wv2'])
            dec_layer.mha2.Wo = self.update_param(dec_layer.mha2.Wo, dec_layer_grads['mha2_grads']['dWo'], dm['Wo2'], dv['Wo2'])
            dec_layer.ffn.W1 = self.update_param(dec_layer.ffn.W1, dec_layer_grads['ffn_grads']['dW1'], dm['W1'], dv['W1'])
            dec_layer.ffn.b1 = self.update_param(dec_layer.ffn.b1, dec_layer_grads['ffn_grads']['db1'], dm['b1'], dv['b1'])
            dec_layer.ffn.W2 = self.update_param(dec_layer.ffn.W2, dec_layer_grads['ffn_grads']['dW2'], dm['W2'], dv['W2'])
            dec_layer.ffn.b2 = self.update_param(dec_layer.ffn.b2, dec_layer_grads['ffn_grads']['db2'], dm['b2'], dv['b2'])
            dec_layer.layernorm1.gamma = self.update_param(dec_layer.layernorm1.gamma, dec_layer_grads['d_gamma1'].flatten(), dm['g1'], dv['g1'])
            dec_layer.layernorm1.beta = self.update_param(dec_layer.layernorm1.beta, dec_layer_grads['d_beta1'].flatten(), dm['beta1'], dv['beta1'])
            dec_layer.layernorm2.gamma = self.update_param(dec_layer.layernorm2.gamma, dec_layer_grads['d_gamma2'].flatten(), dm['g2'], dv['g2'])
            dec_layer.layernorm2.beta = self.update_param(dec_layer.layernorm2.beta, dec_layer_grads['d_beta2'].flatten(), dm['beta2'], dv['beta2'])
            dec_layer.layernorm3.gamma = self.update_param(dec_layer.layernorm3.gamma, dec_layer_grads['d_gamma3'].flatten(), dm['g3'], dv['g3'])
            dec_layer.layernorm3.beta = self.update_param(dec_layer.layernorm3.beta, dec_layer_grads['d_beta3'].flatten(), dm['beta3'], dv['beta3'])

        # 4. FIXED: Update the unified vocabulary sequence output head parameters
        if self.m_out is None:
            self.m_out = {
                'W_vocab': np.zeros_like(model.output_layer.W_vocab),
                'b_vocab': np.zeros_like(model.output_layer.b_vocab)
            }
            self.v_out = {k: np.zeros_like(v) for k, v in self.m_out.items()}

        om, ov = self.m_out, self.v_out
        
        # Apply Adam momentum updates to the single vocabulary projection array
        model.output_layer.W_vocab = self.update_param(
            model.output_layer.W_vocab, 
            grads['output_layer_grads']['dW_vocab'], 
            om['W_vocab'], 
            ov['W_vocab']
        )
        model.output_layer.b_vocab = self.update_param(
            model.output_layer.b_vocab, 
            grads['output_layer_grads']['db_vocab'], 
            om['b_vocab'], 
            ov['b_vocab']
        )

        return embedding_matrix


print("Robust, decoupled Adam Optimizer engine successfully configured!")

