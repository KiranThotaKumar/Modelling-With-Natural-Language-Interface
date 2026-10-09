#Modelling\nlp\transformer_architecture\feed_forward_network.py

import numpy as np


class FeedForwardNetwork:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

        # Math Dimensions: W1 maps (d_model -> d_ff) expansion space
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((1, d_ff))

        # Math Dimensions: W2 maps (d_ff -> d_model) compression space
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((1, d_model))

        self.cache = None

    def __call__(self, x):
        # Forward Matrix Transform: X @ W1 + b1 -> Linear1 (B, S, d_ff)
        linear1_output = np.matmul(x, self.W1) + self.b1
        
        # Non-Linear Activation Mapping: max(0, Linear1) -> ReLU Gate
        relu_output = np.maximum(0, linear1_output)

        # Forward Matrix Transform: ReLU @ W2 + b2 -> Output (B, S, d_model)
        output = np.matmul(relu_output, self.W2) + self.b2

        # Cache structural forward states for exact chain rule backward alignment
        self.cache = (x, linear1_output, relu_output)
        return output

    def backward(self, d_output):
        x, linear1_output, relu_output = self.cache

        # Flatten 3D matrices (B, S, D) -> (B*S, D) to unify position-independent tokens
        relu_output_flat = relu_output.reshape(-1, self.d_ff)
        d_output_flat = d_output.reshape(-1, self.d_model)
        x_flat = x.reshape(-1, self.d_model)

        # 1. Linear2 Backward: dW2 = (X_relu)^T @ dO
        dW2 = np.matmul(relu_output_flat.T, d_output_flat)
        # 1. Linear2 Backward: db2 = sum over all independent tokens
        db2 = np.sum(d_output_flat, axis=0, keepdims=True)

        # 2. Propagate error gradient past Linear2 Matrix frame: dX_relu = dO @ (W2)^T
        d_relu_output = np.matmul(d_output, self.W2.T)

        # 3. Activation Gate Backward: Multiply by indicator function I(Z_1 > 0)
        d_linear1_output = d_relu_output * (linear1_output > 0)
        d_linear1_output_flat = d_linear1_output.reshape(-1, self.d_ff)

        # 4. Linear1 Backward: dW1 = (X_input)^T @ dZ_1
        dW1 = np.matmul(x_flat.T, d_linear1_output_flat)
        # 4. Linear1 Backward: db1 = sum over all independent tokens
        db1 = np.sum(d_linear1_output_flat, axis=0, keepdims=True)

        # 5. Final Gradient to enter lower layer boundary: dX = dZ_1 @ (W1)^T
        dx = np.matmul(d_linear1_output, self.W1.T)

        grads = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }
        return dx, grads

print("Feed-Forward Network with math tracking loaded successfully!")
