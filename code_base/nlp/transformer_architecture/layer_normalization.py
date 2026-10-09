#\Modelling\nlp\transformer_architecture\layer_normalization.py

import numpy as np


class LayerNormalization:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.gamma = None          # Scale field parameter matrix
        self.beta = None           # Shift field parameter matrix
        self.x_normalized = None   # Cached normalized tensor
        self.mean = None           # Cached feature axis means
        self.variance = None       # Cached feature axis variances
        self.x_input = None        # Cached raw forward inputs

    def __call__(self, x):
        # Cache raw spatial input for analytical gradient evaluation
        self.x_input = x

        # Lazily initialize coordinate weights on the final feature axis
        if self.gamma is None:
            self.gamma = np.ones(x.shape[-1])
            self.beta = np.zeros(x.shape[-1])

        # Evaluate statistical moments along the trailing feature channel axis (-1)
        # Math: mu = (1/D) * sum(x_i) ; var = (1/D) * sum((x_i - mu)^2)
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.variance = np.var(x, axis=-1, keepdims=True)

        # Standard normal feature shift: x_hat = (x - mu) / sqrt(var + eps)
        self.x_normalized = (x - self.mean) / np.sqrt(self.variance + self.epsilon)

        # Apply learnable scaling and shifting fields: Output = gamma * x_hat + beta
        output = self.gamma * self.x_normalized + self.beta
        return output

    def backward(self, d_out):
        x = self.x_input
        feature_count = x.shape[-1]  # D parameter dimension size
        x_centered = x - self.mean
        inv_std = 1.0 / np.sqrt(self.variance + self.epsilon)

        # Math: Dynamic axis collapse (collapses everything except the trailing feature channel dimension)
        reduce_axes = tuple(range(d_out.ndim - 1))

        # FIXED: Explicit axis target tracking prevents 2D vs 3D sequence contraction scrambling
        d_beta = np.sum(d_out, axis=reduce_axes)
        d_gamma = np.sum(d_out * self.x_normalized, axis=reduce_axes)

        # Upstream gradient transformation through the scaling field
        d_x_normalized = d_out * self.gamma
        
        # 1. Variance derivative path calculation
        d_variance = np.sum(d_x_normalized * x_centered, axis=-1, keepdims=True) * -0.5 * inv_std**3
        
        # 2. Mean derivative path calculation incorporating variance shifts
        d_mean = np.sum(d_x_normalized * -inv_std, axis=-1, keepdims=True) + d_variance * np.sum(-2.0 * x_centered, axis=-1, keepdims=True) / feature_count
        
        # 3. Consolidated analytical input layer gradient
        dx = d_x_normalized * inv_std + d_variance * 2.0 * x_centered / feature_count + d_mean / feature_count

        # Force precise shape conformity to match parameter fields perfectly
        d_beta = d_beta.reshape(self.beta.shape)
        d_gamma = d_gamma.reshape(self.gamma.shape)

        return dx, d_gamma, d_beta


print("Layer Normalisation with exact analytical derivatives configured successfully!")

