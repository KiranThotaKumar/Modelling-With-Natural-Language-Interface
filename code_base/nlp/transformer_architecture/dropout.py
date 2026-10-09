#\Modelling\nlp\transformer_architecture\dropout.py

import numpy as np

class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None  # Stores random binary operational mask

    def __call__(self, x, training=True):
        if not training or self.rate == 0.0:
            return x
        # Math: Mask drawn from Bernoulli distribution scaled by 1/(1-p)
        self.mask = (np.random.rand(*x.shape) > self.rate) / (1.0 - self.rate)
        return x * self.mask

    def backward(self, d_out):
        # FIXED: Safeguard evaluation loops if backward is called out of cycle
        if self.mask is None:
            return d_out
        # Math: dX = (d_out * Mask) / (1-p)
        return d_out * self.mask

