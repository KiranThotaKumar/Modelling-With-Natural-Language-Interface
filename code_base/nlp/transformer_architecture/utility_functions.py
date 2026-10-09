# \Modelling\nlp\transformer_architecture\utility_functions.py


import numpy as np


def sigmoid(x):
    # Numerical stability clip to prevent exponential overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

#def tanh_activation(x):
    #return np.tanh(x)

def softmax(x):
    # Numerically stable softmax along the last axis
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
