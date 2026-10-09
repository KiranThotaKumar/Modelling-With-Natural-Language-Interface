#\Modelling\nlp\transformer_architecture\loss_functions.py

import numpy as np



def cross_entropy_loss(predictions, targets):
    batch_size = predictions.shape[0]
    # Prevent log(0) errors
    predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
    true_class_probabilities = predictions[np.arange(batch_size), targets.astype(int)]
    loss = -np.log(true_class_probabilities)
    return np.mean(loss)

def mse_loss(predictions, targets):
    # Mean Squared Error for scalar slot outputs
    return np.mean((predictions - targets) ** 2)

print("Activation and Loss functions loaded successfully!")
