#\Modelling\nlp\transformer_architecture\embeddings.py

import numpy as np


def get_embeddings(sequences, embedding_matrix):
    # Math: Pull coordinate matrix rows matching token index vectors
    return embedding_matrix[sequences]
