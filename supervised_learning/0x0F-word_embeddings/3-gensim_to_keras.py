#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
from gensim.models import Word2Vec
import tensorflow.keras as keras


def gensim_to_keras(model):
    """[summary]

    Args:
        model ([type]): [description]

    Returns:
        [type]: [description]
    """
    return model.wv.get_keras_embedding(
        train_embeddings=True
    )
