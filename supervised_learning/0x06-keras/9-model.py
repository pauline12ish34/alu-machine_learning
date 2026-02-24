#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def save_model(network, filename):
    """[summary]

    Args:
        network ([type]): [description]
        filename ([type]): [description]

    Returns:
        [type]: [description]
    """
    network.save(filename)
    return None


def load_model(filename):
    """[summary]

    Args:
        filename ([type]): [description]

    Returns:
        [type]: [description]
    """
    return K.models.load_model(filename)
