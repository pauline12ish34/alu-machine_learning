#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def save_config(network, filename):
    """[summary]

    Args:
        network ([type]): [description]
        filename ([type]): [description]

    Returns:
        [type]: [description]
    """
    with open(filename, 'w') as fil:
        fil.write(network.to_json())
    return None


def load_config(filename):
    """[summary]

    Args:
        filename ([type]): [description]

    Returns:
        [type]: [description]
    """
    with open(filename, "r") as fil:
        return K.models.model_from_json(fil.read())
