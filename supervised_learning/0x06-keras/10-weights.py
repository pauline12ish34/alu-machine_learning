#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """[summary]

    Args:
        network ([type]): [description]
        filename ([type]): [description]
        save_format (str, optional): [description]. Defaults to 'h5'.

    Returns:
        [type]: [description]
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """[summary]

    Args:
        network ([type]): [description]
        filename ([type]): [description]

    Returns:
        [type]: [description]
    """
    network.load_weights(filename)
    return None
