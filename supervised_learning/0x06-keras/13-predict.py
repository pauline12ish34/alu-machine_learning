#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """[summary]

    Args:
        network ([type]): [description]
        data ([type]): [description]
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    return network.predict(data, verbose=verbose)
