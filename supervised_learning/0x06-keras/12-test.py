#!/usr/bin/env python3
"""[summary]

    Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """[summary]

    Args:
        network ([type]): [description]
        data ([type]): [description]
        labels ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    return network.evaluate(x=data, y=labels,
                            verbose=verbose)
