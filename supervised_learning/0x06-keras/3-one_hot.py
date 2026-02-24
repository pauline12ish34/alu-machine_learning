#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """[summary]

    Args:
        labels ([type]): [description]
        classes ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    return K.utils.to_categorical(labels, classes)
