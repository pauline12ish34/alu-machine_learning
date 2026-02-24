#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def one_hot_decode(one_hot):
    """[summary]

    Args:
        Y ([type]): [description]
        classes ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
