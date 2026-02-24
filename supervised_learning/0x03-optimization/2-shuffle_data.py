#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def shuffle_data(X, Y):
    """[summary]

    Args:
        X ([type]): [description]
        Y ([type]): [description]

    Returns:
        [type]: [description]
    """
    shuff = np.random.permutation(X.shape[0])
    return X[shuff], Y[shuff]
