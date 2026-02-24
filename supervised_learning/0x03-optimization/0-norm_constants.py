#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def normalization_constants(X):
    """[summary]

    Args:
        X ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
