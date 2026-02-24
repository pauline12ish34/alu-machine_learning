#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """[summary]

    Args:
        Z ([type]): [description]
        gamma ([type]): [description]
        beta ([type]): [description]
        epsilon ([type]): [description]

    Returns:
        [type]: [description]
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    znorm = (Z - mean) / np.sqrt(var + epsilon)
    zv = gamma * znorm + beta
    return zv
