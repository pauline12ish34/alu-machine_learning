#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """[summary]

    Args:
        alpha ([type]): [description]
        beta1 ([type]): [description]
        beta2 ([type]): [description]
        epsilon ([type]): [description]
        var ([type]): [description]
        grad ([type]): [description]
        v ([type]): [description]
        s ([type]): [description]
        t ([type]): [description]

    Returns:
        [type]: [description]
    """

    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad ** 2
    v_bc = v / (1 - beta1 ** t)
    s_bc = s / (1 - beta2 ** t)
    var = var - alpha * (v_bc / (s_bc ** (1/2) + epsilon))
    return var, v, s
