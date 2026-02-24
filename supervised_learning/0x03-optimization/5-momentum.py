#!/usr/bin/env python3
"""[summary]
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """[summary]

    Args:
        alpha ([type]): [description]
        beta1 ([type]): [description]
        var ([type]): [description]
        grad ([type]): [description]
        v ([type]): [description]

    Returns:
        [type]: [description]
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - v * alpha
    return var, v
