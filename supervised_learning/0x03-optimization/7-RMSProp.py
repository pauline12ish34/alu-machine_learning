#!/usr/bin/env python3
"""[summary]
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """[summary]

    Args:
        alpha ([type]): [description]
        beta2 ([type]): [description]
        epsilon ([type]): [description]
        var ([type]): [description]
        grad ([type]): [description]
        s ([type]): [description]

    Returns:
        [type]: [description]
    """
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / (s ** (1/2) + epsilon)
    return var, s
