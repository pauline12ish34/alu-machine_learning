#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """[summary]

    Args:
        cost ([type]): [description]
        lambtha ([type]): [description]
        weights ([type]): [description]
        L ([type]): [description]
        m ([type]): [description]

    Returns:
        [type]: [description]
    """
    w = 0
    for layer in range(1, L + 1):
        w += np.linalg.norm(weights['W' + str(layer)])

    return cost + ((lambtha / (2 * m)) * w)
