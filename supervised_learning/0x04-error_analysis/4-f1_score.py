#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """[summary]

    Args:
        confusion ([type]): [description]

    Returns:
        [type]: [description]
    """
    return (2 * precision(confusion) * sensitivity(confusion)) / \
        (precision(confusion) + sensitivity(confusion))
