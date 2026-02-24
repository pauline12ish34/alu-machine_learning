#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def moving_average(data, beta):
    """[summary]

    Args:
        data ([type]): [description]
        beta ([type]): [description]

    Returns:
        [type]: [description]
    """
    v1 = 0
    mv_avrg = []
    for i in range(0, len(data)):
        v1 = beta * v1 + (1 - beta) * data[i]
        v2 = v1/(1 - beta ** (i + 1))
        mv_avrg.append(v2)
    return mv_avrg
