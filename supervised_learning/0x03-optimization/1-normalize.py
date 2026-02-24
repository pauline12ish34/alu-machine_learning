#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def normalize(X, m, s):
    """[summary]

    Args:
        X ([type]): [description]
        m ([type]): [description]
        s ([type]): [description]

    Returns:
        [type]: [description]
    """
    return (X - m) / s
