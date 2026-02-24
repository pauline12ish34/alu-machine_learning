#!/usr/bin/env python3
"""[summary]"""
import numpy as np


def specificity(confusion):
    """[summary]

    Args:
        confusion ([type]): [description]

    Returns:
        [type]: [description]
    """
    fp = np.sum(confusion, axis=0) - np.diagonal(confusion)
    tn = (
        (np.sum(confusion) - np.sum(confusion, axis=1))) - fp
    return tn / ((np.sum(confusion) - np.sum(confusion, axis=1)))
