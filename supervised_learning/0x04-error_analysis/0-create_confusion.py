#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """[summary]

    Args:
        labels ([type]): [description]
        logits ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.matmul(labels.T, logits)
