#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """[summary]

    Args:
        alpha ([type]): [description]
        decay_rate ([type]): [description]
        global_step ([type]): [description]
        decay_step ([type]): [description]

    Returns:
        [type]: [description]
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
