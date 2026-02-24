#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """[summary]

    Args:
        max_seq_len ([type]): [description]
        dm ([type]): [description]

    Returns:
        [type]: [description]
    """
    pos = np.arange(
        max_seq_len)[:, np.newaxis]
    d = np.arange(
        dm)[np.newaxis, :]
    PE = pos / np.power(
        10000, (2 * (d//2
                     )) / np.float32(dm))
    PE[:, 0::2] = np.sin(PE[:, 0::2])
    PE[:, 1::2] = np.cos(PE[:, 1::2])
    return PE
