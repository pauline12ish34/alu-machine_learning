#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """[summary]

    Args:
        rnn_cell ([type]): [description]
        X ([type]): [description]
        h_0 ([type]): [description]

    Returns:
        [type]: [description]
    """
    # t, m, i = X.shape
    H = []
    Y = []
    H.append(h_0)
    for step in range(X.shape[0]):
        h, y = rnn_cell.forward(H[-1], X[step])
        H.append(h)
        Y.append(y)
    return np.array(H), np.array(Y)
