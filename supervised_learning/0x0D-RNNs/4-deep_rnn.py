#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """[summary]

    Args:
        rnn_cells ([type]): [description]
        X ([type]): [description]
        h_0 ([type]): [description]

    Returns:
        [type]: [description]
    """
    l, _, h = h_0.shape
    t_max, m, i = X.shape
    o = rnn_cells[-1].Wy.shape[1]
    H = np.zeros((1, l, m, h))
    H[0, :, :, :] = h_0
    Y = np.zeros((t_max, m, o))
    for t in range(t_max):
        H_t = np.zeros((l, m, h))
        for lay in range(l):
            rnn_cell = rnn_cells[lay]
            h_prev = H[t, lay, :, :]
            if lay == 0:
                x_t = X[t, :, :]
            h_next, y = rnn_cell.forward(
                h_prev, x_t)
            x_t = h_next
            H_t[lay, :, :] = h_next
        H = np.append(
            H, H_t[np.newaxis, :, :, :], axis=0)
        Y[t, :, :] = y
    return H, Y
