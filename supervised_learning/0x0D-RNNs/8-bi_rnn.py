#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """[summary]

    Args:
        bi_cell ([type]): [description]
        X ([type]): [description]
        h_0 ([type]): [description]
        h_t ([type]): [description]

    Returns:
        [type]: [description]
    """
    t_max, m, i = X.shape
    h = h_0.shape[1]
    Hf = np.zeros((1, m, h))
    Hf[0, :, :] = h_0
    for t in range(t_max):
        h_prev = Hf[t]
        x_t = X[t]
        h_next = bi_cell.forward(h_prev, x_t)
        Hf = np.append(Hf, h_next[np.newaxis, :, :], axis=0)
    Hb = np.zeros((1, m, h))
    Hb[0, :, :] = h_t
    for t in range(t_max - 1, -1, -1):
        h_next = Hb[0]
        x_t = X[t]
        h_pev = bi_cell.backward(h_next, x_t)
        Hb = np.append(h_pev[np.newaxis, :, :], Hb, axis=0)
    Hf, Hb = Hf[1:], Hb[0:-1]
    return np.concatenate(
        (Hf, Hb), axis=-1), bi_cell.output(
            np.concatenate((Hf, Hb), axis=-1))
