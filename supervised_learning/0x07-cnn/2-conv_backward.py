#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """[summary]

    Args:
        dZ ([type]): [description]
        A_prev ([type]): [description]
        W ([type]): [description]
        b ([type]): [description]
        padding (str, optional): [description]. Defaults to "same".
        stride (tuple, optional): [description]. Defaults to (1, 1).

    Returns:
        [type]: [description]
    """
    (m, h_new, w_new, c_new) = dZ.shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, c_prev, c_new) = W.shape
    sh, sw = stride
    pw, ph = 0, 0
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    dW = np.zeros(W.shape)
    if padding == 'same':
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2) + 1
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2) + 1
    pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                 mode='constant',
                 constant_values=0)
    dA = np.zeros(pad.shape)
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for l in range(c_new):
                    dA[i, j * sh: j * sh + kh,
                       k * sw: k * sw + kw, :] += dZ[i,
                                                     j, k, l] * W[:, :, :, l]
                    dW[:, :, :, l] += pad[i, j * sh: j * sh + kh,
                                          k * sw: k * sw + kw, :] \
                        * dZ[i, j, k, l]
    if padding == 'same':
        dA = dA[:, ph:-ph, pw:-pw, :]
    return dA, dW, db
