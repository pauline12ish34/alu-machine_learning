#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """[summary]

    Args:
        A_prev ([type]): [description]
        W ([type]): [description]
        b ([type]): [description]
        activation ([type]): [description]
        padding (str, optional): [description]. Defaults to "same".
        stride (tuple, optional): [description]. Defaults to (1, 1).

    Returns:
        [type]: [description]
    """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, c_prev, c_new) = W.shape
    (sh, sw) = stride
    pw, ph = 0, 0

    if padding == 'same':
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2)
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2)

    pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                 mode='constant',
                 constant_values=0)
    z_h = int((h_prev + 2 * ph - kh) / sh + 1)
    z_w = int((w_prev + 2 * pw - kw) / sw + 1)
    z = np.zeros((m, z_h, z_w, c_new))
    for i in range(z_h):
        for j in range(z_w):
            for k in range(c_new):
                z[:, i, j, k] = (pad[:, i * sh: i * sh + kh,
                                     j * sw: j * sw + kw, :] *
                                 W[:, :, :, k]).sum(axis=(1, 2, 3))
                z[:, i, j, k] = activation(z[:, i, j, k] + b[0, 0, 0, k])
    return z
