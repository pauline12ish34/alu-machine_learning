#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """[summary]

    Args:
        A_prev ([type]): [description]
        kernel_shape ([type]): [description]
        stride (tuple, optional): [description]. Defaults to (1, 1).
        mode (str, optional): [description]. Defaults to 'max'.

    Returns:
        [type]: [description]
    """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw) = kernel_shape
    (sh, sw) = stride
    pool = np.average
    if mode == 'max':
        pool = np.max
    z_h = int((h_prev - kh) / sh + 1)
    z_w = int((w_prev - kw) / sw + 1)
    z = np.zeros((m, z_h, z_w, c_prev))
    for i in range(z_h):
        for j in range(z_w):
            z[:, i, j, :] = pool(A_prev[:, i * sh: i * sh + kh,
                                        j * sw: j * sw + kw, :],
                                 axis=(1, 2))
    return z
