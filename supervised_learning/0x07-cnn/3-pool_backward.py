#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """[summary]

    Args:
        dA ([type]): [description]
        A_prev ([type]): [description]
        kernel_shape ([type]): [description]
        stride (tuple, optional): [description]. Defaults to (1, 1).
        mode (str, optional): [description]. Defaults to 'max'.

    Returns:
        [type]: [description]
    """
    (m, h_new, w_new, c_new) = dA.shape
    (m, h_prev, w_prev, c) = A_prev.shape
    (kh, kw) = kernel_shape
    (sh, sw) = stride
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for l in range(c_new):
                    if mode == "max":
                        z_0in = np.zeros(kernel_shape)
                        max_n = np.amax(
                            A_prev[i, j * sh: j * sh + kh,
                                   k * sw: k * sw + kw, l])
                        np.place(
                            z_0in, A_prev[i, j * sh: j * sh + kh,
                                          k * sw: k * sw + kw,
                                          l] == max_n, 1)
                        dA_prev[i, j * sh: j * sh + kh,
                                k * sw: k * sw + kw,
                                l] += z_0in * dA[i, j, k, l]
                    else:
                        av_p = dA[i, j, k, l] / (kh * kw)
                        z_0in = np.ones(kernel_shape)
                        dA_prev[i, j * sh: j * sh + kh,
                                k * sw: k * sw + kw, l] += z_0in * av_p
    return dA_prev
