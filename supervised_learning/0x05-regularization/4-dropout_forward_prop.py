#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """[summary]

    Args:
        X ([type]): [description]
        weights ([type]): [description]
        L ([type]): [description]
        keep_prob ([type]): [description]

    Returns:
        [type]: [description]
    """

    cache = {}
    cache['A0'] = X

    for i in range(L):
        z = np.matmul(weights['W' + str(i + 1)],
                      cache['A' + str(i)]) + weights['b' + str(i + 1)]
        if i == L - 1:
            cache['A' + str(i + 1)] = np.exp(z) / \
                np.sum(np.exp(z), axis=0, keepdims=True)
        else:
            cache['A' + str(i + 1)] = np.tanh(z)
            A_shape = cache['A' + str(i + 1)].shape
            cache['D' + str(i + 1)] = 1 * (np.random.rand(
                A_shape[0],
                A_shape[1]) < keep_prob)
            cache['A' + str(i + 1)] = np.multiply(cache['A' +
                                                        str(i + 1)],
                                                  cache['D' + str(i + 1)])
            cache['A' + str(i + 1)] /= keep_prob

    return cache
