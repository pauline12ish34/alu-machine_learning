#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""

import numpy as np


def softmax(x):
    """[summary]

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.exp(x) / np.sum(
        np.exp(x), axis=1, keepdims=True)


def sigmoid(x):
    """[summary]

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    return 1 / (1 + np.exp(-x))


class GRUCell:
    """[summary]
    """

    def __init__(self, i, h, o):
        """[summary]

        Args:
            i ([type]): [description]
            h ([type]): [description]
            o ([type]): [description]
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """[summary]

        Args:
            h_prev ([type]): [description]
            x_t ([type]): [description]

        Returns:
            [type]: [description]
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        z_t = sigmoid(np.matmul(h_x, self.Wz) + self.bz)
        r_t = sigmoid(np.matmul(h_x, self.Wr) + self.br)
        h_x = np.concatenate((r_t * h_prev, x_t), axis=1)
        r_h = (1 - z_t) * h_prev + z_t * np.tanh(
            np.matmul(h_x, self.Wh) + self.bh)
        Z_y = np.matmul(r_h, self.Wy) + self.by
        return r_h, softmax(Z_y)
