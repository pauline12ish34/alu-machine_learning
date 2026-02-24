#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


class RNNCell:
    """[summary]
    """

    def __init__(self, i, h, o):
        """[summary]

        Args:
            i ([type]): [description]
            h ([type]): [description]
            o ([type]): [description]
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
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

        xh = np.concatenate((h_prev, x_t), axis=1)
        aux = np.tanh(np.matmul(xh,
                                self.Wh) + self.bh)
        y = np.matmul(aux, self.Wy) + self.by
        return aux, self.softmax(y)

    def softmax(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """

        return np.exp(
            x - np.max(
                x, axis=1,
                keepdims=True
            )) / np.sum(
            np.exp(
                x - np.max(x,
                           axis=1,
                           keepdims=True)),
            axis=1, keepdims=True)
