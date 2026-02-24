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


class BidirectionalCell:
    """[summary]
    """

    def __init__(self, i, h, o):
        """[summary]

        Args:
            i ([type]): [description]
            h ([type]): [description]
            o ([type]): [description]
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(i + h + o, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def output(self, H):
        """[summary]

        Args:
            H ([type]): [description]

        Returns:
            [type]: [description]
        """
        Y = []
        for step in range(H.shape[0]):
            y = softmax(np.matmul(
                H[step], self.Wy) + self.by
            )
            Y.append(y)
        return np.array(Y)

    def backward(self, h_next, x_t):
        """[summary]

        Args:
            h_next ([type]): [description]
            x_t ([type]): [description]

        Returns:
            [type]: [description]
        """
        hx_t = np.concatenate(
            (h_next, x_t), axis=-1)
        Whb, bhb = self.Whb, self.bhb
        return np.tanh(hx_t @ Whb + bhb)

    def forward(self, h_prev, x_t):
        """[summary]

        Args:
            h_prev ([type]): [description]
            x_t ([type]): [description]

        Returns:
            [type]: [description]
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)

        return np.tanh(np.matmul(
            h_x, self.Whf
        ) + self.bhf)
