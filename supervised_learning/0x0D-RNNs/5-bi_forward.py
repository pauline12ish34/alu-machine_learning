#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


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
