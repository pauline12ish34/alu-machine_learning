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


class LSTMCell:
    """[summary]
    """

    def __init__(self, i, h, o):
        """[summary]

        Args:
            i ([type]): [description]
            h ([type]): [description]
            o ([type]): [description]
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """[summary]

        Args:
            h_prev ([type]): [description]
            c_prev ([type]): [description]
            x_t ([type]): [description]

        Returns:
            [type]: [description]
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        f_t = sigmoid(np.matmul(
            h_x, self.Wf) + self.bf
        )
        u_t = sigmoid(np.matmul(
            h_x, self.Wu) + self.bu)
        C_t_tilde = np.tanh(
            np.matmul(h_x, self.Wc) + self.bc)
        c_r = f_t * c_prev + u_t * C_t_tilde
        o_t = sigmoid(np.matmul(h_x, self.Wo) + self.bo)
        h_r = o_t * np.tanh(c_r)
        return h_r, c_r, softmax(
            np.matmul(h_r, self.Wy
                      ) + self.by)
