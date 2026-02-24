#!/usr/bin/env python3
"""[summary]

Raises:
    TypeError: [description]
    ValueError: [description]
    TypeError: [description]
    ValueError: [description]
"""
import numpy as np


class NeuralNetwork:
    """[summary]
    """

    def __init__(self, nx, nodes):
        """[summary]

        Args:
            nx ([type]): [description]
            nodes ([type]): [description]

        Raises:
            TypeError: [description]
            ValueError: [description]
            TypeError: [description]
            ValueError: [description]
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")

        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # layer 1
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # layer 2
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
