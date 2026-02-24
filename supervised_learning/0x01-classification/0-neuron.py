#!/usr/bin/env python3
"""[summary]

    Raises:
        TypeError: [description]
        ValueError: [description]
"""
import numpy as np


class Neuron:
    """[summary]
    """

    def __init__(self, nx):
        """[summary]

        Args:
            nx ([type]): [description]

        Raises:
            TypeError: [description]
            ValueError: [description]
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
