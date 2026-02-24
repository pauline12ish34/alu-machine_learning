#!/usr/bin/env python3
"""[summary]

Raises:
    TypeError: [description]
    ValueError: [description]
    TypeError: [description]
    TypeError: [description]
"""


import numpy as np


class DeepNeuralNetwork:
    """[summary]
    """

    def __init__(self, nx, layers):
        """[summary]

        Args:
            nx features
            layers array with layers

        Raises:
            TypeError: [description]
            ValueError: [description]
            TypeError: [description]
            TypeError: [description]
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.__nx = nx
        self.__layers = layers
        self.__cache = {}
        self.__weights = {}
        self.__L = len(layers)

        for i in range(0, len(layers)):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            w_lk = "W{}".format(i + 1)
            b_k = "b{}".format(i + 1)

            self.__weights[b_k] = np.zeros((layers[i], 1))

            if i == 0:
                self.__weights[w_lk] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights[w_lk] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

    @property
    def L(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__L

    @property
    def cache(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__cache

    @property
    def weights(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__weights
