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
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__W

    @property
    def b(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__b

    @property
    def A(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__A

    def forward_prop(self, X):
        """[summary]

        Args:
            X ([type]): [description]

        Returns:
            [type]: [description]
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = self.sigmoid(Z)
        return self.__A

    def sigmoid(self, Z):
        """[summary]

        Args:
            Z ([type]): [description]

        Returns:
            [type]: [description]
        """
        return 1 / (1 + np.exp(-Z))

    def cost(self, Y, A):
        """[summary]

        Args:
            Y ([type]): [description]
            A ([type]): [description]

        Returns:
            [type]: [description]
        """
        m = Y.shape[1]
        cost = -1 * (1 / m) * np.sum(Y * np.log(A) +
                                     (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """[summary]

        Args:
            X ([type]): [description]
            Y ([type]): [description]

        Returns:
            [type]: [description]
        """
        A = self.forward_prop(X)
        Y_hat = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return Y_hat, cost
