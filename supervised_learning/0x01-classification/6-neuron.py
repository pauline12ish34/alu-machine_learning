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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """[summary]

        Args:
            X ([type]): [description]
            Y ([type]): [description]
            A ([type]): [description]
            alpha (float, optional): [description]. Defaults to 0.05.
        """
        dZ = A - Y
        m = Y.shape[1]
        self.__W -= alpha * (np.matmul(X, dZ.T) / m).T
        self.__b -= alpha * (np.sum(dZ) / m)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """[summary]

        Args:
            X ([type]): [description]
            Y ([type]): [description]
            iterations (int, optional): [description]. Defaults to 5000.
            alpha (float, optional): [description]. Defaults to 0.05.

        Raises:
            TypeError: [description]
            ValueError: [description]
            TypeError: [description]
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
