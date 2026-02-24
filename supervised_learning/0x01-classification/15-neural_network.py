#!/usr/bin/env python3
"""[summary]

Raises:
    TypeError: [description]
    ValueError: [description]
    TypeError: [description]
    ValueError: [description]

Returns:
    [type]: [description]
"""
import numpy as np
import matplotlib.pyplot as plt


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
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # layer 2
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__W1

    @property
    def b1(self):
        """[summary]

        Returns:
            [type]: [description]
        """""
        return self.__b1

    @property
    def A1(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__A1

    @property
    def W2(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__W2

    @property
    def b2(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__b2

    @property
    def A2(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__A2

    def sigmoid(self, Z):
        """[summary]

        Args:
            Z ([type]): [description]

        Returns:
            [type]: [description]
        """
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """[summary]

        Args:
            X ([type]): [description]

        Returns:
            [type]: [description]
        """

        # layer 1
        Z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = self.sigmoid(Z1)

        # layer 2
        Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = self.sigmoid(Z2)

        return (self.A1, self.A2)

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
        A = self.forward_prop(X)[1]
        Y_hat = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return Y_hat, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """[summary]

        Args:
            X ([type]): [description]
            Y ([type]): [description]
            A1 ([type]): [description]
            A2 ([type]): [description]
            alpha (float, optional): [description]. Defaults to 0.05.
        """

        dZ2 = A2 - Y
        m = A1.shape[1]
        dW2 = np.matmul(A1, dZ2.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = (1 - A1) * A1 * np.matmul(self.__W2.T, dZ2)
        dW1 = np.matmul(X, dZ1.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        self.__W2 -= (alpha * dW2).T
        self.__b2 -= alpha * db2
        self.__W1 -= (alpha * dW1).T
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=500, alpha=0.05,
              verbose=True, graph=True, step=100):
        """[summary]

        Args:
            X ([type]): [description]
            Y ([type]): [description]
            iterations (int, optional): [description]. Defaults to 500.
            alpha (float, optional): [description]. Defaults to 0.05.
            verbose (bool, optional): [description]. Defaults to True.
            graph (bool, optional): [description]. Defaults to True.
            step (int, optional): [description]. Defaults to 100.

        Raises:
            TypeError: [description]
            ValueError: [description]
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

        if verbose is True or graph is True:

            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        cost = []
        iters = []

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

            if i % step == 0 or i == iterations:
                cost.append(self.cost(Y, self.__A2))
                iters.append(i)

                if verbose:
                    print("Cost after {} iterations: {}".
                          format(i, self.cost(Y, self.__A2)))

        if graph:
            plt.plot(iters, cost, '-b')
            plt.xlabel('iteration')
            plt.ylabel('cost')

            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
