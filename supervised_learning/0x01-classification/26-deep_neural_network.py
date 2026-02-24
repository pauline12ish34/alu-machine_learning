#!/usr/bin/env python3
"""[summary]

Raises:
    TypeError: [description]
    ValueError: [description]
    TypeError: [description]
    TypeError: [description]
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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
        self.__cache['A0'] = X
        for i in range(0, self.__L):
            W = self.__weights["W{}".format(i + 1)]
            b = self.__weights["b{}".format(i + 1)]
            A = self.__cache["A{}".format(i)]
            Z = np.matmul(W, A) + b
            self.__cache["A{}".format(i + 1)] = self.sigmoid(Z)
        return self.__cache["A{}".format(i + 1)], self.__cache

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
        A = self.forward_prop(X)[0]
        Y_hat = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return Y_hat, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """[summary]

        Args:
            Y ([type]): [description]
            cache ([type]): [description]
            alpha (float, optional): [description]. Defaults to 0.05.
        """
        m = Y.shape[1]
        weights_c = self.weights.copy()
        for i in reversed(range(1, self.__L + 1)):
            w_k = "W{}".format(i)
            b_k = "b{}".format(i)
            A = cache['A{}'.format(i)]
            if i == self.L:
                dz = A - Y
            else:
                dz = A * (1 - A) * np.matmul(
                    weights_c["W{}".format(i + 1)].T, dz)
            self.weights[w_k] = self.weights[w_k] - alpha * \
                (np.matmul(dz, cache["A{}".format(i - 1)].T) / m)
            self.weights[b_k] = self.weights[b_k] - alpha * \
                (np.sum(dz, axis=1, keepdims=True) / m)

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

        # cost = []
        # iters = []

        # for i in range(iterations):
        #     self.forward_prop(X)
        #     self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        #     if i % step == 0 or i == iterations:
        #         cost.append(self.cost(Y, self.__A2))
        #         iters.append(i)

        #         if verbose:
        #             print("Cost after {} iterations: {}".
        #                   format(i, self.cost(Y, self.__A2)))

        # if graph:
        #     plt.plot(iters, cost, '-b')
        #     plt.xlabel('iteration')
        #     plt.ylabel('cost')

        #     plt.title('Training Cost')
        #     plt.show()

        # return self.evaluate(X, Y)

        cost = []
        iters = []
        for i in range(iterations):
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

            if i % step == 0 or step == iterations:
                cost.append(self.cost(Y, A))
                iters.append(i)

                if verbose:
                    print("Cost after {} iterations: {}".format(
                        i, self.cost(Y, A)))

        if graph:
            plt.plot(iters, cost, '-b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Trainig Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """[summary]

        Args:
            filename ([type]): [description]
        """
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """[summary]

        Args:
            filename ([type]): [description]

        Returns:
            [type]: [description]
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
