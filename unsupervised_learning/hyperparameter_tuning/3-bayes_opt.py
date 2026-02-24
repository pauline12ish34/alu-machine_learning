#!/usr/bin/env python3
"""[summary]
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """[summary]
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """[summary]

        Args:
            f ([type]): [description]
            X_init ([type]): [description]
            Y_init ([type]): [description]
            bounds ([type]): [description]
            ac_samples ([type]): [description]
            l (int, optional): [description]. Defaults to 1.
            sigma_f (int, optional): [description]. Defaults to 1.
            xsi (float, optional): [description]. Defaults to 0.01.
            minimize (bool, optional): [description]. Defaults to True.
        """
        self.f = f
        self.gp = GP(X_init,
                     Y_init, l,
                     sigma_f
                     )

        t_min, t_max = bounds
        self.X_s = np.linspace(t_min, t_max,
                               ac_samples).reshape(-1, 1)

        self.xsi = xsi

        self.minimize = minimize
