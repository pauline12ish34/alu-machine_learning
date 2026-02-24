#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np
from scipy.stats import norm
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
        self.gp = GP(X_init, Y_init, l, sigma_f)
        b_min, b_max = bounds
        self.X_s = np.linspace(b_min, b_max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        mu, sigma = self.gp.predict(self.X_s)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        if self.minimize:
            min_val = np.min(self.gp.Y)
            num = min_val - mu - self.xsi
        else:
            max_val = np.max(self.gp.Y)
            num = mu - max_val - self.xsi
        Z = num.astype(float) / sigma.astype(float)
        Z[Z == np.inf] = 0
        cdf_Z = norm.cdf(Z)
        pdf_Z = norm.pdf(Z)
        EI = num * cdf_Z + sigma * pdf_Z
        best_X = self.X_s[np.argmax(EI)]
        return best_X, EI.reshape(-1)
