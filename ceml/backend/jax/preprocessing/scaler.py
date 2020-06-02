# -*- coding: utf-8 -*-
import numpy as np
from ....model import Model
from .affine_preprocessing import AffinePreprocessing


class StandardScaler(Model, AffinePreprocessing):
    """
    Wrapper for the standard scaler.
    """
    def __init__(self, mu, sigma, **kwds):     
        self.mu = mu
        self.sigma = sigma

        A = np.diag(1. / self.sigma)
        super().__init__(A=A, b=-1. * A @ self.mu, **kwds)
    
    def predict(self, x):
        """
        Computes the forward pass.
        """
        return (x - self.mu) / self.sigma


class MinMaxScaler(Model, AffinePreprocessing):
    """
    Wrapper for the min max scaler.
    """
    def __init__(self, min_, scale, **kwds):
        self.min = min_
        self.scale = scale
    
        super().__init__(A=np.diag(self.scale), b=self.min, **kwds)

    def predict(self, x):
        """
        Computes the forward pass.
        """
        return self.scale * x + self.min
