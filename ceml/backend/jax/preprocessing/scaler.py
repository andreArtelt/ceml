# -*- coding: utf-8 -*-
import numpy as np
from ....model import Model
from .affine_preprocessing import AffinePreprocessing


class StandardScaler(Model, AffinePreprocessing):
    """
    Wrapper for the standard scaler.
    """
    def __init__(self, mu, sigma):
        Model.__init__(self)
        
        self.mu = mu
        self.sigma = sigma

        A = np.diag(1. / self.sigma)
        AffinePreprocessing.__init__(self, A, -1. * A @ self.mu)
    
    def predict(self, x):
        """
        Computes the forward pass.
        """
        return (x - self.mu) / self.sigma


class MinMaxScaler(Model, AffinePreprocessing):
    """
    Wrapper for the min max scaler.
    """
    def __init__(self, min_, scale):
        Model.__init__(self)

        self.min = min_
        self.scale = scale
    
        AffinePreprocessing.__init__(self, np.diag(self.scale), self.min)

    def predict(self, x):
        """
        Computes the forward pass.
        """
        return self.scale * x + self.min
