# -*- coding: utf-8 -*-
from ....model import Model


class StandardScaler(Model):
    """
    Wrapper for the standard scaler.
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
        super(StandardScaler, self).__init__()
    
    def predict(self, x):
        """
        Computes the forward pass.
        """
        return (x - self.mu) / self.sigma


class MinMaxScaler(Model):
    """
    Wrapper for the min max scaler.
    """
    def __init__(self, min_, scale):
        self.min = min_
        self.scale = scale
        
        super(MinMaxScaler, self).__init__()
    
    def predict(self, x):
        """
        Computes the forward pass.
        """
        return self.scale * x + self.min
