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
