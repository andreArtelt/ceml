# -*- coding: utf-8 -*-
import jax.numpy as npx
from ....model import Model


class PCA(Model):
    """
    Wrapper for PCA - Principle component analysis.
    """
    def __init__(self, w):
        self.w = w
        
        super(PCA, self).__init__()
    
    def predict(self, x):
        """
        Computes the forward pass.
        """
        return npx.dot(self.w, x)
