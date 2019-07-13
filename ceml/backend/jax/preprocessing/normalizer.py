# -*- coding: utf-8 -*-
import jax.numpy as npx

from ....model import Model


class Normalizer(Model):
    """
    Wrapper for the normalizer.
    """
    def __init__(self):
        super(Normalizer, self).__init__()
    
    def predict(self, x):
        """
        Computes the forward pass.
        """
        return x / npx.linalg.norm(x)