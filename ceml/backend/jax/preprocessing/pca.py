# -*- coding: utf-8 -*-
import jax.numpy as npx
import numpy as np
from ....model import Model
from .affine_preprocessing import AffinePreprocessing


class PCA(Model, AffinePreprocessing):
    """
    Wrapper for PCA - Principle component analysis.
    """
    def __init__(self, w):
        Model.__init__(self)

        self.w = w

        AffinePreprocessing.__init__(self, self.w, np.zeros(self.w.shape[0]))

    def predict(self, x):
        """
        Computes the forward pass.
        """
        return npx.dot(self.w, x)
