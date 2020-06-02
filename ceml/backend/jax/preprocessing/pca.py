# -*- coding: utf-8 -*-
import jax.numpy as npx
import numpy as np
from ....model import Model
from .affine_preprocessing import AffinePreprocessing


class PCA(Model, AffinePreprocessing):
    """
    Wrapper for PCA - Principle component analysis.
    """
    def __init__(self, w, **kwds):
        self.w = w

        super().__init__(A=self.w, b=np.zeros(self.w.shape[0]), **kwds)

    def predict(self, x):
        """
        Computes the forward pass.
        """
        return npx.dot(self.w, x)
