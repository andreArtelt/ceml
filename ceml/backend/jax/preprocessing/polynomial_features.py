# -*- coding: utf-8 -*-
import jax.numpy as npx
from ....model import Model


class PolynomialFeatures(Model):
    """
    Wrapper for polynomial feature transformation.
    """
    def __init__(self, powers):
        self.powers = powers

        super(PolynomialFeatures, self).__init__()
    
    def predict(self, x):
        """
        Computes the forward pass.
        """
        #return npx.array([npx.prod(npx.power(x, self.powers[i,:])) for i in range(self.powers.shape[0])])
        return npx.array([npx.exp(npx.sum(npx.log(npx.power(x, self.powers[i,:])))) for i in range(self.powers.shape[0])])   # NOTE: Because jax can not compute the gradient of npx.prod, we use the exp-log trick so that we can use npx.sum instead of npx.prod - note that jax can compute the gradient of npx.sum
