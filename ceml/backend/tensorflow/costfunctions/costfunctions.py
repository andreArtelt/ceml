# -*- coding: utf-8 -*-
import tensorflow as tf
from .losses import l1, l2, lmad, negloglikelihood, min_of_list
from ....costfunctions import CostFunctionDifferentiable


class CostFunctionDifferentiableTf(CostFunctionDifferentiable):
    """
    Base class of differentiable cost functions implemented in tensorflow.
    """
    def __init__(self, **kwds):
        super().__init__(**kwds)
    
    def grad(self):
        """
        Warning
        -------
        Do not use this method!

        Use 'tf.GradientTape' for computing the gradient.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Use 'tf.GradientTape' for computing the gradient")


class DummyCost(CostFunctionDifferentiableTf):
    """
    Dummy cost function - always returns zero.
    """
    def __init__(self, **kwds):
        super().__init__(**kwds)
    
    def score_impl(self, x):
        return 0.0


class L1Cost(CostFunctionDifferentiableTf):
    """
    L1 cost function.
    """
    def __init__(self, x_orig, **kwds):
        self.x_orig = x_orig

        super().__init__(**kwds)
    
    def score_impl(self, x):
        return l1(x, self.x_orig) 


class L2Cost(CostFunctionDifferentiableTf):
    """
    L2 cost function.
    """
    def __init__(self, x_orig, **kwds):
        self.x_orig = x_orig

        super().__init__(**kwds)
    
    def score_impl(self, x):
        return l2(x, self.x_orig) 


class LMadCost(CostFunctionDifferentiableTf):
    """
    Manhattan distance weighted feature-wise with the inverse median absolute deviation (MAD).
    """
    def __init__(self, x_orig, mad, **kwds):
        self.x_orig = x_orig
        self.mad = mad

        super().__init__(**kwds)
    
    def score_impl(self, x):
        return lmad(x, self.x_orig, self.mad) 


class SquaredError(CostFunctionDifferentiableTf):
    """
    Squared error cost function.
    """
    def __init__(self, y_target, **kwds):
        self.y_target = y_target

        super().__init__(**kwds)
    
    def score_impl(self, y):
        """
        Computes the loss - squared error.
        """
        return l2(y, self.y_target)


class NegLogLikelihoodCost(CostFunctionDifferentiableTf):
    """
    Negative-log-likelihood cost function.
    """
    def __init__(self, y_target, **kwds):
        self.y_target = y_target
        
        super().__init__(**kwds)
    
    def score_impl(self, y):
        return negloglikelihood(y, self.y_target) 


class RegularizedCost(CostFunctionDifferentiableTf):
    """
    Regularized cost function.
    """
    def __init__(self, penalize_input, penalize_output, C=1.0, **kwds):
        if not isinstance(penalize_input, CostFunctionDifferentiable):
            raise TypeError("penalize_input has to be an instance of 'CostFunctionDifferentiable'")
        if not isinstance(penalize_output, CostFunctionDifferentiable):
            raise TypeError("penalize_output has to be an instance of 'CostFunctionDifferentiable'")

        self.penalize_input = penalize_input
        self.penalize_output = penalize_output
        self.C = C

        super().__init__(**kwds)

    def score_impl(self, x):
        return self.C * self.penalize_input.score_impl(x) + self.penalize_output.score_impl(x)
