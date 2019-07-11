# -*- coding: utf-8 -*-
import numpy as np
from jax import grad
import jax.numpy as npx
from .losses import l1, l2, lmad, custom_dist, negloglikelihood, min_of_list
from ....costfunctions import CostFunctionDifferentiable


class CostFunctionDifferentiableJax(CostFunctionDifferentiable):
    """
    Base class of differentiable cost functions implemented in jax.
    """
    def __init__(self, input_to_output=None):
        super(CostFunctionDifferentiableJax, self).__init__(input_to_output)
    
    def grad(self, mask=None):
        """
        Computes the gradient with respect to the input.

        Parameters
        ----------
        mask : `numpy.array`, optional
            A mask that is multiplied elementwise to the gradient - can be used to mask some features/dimensions.

            If `mask` is None, the gradient is not masked.

            The default is None.

        Returns
        -------
        `callable`
            The gradient.
        """
        if mask is not None:
            cost_grad = grad(self.score)
            return lambda x: npx.multiply(cost_grad(x), mask)
        else:
            return grad(self.score)


class TopKMinOfListDistCost(CostFunctionDifferentiableJax):
    """
    Computes the sum of the distances to the k closest samples.
    """
    def __init__(self, dist, samples, k, input_to_output=None):
        self.dist = dist
        self.samples = samples
        self.k = k

        super(TopKMinOfListDistCost, self).__init__(input_to_output)
    
    def score_impl(self, x):
        """
        Computes the loss.
        """
        d = npx.array([self.dist(x, x1) for x1 in self.samples])
        return npx.sum(d[npx.argsort(d)[:self.k]])


class DummyCost(CostFunctionDifferentiableJax):
    """
    Dummy cost function - always returns zero.
    """
    def __init__(self):
        super(DummyCost, self).__init__()
    
    def score_impl(self, x):
        """
        Computes the loss - always returns zero.
        """
        return 0.0


class L1Cost(CostFunctionDifferentiableJax):
    """
    L1 cost function.
    """
    def __init__(self, x_orig, input_to_output=None):
        self.x_orig = x_orig

        super(L1Cost, self).__init__(input_to_output)
    
    def score_impl(self, x):
        """
        Computes the loss - l1 norm.
        """
        return l1(x, self.x_orig) 


class L2Cost(CostFunctionDifferentiableJax):
    """
    L2 cost function.
    """
    def __init__(self, x_orig, input_to_output=None):
        self.x_orig = x_orig

        super(L2Cost, self).__init__(input_to_output)
    
    def score_impl(self, x):
        """
        Computes the loss - l2 norm.
        """
        return l2(x, self.x_orig) 


class LMadCost(CostFunctionDifferentiableJax):
    """
    Manhattan distance weighted feature-wise with the inverse median absolute deviation (MAD).
    """
    def __init__(self, x_orig, mad, input_to_output=None):
        self.x_orig = x_orig
        self.mad = mad

        super(LMadCost, self).__init__(input_to_output=None)
    
    def score_impl(self, x):
        """
        Computes the loss.
        """
        return lmad(x, self.x_orig, self.mad) 


class MinOfListDistCost(CostFunctionDifferentiableJax):
    """
    Minimum distance to a list of data points.
    """
    def __init__(self, dist, samples, input_to_output=None):
        self.dist = dist
        self.samples = samples

        super(MinOfListDistCost, self).__init__(input_to_output)
    
    def score_impl(self, x):
        """
        Computes the loss.
        """
        return min_of_list([self.dist(x, x1) for x1 in self.samples])


class MinOfListDistExCost(CostFunctionDifferentiableJax):
    """
    Minimum distance to a list of data points.

    In contrast to :class:`MinOfListDistCost`, :class:`MinOfListDistExCost` uses a user defined metric matrix (distortion of the Euclidean distance).
    """
    def __init__(self, omegas, samples, input_to_output=None):
        self.omegas = omegas
        self.samples = samples

        super(MinOfListDistExCost, self).__init__(input_to_output)
    
    def score_impl(self, x):
        """
        Computes the loss.
        """
        return min_of_list([custom_dist(x, x1, omega) for x1, omega in zip(self.samples, self.omegas)])


class NegLogLikelihoodCost(CostFunctionDifferentiableJax):
    """
    Negative-log-likelihood cost function.
    """
    def __init__(self, input_to_output, y_target):
        self.y_target = y_target
        
        super(NegLogLikelihoodCost, self).__init__(input_to_output)
    
    def score_impl(self, y):
        """
        Computes the loss - negative-log-likelihood.
        """
        return negloglikelihood(y, self.y_target) 


class SquaredError(CostFunctionDifferentiableJax):
    """
    Squared error cost function.
    """
    def __init__(self, input_to_output, y_target):
        self.y_target = y_target

        super(SquaredError, self).__init__(input_to_output)
    
    def score_impl(self, y):
        """
        Computes the loss - squared error.
        """
        return l2(y, self.y_target)


class RegularizedCost(CostFunctionDifferentiableJax):
    """
    Regularized cost function.
    """
    def __init__(self, penalize_input, penalize_output, C=1.0):
        if not isinstance(penalize_input, CostFunctionDifferentiable):
            raise TypeError(f"penalize_input has to be an instance of 'CostFunctionDifferentiable' but not of '{type(penalize_input)}'")
        if not isinstance(penalize_output, CostFunctionDifferentiable):
            raise TypeError(f"penalize_output has to be an instance of 'CostFunctionDifferentiable' but not of {type(penalize_output)}")

        self.penalize_input = penalize_input
        self.penalize_output = penalize_output
        self.C = C

        super(RegularizedCost, self).__init__()

    def score_impl(self, x):
        """
        Computes the loss.
        """
        return self.C * self.penalize_input(x) + self.penalize_output(x)
