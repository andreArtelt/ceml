# -*- coding: utf-8 -*-
import torch
from .losses import l1, l2, lmad, negloglikelihood, min_of_list
from ....costfunctions import CostFunctionDifferentiable


class CostFunctionDifferentiableTorch(CostFunctionDifferentiable):
    """
    Base class of differentiable cost functions implemented in PyTorch.
    """
    def __init__(self):
        super(CostFunctionDifferentiableTorch, self).__init__()
    
    def grad(self):
        """
        Warning
        -------
        Do not use this method!

        Call '.backward()' of the output tensor. After that, the gradient of each variable 'myvar' - that is supposed to have gradient - can be accessed as 'myvar.grad'

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Call '.backward()' of the output tensor.\nAfter that, the gradient of each variable 'myvar' - that is supposed to have gradient - can be accessed as 'myvar.grad'")


class DummyCost(CostFunctionDifferentiableTorch):
    """
    Dummy cost function - always returns zero.
    """
    def __init__(self):
        super(DummyCost, self).__init__()
    
    def score_impl(self, x):
        """
        Computes the loss - always return zero.
        """
        return 0.0


class L1Cost(CostFunctionDifferentiableTorch):
    """
    L1 cost function.
    """
    def __init__(self, x_orig):
        self.x_orig = x_orig

        super(L1Cost, self).__init__()
    
    def score_impl(self, x):
        """
        Computes the loss - l1 norm.
        """
        return l1(x, self.x_orig) 


class L2Cost(CostFunctionDifferentiableTorch):
    """
    L2 cost function.
    """
    def __init__(self, x_orig):
        self.x_orig = x_orig

        super(L2Cost, self).__init__()
    
    def score_impl(self, x):
        """
        Computes the loss - l2 norm.
        """
        return l2(x, self.x_orig) 


class LMadCost(CostFunctionDifferentiableTorch):
    """
    Manhattan distance weighted feature-wise with the inverse median absolute deviation (MAD).
    """
    def __init__(self, x_orig, mad):
        self.x_orig = x_orig
        self.mad = mad

        super(LMadCost, self).__init__()
    
    def score_impl(self, x):
        """
        Computes the loss.
        """
        return lmad(x, self.x_orig, self.mad) 


class MinOfListCost(CostFunctionDifferentiableTorch):
    """
    Minimum distance to a list of data points.
    """
    def __init__(self, dist, samples):
        self.dist = dist
        self.samples = samples

        super(MinOfListCost, self).__init__()
    
    def score_impl(self, x):
        """
        Computes the loss.
        """
        return min_of_list([self.dist(x, x1) for x1 in self.samples])


class NegLogLikelihoodCost(CostFunctionDifferentiableTorch):
    """
    Negative-log-likelihood cost function.
    """
    def __init__(self, input_to_output, y_target):
        self.y_target = y_target
        self.input_to_output = input_to_output

        super(NegLogLikelihoodCost, self).__init__()
    
    def score_impl(self, x):
        """
        Computes the loss - negative-log-likelihood.
        """
        return negloglikelihood(self.input_to_output(x), self.y_target) 


class SquaredError(CostFunctionDifferentiableTorch):
    """
    Squared error cost function.
    """
    def __init__(self, input_to_output, y_target):
        self.y_target = y_target
        self.input_to_output = input_to_output

        super(SquaredError, self).__init__()
    
    def score_impl(self, x):
        """
        Computes the loss - squared error.
        """
        return l2(self.input_to_output(x), self.y_target)


class RegularizedCost(CostFunctionDifferentiableTorch):
    """
    Regularized cost function.
    """
    def __init__(self, penalize_input, penalize_output, C=1.0):
        if not isinstance(penalize_input, CostFunctionDifferentiable):
            raise TypeError("penalize_input has to be an instance of 'CostFunctionDifferentiable'")
        if not isinstance(penalize_output, CostFunctionDifferentiable):
            raise TypeError("penalize_output has to be an instance of 'CostFunctionDifferentiable'")

        self.penalize_input = penalize_input
        self.penalize_output = penalize_output
        self.C = C
        
        super(RegularizedCost, self).__init__()

    def score_impl(self, x):
        """
        Computes the loss.
        """
        return self.C * self.penalize_input(x) + self.penalize_output(x)
