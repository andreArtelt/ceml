# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class CostFunction(ABC):
    """Base class of a cost function.

    Note
    ----
    The class :class:`CostFunction` can not be instantiated because it contains an abstract method. 
    """
    def __init__(self, input_to_output=None):
        self.input_to_output = input_to_output if input_to_output is not None else lambda z: z

        super().__init__()
    
    def score(self, x):
        return self.score_impl(self.input_to_output(x))

    @abstractmethod
    def score_impl(self, x):
        """Applying the cost function to a given input.

        Abstract method for computing applying the cost function to a given input `x`.

        Note
        ----
        All derived classes must implement this method.
        """
        raise NotImplementedError()
    
    def __call__(self, x):
        return self.score(x)


class CostFunctionDifferentiable(CostFunction):
    """Base class of a differentiable cost function.

    Note
    ----
    The class :class:`CostFunctionDifferentiable` can not be instantiated because it contains an abstract method. 
    """
    def __init__(self, input_to_output=None):
        super(CostFunctionDifferentiable, self).__init__(input_to_output)
    
    @abstractmethod
    def grad(self, mask=None):
        """Computes the gradient.

        Abstract method for computing the gradient of the cost function.

        Returns
        -------
        `callable`
            A function that computes the gradient for a given input.

        Note
        ----
        All derived classes must implement this method.
        """
        raise NotImplementedError()


class RegularizedCost(CostFunction):
    """Regularized cost function.
    
    The :class:`RegularizedCost` class implements a regularized cost function. The cost function is the sum of a regularization term (weighted by the regularization strength `C`) and a term that penalizes wrong predictions.

    Parameters
    ----------
    input_to_output : `callable`
        Function for computing the output from the input. The output is then put into the `penalize_output` function.
    penalize_input : :class:`ceml.costfunctions.costfunctions.CostFunction`
        Regularization of the input.
    penalize_output : :class:`ceml.costfunctions.costfunctions.CostFunction`
        Loss function for the output/prediction.
    C : `float`
        Regularization strength.
    """
    def __init__(self, penalize_input, penalize_output, C=1.0):
        if not isinstance(penalize_input, CostFunction):
            raise TypeError(f"penalize_input has to be an instance of 'CostFunction' but not of '{type(penalize_input)}'")
        if not isinstance(penalize_output, CostFunction):
            raise TypeError(f"penalize_output has to be an instance of 'CostFunction' but not of {type(penalize_output)}")

        self.penalize_input = penalize_input
        self.penalize_output = penalize_output
        self.C = C

        super(RegularizedCost, self).__init__()

    def score_impl(self, x):
        """Applying the cost function to a given input.

        Computes the cost function fo a given input `x`.

        Parameters
        ----------
        x : `numpy.array`
            Value of the unknown variable.

        Returns
        -------
        `float`
            The loss/cost.
        """
        return self.C * self.penalize_input(x) + self.penalize_output(x)
