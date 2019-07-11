# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
from ..costfunctions import CostFunctionDifferentiable


class Optimizer(ABC):
    """
    Abstract base class of an optimizer.

    All optimizers must be derived from the :class:`Optimizer` class.

    Note
    ----
    Any class derived from :class:`Optimizer` has to implement the abstract method `__call__`.
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def __call__(self):
        raise NotImplementedError()


def is_optimizer_grad_based(desc):
    """
    Determines whether a specific optimization algorithm (specified by a description in `desc`) needs a gradient.

    Supported descriptions:

        - nelder-mead: *Gradient-free* Nelder-Mead optimizer (also called Downhill-Simplex)
        - powell: Gradient-free Powell optimizer
        - bfgs: BFGS optimizer
        - cg: Conjugate gradients optimizer

    Parameters
    ----------
    desc : `str`
        Description of the optimization algorithm.
    
    Returns
    -------
    `bool`
        True if the optimization algorithm needs a gradient, False otherwise.
    
    Raises
    ------
    ValueError
        If `desc` contains an invalid description.
    """
    if desc == "nelder-mead":
        return False
    elif desc == "powell":
        return False
    elif desc == "bfgs":
        return True
    elif desc == "cg":
        return True
    else:
        raise ValueError(f"Invalid value of 'desc'.\n'desc' has to be 'nelder-mead', 'powell', 'cg' or 'bfgs' but not '{desc}'")


def desc_to_optim(desc, f, x0, f_grad=None, tol=None, max_iter=None):
    """
    Creates and initializes an optimization algorithm (instance of :class:`ceml.optim.optimizer.Optimizer`) specified by a description of the algorithm.

    Supported descriptions:

        - nelder-mead: *Gradient-free* Nelder-Mead optimizer (also called Simplex search algorithm)
        - powell: *Gradient-free* Powell optimizer
        - bfgs: BFGS optimizer
        - cg: Conjugate gradients optimizer

    Parameters
    ----------
    desc : `str`
        Description of the optimization algorithm.
    f : instance of :class:`ceml.costfunctions.costfunctions.CostFunction` or `callable`
        The objective that has to be minimized.
    x0 : `numpy.array`
        The initial value of the unknown variable.
    f_grad : `callable`, optional
        The gradient of the objective.

        If `f_grad` is None, no gradient is used. Note that some optimization algorithms require a gradient!

        The default is None.
    tol : `float`, optional
        Tolerance for termination.

        `tol=None` is equivalent to `tol=0`.

        The default is None.
    max_iter : `int`, optional
        Maximum number of iterations.

        If `max_iter` is None, the default value of the particular optimization algorithm is used.

        Default is None.
    
    Returns
    -------
    `callable`
        An instance of :class:`ceml.optim.optimizer.Optimizer`
    
    Raises
    ------
    ValueError
         If `desc` contains an invalid description or if no gradient is specified but and `desc` describes a gradient based optimization algorithm.
    """
    if is_optimizer_grad_based(desc) and f_grad is None:
        raise ValueError("You have to specify the gradient of the cost function if you want to use a gradient-based optimization algorithm.")

    if desc == "nelder-mead":
        return NelderMead(f=f, x0=x0, tol=tol, max_iter=max_iter)
    elif desc == "powell":
        return Powell(f=f, x0=x0, tol=tol, max_iter=max_iter)
    elif desc == "bfgs":
        return BFGS(f=f, f_grad=f_grad, x0=x0, tol=tol, max_iter=max_iter)
    elif desc == "cg":
        return ConjugateGradients(f=f, f_grad=f_grad, x0=x0, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Invalid value of 'desc'.\n'desc' has to be 'nelder-mead', 'powell', 'cg' or 'bfgs' but not '{desc}'")


class NelderMead(Optimizer):
    """
    Nelder-Mead optimization algorithm.

    Parameters
    ----------
    f : `callable`
        The objective that is minimized.
    x0 : `numpy.array`
        The initial value of the unknown variable.
    tol : `float`, optional
        Tolerance for termination.

        `tol=None` is equivalent to `tol=0`.

        The default is None.
    max_iter : `int`, optional
        Maximum number of iterations.

        If `max_iter` is None, the default value of the particular optimization algorithm is used.

        Default is None.

    Note
    ----
    The Nelder-Mead algorithm is gradient-free optimization algorithm.
    """
    def __init__(self, f, x0, tol=None, max_iter=None):
        self.f = f
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter

        super(NelderMead, self).__init__()
    
    def __call__(self):
        optimum = minimize(fun=self.f, x0=self.x0, method="Nelder-Mead")
        return optimum["x"]


class Powell(Optimizer):
    """
    Powell optimization algorithm.

    Parameters
    ----------
    f : `callable`
        The objective that is minimized.
    x0 : `numpy.array`
        The initial value of the unknown variable.
    tol : `float`, optional
        Tolerance for termination.

        `tol=None` is equivalent to `tol=0`.

        The default is None.
    max_iter : `int`, optional
        Maximum number of iterations.

        If `max_iter` is None, the default value of the particular optimization algorithm is used.

        Default is None.

    Note
    ----
    The Powell algorithm is gradient-free optimization algorithm.
    """
    def __init__(self, f, x0, tol=None, max_iter=None):
        self.f = f
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter

        super(Powell, self).__init__()
    
    def __call__(self):
        optimum = minimize(fun=self.f, x0=self.x0, method="Nelder-Mead")
        return optimum["x"]


class ConjugateGradients(Optimizer):
    """
    Conjugate gradients optimization algorithm.

    Parameters
    ----------
    f : `callable`
        The objective that is minimized.
    f_grad : `callable`
        The gradient of the objective.
    x0 : `numpy.array`
        The initial value of the unknown variable.
    tol : `float`, optional
        Tolerance for termination.

        `tol=None` is equivalent to `tol=0`.

        The default is None.
    max_iter : `int`, optional
        Maximum number of iterations.

        If `max_iter` is None, the default value of the particular optimization algorithm is used.

        Default is None.
    """
    def __init__(self, f, f_grad, x0, tol=None, max_iter=None):
        self.f = f
        self.f_grad = f_grad
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter

        super(ConjugateGradients, self).__init__()
    
    def __call__(self):
        optimum = minimize(fun=self.f, x0=self.x0, jac=self.f_grad, method="CG")
        return np.array(optimum["x"])


class BFGS(Optimizer):
    """
    BFGS optimization algorithm.

    Parameters
    ----------
    f : `callable`
        The objective that is minimized.
    f_grad : `callable`
        The gradient of the objective.
    x0 : `numpy.array`
        The initial value of the unknown variable.
    tol : `float`, optional
        Tolerance for termination.

        `tol=None` is equivalent to `tol=0`.

        The default is None.
    max_iter : `int`, optional
        Maximum number of iterations.

        If `max_iter` is None, the default value of the particular optimization algorithm is used.

        Default is None.

    Note
    ----
    The BFGS optimization algorithm is a Quasi-Newton method.
    """
    def __init__(self, f, f_grad, x0, tol=None, max_iter=None):
        self.f = f
        self.f_grad = f_grad
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter

        super(BFGS, self).__init__()
    
    def __call__(self):
        optimum = minimize(fun=self.f, x0=self.x0, jac=self.f_grad, method="BFGS")
        return np.array(optimum["x"])
