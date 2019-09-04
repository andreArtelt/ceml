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
    Any class derived from :class:`Optimizer` has to implement the abstract methods `init`, `__call__` and `is_grad_based`.
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def init(self, ):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self):
        raise NotImplementedError()
    
    @abstractmethod
    def is_grad_based(self):
        raise NotImplementedError()


def is_optimizer_grad_based(optim):
    """
    Determines whether a specific optimization algorithm (specified by a description in `desc`) needs a gradient.

    Supported descriptions:

        - nelder-mead: *Gradient-free* Nelder-Mead optimizer (also called Downhill-Simplex)
        - powell: Gradient-free Powell optimizer
        - bfgs: BFGS optimizer
        - cg: Conjugate gradients optimizer

    Parameters
    ----------
    optim : `str` or instance of :class:`ceml.optim.optimizer.Optimizer`
        Description of the optimization algorithm or an instance of :class:`ceml.optim.optimizer.Optimizer`.
    
    Returns
    -------
    `bool`
        True if the optimization algorithm needs a gradient, False otherwise.
    
    Raises
    ------
    ValueError
        If `optim` contains an invalid description.
    TypeError
        If `optim` is neither a string nor an instance of :class:`ceml.optim.optimizer.Optimizer`.
    """
    if isinstance(optim, str):
        if optim == "nelder-mead":
            return False
        elif optim == "powell":
            return False
        elif optim == "bfgs":
            return True
        elif optim == "cg":
            return True
        else:
            raise ValueError(f"Invalid value of 'optim'.\n'optim' has to be 'nelder-mead', 'powell', 'cg' or 'bfgs' but not '{optim}'")
    elif isinstance(optim, Optimizer):
        return optim.is_grad_based()
    else:
        raise TypeError(f"optim has to be either a string or an instance of 'ceml.optim.optimizer.Optimizer' but not of {type(optim)}")


def prepare_optim(optim, f, x0, f_grad=None, tol=None, max_iter=None):
    """
    Creates and initializes an optimization algorithm (instance of :class:`ceml.optim.optimizer.Optimizer`) specified by a description of the algorithm.

    Supported descriptions:

        - nelder-mead: *Gradient-free* Nelder-Mead optimizer (also called downhill simplex method)
        - powell: *Gradient-free* Powell optimizer
        - bfgs: BFGS optimizer
        - cg: Conjugate gradients optimizer

    Parameters
    ----------
    optim : `str` or instance of :class:`ceml.optim.optimizer.Optimizer`
        Description of the optimization algorithm or an instance of :class:`ceml.optim.optimizer.Optimizer`.
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
         If `optim` contains an invalid description or if no gradient is specified but and `optim` describes a gradient based optimization algorithm.
    TypeError
        If `optim` is neither a string nor an instance of :class:`ceml.optim.optimizer.Optimizer`.
    """
    if is_optimizer_grad_based(optim) and f_grad is None:
        raise ValueError("You have to specify the gradient of the cost function if you want to use a gradient-based optimization algorithm.")

    if isinstance(optim, str):
        if optim == "nelder-mead":
            optim = NelderMead()
            optim.init(f=f, x0=x0, tol=tol, max_iter=max_iter)
            return optim
        elif optim == "powell":
            optim = Powell()
            optim.init(f=f, x0=x0, tol=tol, max_iter=max_iter)
            return optim
        elif optim == "bfgs":
            optim = BFGS()
            optim.init(f=f, f_grad=f_grad, x0=x0, tol=tol, max_iter=max_iter)
            return optim
        elif optim == "cg":
            optim = ConjugateGradients()
            optim.init(f=f, f_grad=f_grad, x0=x0, tol=tol, max_iter=max_iter)
            return optim
        else:
            raise ValueError(f"Invalid value of 'optim'.\n'optim' has to be 'nelder-mead', 'powell', 'cg' or 'bfgs' but not '{optim}'")
    elif isinstance(optim, Optimizer):
        args = {'f': f, 'x0': x0, 'tol': tol, 'max_iter': max_iter}
        if is_optimizer_grad_based(optim):
            args['f_grad'] = f_grad

        optim.init(**args)
        return optim
    else:
        raise TypeError(f"optim has to be either a string or an instance of 'ceml.optim.optimizer.Optimizer' but not of {type(optim)}")


class NelderMead(Optimizer):
    """
    Nelder-Mead optimization algorithm.

    Note
    ----
    The Nelder-Mead algorithm is a gradient-free optimization algorithm.
    """
    def __init__(self):
        self.f = None
        self.x0 = None
        self.tol = None
        self.max_iter = None

        super(NelderMead, self).__init__()
    
    def init(self, f, x0, tol=None, max_iter=None):
        """
        Initializes all parameters.

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
        """
        self.f = f
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter

    def is_grad_based(self):
        return False

    def __call__(self):
        optimum = minimize(fun=self.f, x0=self.x0, tol=self.tol, options={'maxiter': self.max_iter}, method="Nelder-Mead")
        return optimum["x"]


class Powell(Optimizer):
    """
    Powell optimization algorithm.

    Note
    ----
    The Powell algorithm is a gradient-free optimization algorithm.
    """
    def __init__(self):
        self.f = None
        self.x0 = None
        self.tol = None
        self.max_iter = None

        super(Powell, self).__init__()
    
    def init(self, f, x0, tol=None, max_iter=None):
        """
        Initializes all parameters.

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
        """
        self.f = f
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter

    def is_grad_based(self):
        return False

    def __call__(self):
        optimum = minimize(fun=self.f, x0=self.x0, tol=self.tol, options={'maxiter': self.max_iter}, method="Nelder-Mead")
        return optimum["x"]


class ConjugateGradients(Optimizer):
    """
    Conjugate gradients optimization algorithm.
    """
    def __init__(self):
        self.f = None
        self.f_grad = None
        self.x0 = None
        self.tol = None
        self.max_iter = None

        super(ConjugateGradients, self).__init__()
    
    def init(self, f, f_grad, x0, tol=None, max_iter=None):
        """
        Initializes all parameters.

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
        self.f = f
        self.f_grad = f_grad
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter

    def is_grad_based(self):
        return False

    def __call__(self):
        optimum = minimize(fun=self.f, x0=self.x0, jac=self.f_grad, tol=self.tol, options={'maxiter': self.max_iter}, method="CG")
        return np.array(optimum["x"])


class BFGS(Optimizer):
    """
    BFGS optimization algorithm.

    Note
    ----
    The BFGS optimization algorithm is a Quasi-Newton method.
    """
    def __init__(self):
        self.f = None
        self.f_grad = None
        self.x0 = None
        self.tol = None
        self.max_iter = None

        super(BFGS, self).__init__()
    
    def init(self, f, f_grad, x0, tol=None, max_iter=None):
        """
        Initializes all parameters.

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
        self.f = f
        self.f_grad = f_grad
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter

    def is_grad_based(self):
        return False

    def __call__(self):
        optimum = minimize(fun=self.f, x0=self.x0, jac=self.f_grad, tol=self.tol, options={'maxiter': self.max_iter}, method="BFGS")
        return np.array(optimum["x"])
