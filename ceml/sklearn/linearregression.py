# -*- coding: utf-8 -*-
import sklearn.linear_model
import numpy as np

from ..backend.jax.layer import affine, create_tensor
from ..backend.jax.costfunctions import RegularizedCost, SquaredError
from ..model import ModelWithLoss
from .counterfactual import SklearnCounterfactual
from ..optim import MathematicalProgram, ConvexQuadraticProgram


class LinearRegression(ModelWithLoss):
    """Class for rebuilding/wrapping the :class:`sklearn.linear_model.base.LinearModel` class

    The :class:`LinearRegression` class rebuilds a softmax regression model from a given weight vector and intercept.

    Parameters
    ----------
    model : instance of :class:`sklearn.linear_model.base.LinearModel`
        The linear regression model (e.g. :class:`sklearn.linear_model.LinearRegression` or :class:`sklearn.linear_model.Ridge`).

    Attributes
    ----------
    w : `numpy.ndarray`
        The weight vector (a matrix if we have a multi-dimensional output).
    b : `numpy.ndarray`
        The intercept/bias (a vector if we have a multi-dimensional output).
    dim : `int`
        Dimensionality of the input data.
    """
    def __init__(self, model):
        if not isinstance(model, sklearn.linear_model.base.LinearModel):
            raise TypeError(f"model has to be an instance of a linear regression model like 'sklearn.linear_model.LinearRegression', 'sklearn.linear_model.Ridge', 'sklearn.linear_model.Lasso', 'sklearn.linear_model.HuberRegressor' or 'sklearn.linear_model.ElasticNet' but not of {type(model)}")

        self.w = model.coef_
        self.b = model.intercept_
        self.dim = model.coef_.shape[0]

        super(LinearRegression, self).__init__()

    def predict(self, x):
        """Predict the output of a given input.

        Computes the regression on a given input `x`.

        Parameters
        ----------
        x : `numpy.ndarray`
            The input `x` whose output is going to be predicted.
        
        Returns
        -------
        `jax.numpy.array`
            An array containing the predicted output.
        """
        return affine(x, self.w, self.b)
    
    def get_loss(self, y_target, pred=None):
        """Creates and returns a loss function.

        Build a squared-error cost function where the target is `y_target`.

        Parameters
        ----------
        y_target : `float`
            The target value.
        pred : `callable`, optional
            A callable that maps an input to the output (regression).

            If `pred` is None, the class method `predict` is used for mapping the input to the output (regression)

            The default is None.
        
        Returns
        -------
        :class:`ceml.backend.jax.costfunctions.SquaredError`
            Initialized squared-error cost function. Target is `y_target`.
        """
        if pred is None:
            return SquaredError(self.predict, y_target)
        else:
            return SquaredError(pred, y_target)


class LinearRegressionCounterfactual(SklearnCounterfactual, MathematicalProgram, ConvexQuadraticProgram):
    """Class for computing a counterfactual of a linear regression model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model):
        super(LinearRegressionCounterfactual, self).__init__(model)
    
    def rebuild_model(self, model):
        """Rebuild a :class:`sklearn.linear_model.base.LinearModel` model.

        Converts a :class:`sklearn.linear_model.base.LinearModel` into a :class:`ceml.sklearn.linearregression.LinearRegression`.

        Parameters
        ----------
        model : instance of :class:`sklearn.linear_model.base.LinearModel`
            The `sklearn` linear regression model (e.g. :class:`sklearn.linear_model.LinearRegression` or :class:`sklearn.linear_model.Ridge`). 

        Returns
        -------
        :class:`ceml.sklearn.linearregression.LinearRegression`
            The wrapped linear regression model.
        """
        if not isinstance(model, sklearn.linear_model.base.LinearModel):
            raise TypeError(f"model has to be an instance of a linear regression model like 'sklearn.linear_model.LinearRegression', 'sklearn.linear_model.Ridge', 'sklearn.linear_model.Lasso', 'sklearn.linear_model.HuberRegressor' or 'sklearn.linear_model.ElasticNet' but not of {type(model)}")
    
        return LinearRegression(model)
    
    def _build_constraints(self, var_x, y):
        constraints = []

        constraints.append(self.mymodel.w @ var_x + self.mymodel.b - y <= self.epsilon)
        constraints.append(-self.mymodel.w @ var_x - self.mymodel.b + y <= self.epsilon)
        
        return constraints

    def solve(self, x_orig, y_target, regularization, features_whitelist, return_as_dict):
        mad = None
        if regularization == "l1":
            mad = np.ones(self.mymodel.dim)

        xcf = self.build_solve_opt(x_orig, y_target, features_whitelist, mad=mad)
        delta = x_orig - xcf
        y_cf = self.model.predict([xcf])

        return self.__build_result_dict(xcf, y_target, delta) if return_as_dict else xcf, y_cf, delta


def linearregression_generate_counterfactual(model, x, y_target, features_whitelist=None, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : a :class:`sklearn.linear_model.base.LinearModel` instance.
        The linear regression model (e.g. :class:`sklearn.linear_model.LinearRegression` or :class:`sklearn.linear_model.Ridge`) that is used for computing the counterfactual.
    x : `numpy.ndarray`
        The input `x` whose prediction has to be explained.
    y_target : `float`
        The requested prediction of the counterfactual.
    features_whitelist : `list(int)`, optional
        List of feature indices (dimensions of the input space) that can be used when computing the counterfactual.
        
        If `features_whitelist` is None, all features can be used.

        The default is None.
    regularization : `str` or :class:`ceml.costfunctions.costfunctions.CostFunction`, optional
        Regularizer of the counterfactual. Penalty for deviating from the original input `x`.
        
        Supported values:
        
            - l1: Penalizes the absolute deviation.
            - l2: Penalizes the squared deviation.

        `regularization` can be a description of the regularization, an instance of :class:`ceml.costfunctions.costfunctions.CostFunction` (or :class:`ceml.costfunctions.costfunctions.CostFunctionDifferentiable` if your cost function is differentiable) or None if no regularization is requested.

        If `regularization` is None, no regularization is used.

        The default is "l1".
    C : `float` or `list(float)`, optional
        The regularization strength. If `C` is a list, all values in `C` are tried and as soon as a counterfactual is found, this counterfactual is returned and no other values of `C` are tried.

        `C` is ignored if no regularization is used (`regularization=None`).

        The default is 1.0
    optimizer : `str` or instance of :class:`ceml.optim.optimizer.Optimizer`, optional
        Name/Identifier of the optimizer that is used for computing the counterfactual.
        See :func:`ceml.optim.optimizer.prepare_optim` for details.

        As an alternative, we can use any (custom) optimizer that is derived from the :class:`ceml.optim.optimizer.Optimizer` class.

        The default is "nelder-mead".

        Linear regression supports the use of mathematical programs for computing counterfactuals - set `optimizer` to "mp" for using a convex quadratic program for computing the counterfactual. Note that in this case the hyperparameter `C` is ignored.
    return_as_dict : `boolean`, optional
        If True, returns the counterfactual, its prediction and the needed changes to the input as dictionary.
        If False, the results are returned as a triple.

        The default is True.
    done : `callable`, optional
        A callable that returns `True` if a counterfactual with a given output/prediction is accepted and `False` otherwise.

        If `done` is None, the output/prediction of the counterfactual must match `y_target` exactly.

        The default is None.

        Note
        ----
        It might not always be possible to achieve a given output/prediction exactly.

    Returns
    -------
    `dict` or `triple`
        A dictionary where the counterfactual is stored in 'x_cf', its prediction in 'y_cf' and the changes to the original input in 'delta'.

        (x_cf, y_cf, delta) : triple if `return_as_dict` is False
    
    Raises
    ------
    Exception
        If no counterfactual was found.
    """
    cf = LinearRegressionCounterfactual(model)

    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, return_as_dict, done)
