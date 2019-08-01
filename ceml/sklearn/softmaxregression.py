# -*- coding: utf-8 -*-
import sklearn.linear_model

from ..backend.jax.layer import create_tensor, affine, softmax
from ..backend.jax.costfunctions import NegLogLikelihoodCost
from ..model import ModelWithLoss
from .counterfactual import SklearnCounterfactual


class SoftmaxRegression(ModelWithLoss):
    """Class for rebuilding/wrapping the :class:`sklearn.linear_model.LogisticRegression` class.

    The :class:`SoftmaxRegression` class rebuilds a softmax regression model from a given weight vector and intercept.

    Parameters
    ----------
    model : instance of :class:`sklearn.linear_model.LogisticRegression`
        The softmax regression model.

    Attributes
    ----------
    w : `jax.numpy.array`
        The weight vector (a matrix if we have more than two classes).
    b : `jax.numpy.array`
        The intercept/bias (a vector if we have more than two classes). 
        
    Raises
    ------
    TypeError
        If `model` is not an instance of :class:`sklearn.linear_model.LogisticRegression`
    """
    def __init__(self, model):
        if not isinstance(model, sklearn.linear_model.LogisticRegression):
            raise TypeError(f"model has to be an instance of 'sklearn.linear_model.LogisticRegression' not of {type(model)}")

        self.w = create_tensor(model.coef_)
        self.b = create_tensor(model.intercept_)

        super(SoftmaxRegression, self).__init__()

    def predict(self, x):
        """Predict the output of a given input.

        Computes the class probabilities for a given input `x`.

        Parameters
        ----------
        x : `numpy.ndarray`
            The input `x` that is going to be classified.
        
        Returns
        -------
        `jax.numpy.array`
            An array containing the class probabilities.
        """
        return softmax(affine(x, self.w, self.b))
    
    def get_loss(self, y_target, pred=None):
        """Creates and returns a loss function.

        Builds a negative-log-likehood cost function where the target is `y_target`.

        Parameters
        ----------
        y_target : `int`
            The target class.
        pred : `callable`, optional
            A callable that maps an input to the output (class probabilities).

            If `pred` is None, the class method `predict` is used for mapping the input to the output (class probabilities)

            The default is None.
        
        Returns
        -------
        :class:`ceml.backend.jax.costfunctions.NegLogLikelihoodCost`
            Initialized negative-log-likelihood cost function. Target label is `y_target`.
        """
        if pred is None:
            return NegLogLikelihoodCost(self.predict, y_target)
        else:
            return NegLogLikelihoodCost(pred, y_target)


class SoftmaxCounterfactual(SklearnCounterfactual):
    """Class for computing a counterfactual of a softmax regression model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model):
        super(SoftmaxCounterfactual, self).__init__(model)
    
    def rebuild_model(self, model):
        """Rebuilds a :class:`sklearn.linear_model.LogisticRegression` model.

        Converts a :class:`sklearn.linear_model.LogisticRegression` into a :class:`ceml.sklearn.softmaxregression.SoftmaxRegression`.

        Parameters
        ----------
        model : instance of :class:`sklearn.linear_model.LogisticRegression`
            The `sklearn` softmax regression model. 

        Returns
        -------
        :class:`ceml.sklearn.softmaxregression.SoftmaxRegression`
            The wrapped softmax regression model.
        """
        if not isinstance(model, sklearn.linear_model.LogisticRegression):
            raise TypeError(f"model has to be an instance of 'sklearn.linear_model.LogisticRegression' not of {type(model)}")
        if model.multi_class != "multinomial":
            raise ValueError(f"multi_class has to be 'multinomial' not {model.multi_class}")

        return SoftmaxRegression(model)


def softmaxregression_generate_counterfactual(model, x, y_target, features_whitelist=None, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : a :class:`sklearn.linear_model.LogisticRegression` instance.
        The softmax regression model that is used for computing the counterfactual.

        **Note:** `model.multi_class` must be set to `multinomial`.
    x : `numpy.ndarray`
        The input `x` whose prediction has to be explained.
    y_target : `int` or `float` or a callable that returns True if a given prediction is accepted.
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
        See :func:`ceml.optimizer.optimizer.desc_to_optim` for details.

        As an alternative, we can use any (custom) optimizer that is derived from the :class:`ceml.optim.optimizer.Optimizer` class.

        The default is "nelder-mead".
    return_as_dict : `boolean`, optional
        If True, returns the counterfactual, its prediction and the needed changes to the input as dictionary.
        If False, the results are returned as a triple.

        The default is True.
    done : `callable`, optional
        Not used.

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
    cf = SoftmaxCounterfactual(model)

    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, return_as_dict, done)
