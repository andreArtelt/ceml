# -*- coding: utf-8 -*-
import jax.numpy as npx
import numpy as np
import cvxpy as cp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..backend.jax.layer import log_multivariate_normal, create_tensor
from ..backend.jax.costfunctions import NegLogLikelihoodCost
from ..model import ModelWithLoss
from .counterfactual import SklearnCounterfactual
from ..optim import MathematicalProgram, ConvexQuadraticProgram


class Lda(ModelWithLoss):
    """Class for rebuilding/wrapping the :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis` class.

    The :class:`Lda` class rebuilds a lda model from a given parameters.

    Parameters
    ----------
    model : instance of :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
        The lda model.

    Attributes
    ----------
    class_priors : `numpy.ndarray`
        Class dependend priors.
    means : `numpy.ndarray`
        Class dependend means.
    sigma_inv : `numpy.ndarray`
        Inverted covariance matrix.
    dim : `int`
        Dimensionality of the input data.

    Raises
    ------
    TypeError
        If `model` is not an instance of :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
    """
    def __init__(self, model):
        if not isinstance(model, LinearDiscriminantAnalysis):
            raise TypeError(f"model has to be an instance of 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis' but not of {type(model)}")

        self.class_priors = model.priors_
        self.means = model.means_
        self.sigma_inv = np.linalg.inv(model.covariance_)

        self.dim = self.means.shape[1]
        
        super(Lda, self).__init__()
    
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
        log_proba = create_tensor([npx.log(self.class_priors[i]) + log_multivariate_normal(x, self.means[i], self.sigma_inv, self.dim) for i in range(len(self.class_priors))])
        proba = npx.exp(log_proba)

        return proba / npx.sum(proba)

    def get_loss(self, y_target, pred=None):
        """Creates and returns a loss function.

        Build a negative-log-likehood cost function where the target is `y_target`.

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


class LdaCounterfactual(SklearnCounterfactual, MathematicalProgram, ConvexQuadraticProgram):
    """Class for computing a counterfactual of a lda model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model):
        super(LdaCounterfactual, self).__init__(model)
    
    def rebuild_model(self, model):
        """Rebuild a :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis` model.

        Converts a :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis` into a :class:`ceml.sklearn.lda.Lda`.

        Parameters
        ----------
        model : instance of :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
            The `sklearn` lda model - note that `store_covariance` must be set to True. 

        Returns
        -------
        :class:`ceml.sklearn.lda.Lda`
            The wrapped qda model.
        """
        if not isinstance(model, LinearDiscriminantAnalysis):
            raise TypeError(f"model has to be an instance of 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis' but not of {type(model)}")

        return Lda(model)
    
    def _build_constraints(self, var_x, y):
        constraints = []

        i = y
        q_i = np.dot(self.mymodel.sigma_inv, self.mymodel.means[i])
        b_i = np.log(self.mymodel.class_priors[i]) - .5 * np.dot( self.mymodel.means[i], np.dot(self.mymodel.sigma_inv, self.mymodel.means[i]))

        for j in range(len(self.mymodel.means)):
            if i == j:
                continue
            
            q_j = np.dot(self.mymodel.sigma_inv, self.mymodel.means[j])
            b_j = np.log(self.mymodel.class_priors[j]) - .5 * np.dot(self.mymodel.means[j], np.dot(self.mymodel.sigma_inv, self.mymodel.means[j]))

            constraints.append(q_i.T @ var_x + b_i >= q_j.T @ var_x + b_j + self.epsilon)

        return constraints

    def solve(self, x_orig, y_target, regularization, features_whitelist, return_as_dict):
        mad = None
        if regularization == "l1":
            mad = np.ones(self.mymodel.dim)

        xcf = self.build_solve_opt(x_orig, y_target, features_whitelist, mad=mad)
        delta = x_orig - xcf

        if self.model.predict([xcf]) != y_target:
            raise Exception("No counterfactual found - Consider changing parameters 'regularization', 'features_whitelist', 'optimizer' and try again")

        return self.__build_result_dict(xcf, y_target, delta) if return_as_dict else xcf, y_target, delta


def lda_generate_counterfactual(model, x, y_target, features_whitelist=None, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : a :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis` instance.
        The lda model that is used for computing the counterfactual.
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
        See :func:`ceml.optim.optimizer.prepare_optim` for details.

        As an alternative, we can use any (custom) optimizer that is derived from the :class:`ceml.optim.optimizer.Optimizer` class.

        The default is "nelder-mead".

        Linear discriminant analysis supports the use of mathematical programs for computing counterfactuals - set `optimizer` to "mp" for using a convex quadratic program for computing the counterfactual. Note that in this case the hyperparameter `C` is ignored.
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
    cf = LdaCounterfactual(model)

    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, return_as_dict, done)