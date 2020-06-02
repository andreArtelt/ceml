# -*- coding: utf-8 -*-
import jax.numpy as npx
import numpy as np
import cvxpy as cp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..backend.jax.layer import log_multivariate_normal, create_tensor
from ..backend.jax.costfunctions import NegLogLikelihoodCost
from ..model import ModelWithLoss
from .counterfactual import SklearnCounterfactual
from ..optim import MathematicalProgram, ConvexQuadraticProgram, PlausibleCounterfactualOfHyperplaneClassifier


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
    def __init__(self, model, **kwds):
        if not isinstance(model, LinearDiscriminantAnalysis):
            raise TypeError(f"model has to be an instance of 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis' but not of {type(model)}")

        self.class_priors = model.priors_
        self.means = model.means_
        self.sigma_inv = np.linalg.inv(model.covariance_)

        self.dim = self.means.shape[1]
        
        super().__init__(**kwds)
    
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


class LdaCounterfactual(SklearnCounterfactual, MathematicalProgram, ConvexQuadraticProgram, PlausibleCounterfactualOfHyperplaneClassifier):
    """Class for computing a counterfactual of a lda model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model, **kwds):
        if not isinstance(model, LinearDiscriminantAnalysis):
            raise TypeError(f"model has to be an instance of 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis' but not of {type(model)}")
        if not hasattr(model, "covariance_"):
            raise AttributeError("You have to set store_covariance=True when instantiating a new sklearn.discriminant_analysis.LinearDiscriminantAnalysis model")

        super().__init__(model=model, w=None, b=None, n_dims=model.means_.shape[1], **kwds)
    
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
        return Lda(model)
    
    def _build_constraints(self, var_x, y):
        constraints = []

        # If set, a apply an affine preprocessing to x
        var_x_ = self._apply_affine_preprocessing(var_x)

        # Build constraints
        i = y
        q_i = np.dot(self.mymodel.sigma_inv, self.mymodel.means[i].T)
        b_i = np.log(self.mymodel.class_priors[i]) - .5 * np.dot( self.mymodel.means[i], np.dot(self.mymodel.sigma_inv, self.mymodel.means[i].T))

        for j in range(len(self.mymodel.means)):    # One vs. One
            if i == j:
                continue
            
            q_j = np.dot(self.mymodel.sigma_inv, self.mymodel.means[j])
            b_j = np.log(self.mymodel.class_priors[j]) - .5 * np.dot(self.mymodel.means[j], np.dot(self.mymodel.sigma_inv, self.mymodel.means[j]))

            constraints.append(q_i.T @ var_x_ + b_i >= q_j.T @ var_x_ + b_j + self.epsilon)

        return constraints

    def solve(self, x_orig, y_target, regularization, features_whitelist, return_as_dict):
        mad = None
        if regularization == "l1":
            mad = np.ones(x_orig.shape[0])

        xcf = self.build_solve_opt(x_orig, y_target, features_whitelist, mad=mad)
        delta = x_orig - xcf

        if self._model_predict([xcf]) != y_target:
            raise Exception("No counterfactual found - Consider changing parameters 'regularization', 'features_whitelist', 'optimizer' and try again")

        if return_as_dict is True:
            return self._SklearnCounterfactual__build_result_dict(xcf, y_target, delta)
        else:
            return xcf, y_target, delta


def lda_generate_counterfactual(model, x, y_target, features_whitelist=None, regularization="l1", C=1.0, optimizer="mp", return_as_dict=True, done=None, plausibility=None):
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

        Linear discriminant analysis supports the use of mathematical programs for computing counterfactuals - set `optimizer` to "mp" for using a convex quadratic program for computing the counterfactual. Note that in this case the hyperparameter `C` is ignored.

        As an alternative, we can use any (custom) optimizer that is derived from the :class:`ceml.optim.optimizer.Optimizer` class.

        The default is "mp".
    return_as_dict : `boolean`, optional
        If True, returns the counterfactual, its prediction and the needed changes to the input as dictionary.
        If False, the results are returned as a triple.

        The default is True.
    done : `callable`, optional
        Not used.
    plausibility: `dict`, optional.
        If set to a valid dictionary (see :func:`ceml.sklearn.plausibility.prepare_computation_of_plausible_counterfactuals`), a plausible counterfactual (as proposed in Artelt et al. 2020) is computed. Note that in this case, all other parameters are ignored.

        If `plausibility` is None, the closest counterfactual is computed.

        The default is None.

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

    if optimizer == "auto":
        optimizer = "mp"

    if plausibility is None:
        return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, return_as_dict, done)
    else:
        raise NotImplementedError()