# -*- coding: utf-8 -*-
import jax.numpy as npx
import numpy as np
import cvxpy as cp
import sklearn.naive_bayes

from ..backend.jax.layer import log_normal_distribution, create_tensor
from ..backend.jax.costfunctions import NegLogLikelihoodCost
from ..model import ModelWithLoss
from .counterfactual import SklearnCounterfactual
from ..optim import MathematicalProgram, SDP, DCQP


class GaussianNB(ModelWithLoss):
    """Class for rebuilding/wrapping the :class:`sklearn.naive_bayes.GaussianNB` class

    The :class:`GaussianNB` class rebuilds a gaussian naive bayes model from a given set of parameters (priors, means and variances).

    Parameters
    ----------
    model : instance of :class:`sklearn.naive_bayes.GaussianNB`
        The gaussian naive bayes model.

    Attributes
    ----------
    class_priors : `numpy.ndarray`
        Class dependend priors.
    means : `numpy.array`
        Class and feature dependend means.
    variances : `numpy.ndarray`
        Class and feature dependend variances.
    dim  : `int`
        Dimensionality of the input data.
    is_binary : `boolean`
        True if `model` is a binary classifier, False otherwise.
    """
    def __init__(self, model, **kwds):
        if not isinstance(model, sklearn.naive_bayes.GaussianNB):
            raise TypeError(f"model has to be an instance of 'sklearn.naive_bayes.GaussianNB' but not of {type(model)}")

        self.class_priors = model.class_prior_
        self.means = model.theta_
        self.variances = model.sigma_

        self.dim = self.means.shape[1]
        self.is_binary = self.means.shape[0] == 2
        
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
        feature_wise_normal = lambda z, mu, v: npx.sum(npx.array([log_normal_distribution(z[i], mu[i], v[i]) for i in range(z.shape[0])]))

        log_proba = create_tensor([npx.log(self.class_priors[i]) + feature_wise_normal(x, self.means[i], self.variances[i]) for i in range(len(self.class_priors))])
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


class GaussianNbCounterfactual(SklearnCounterfactual, MathematicalProgram, SDP, DCQP):
    """Class for computing a counterfactual of a gaussian naive bayes model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model, **kwds):
        super().__init__(model=model, **kwds)
    
    def rebuild_model(self, model):
        """Rebuild a :class:`sklearn.naive_bayes.GaussianNB` model.

        Converts a :class:`sklearn.naive_bayes.GaussianNB` into a :class:`ceml.sklearn.naivebayes.GaussianNB`.

        Parameters
        ----------
        model : instance of :class:`sklearn.naive_bayes.GaussianNB`
            The `sklearn` gaussian naive bayes model. 

        Returns
        -------
        :class:`ceml.sklearn.naivebayes.GaussianNB`
            The wrapped gaussian naive bayes model.
        """
        if not isinstance(model, sklearn.naive_bayes.GaussianNB):
            raise TypeError(f"model has to be an instance of 'sklearn.naive_bayes.GaussianNB' but not of {type(model)}")

        return GaussianNB(model)

    def _build_constraints(self, var_X, var_x, y):
        i = y
        j = 0 if y == 1 else 1

        A = np.diag(-1. / (2. * self.mymodel.variances[j, :])) + np.diag(1. / (2. * self.mymodel.variances[i, :]))
        b = (self.mymodel.means[j, :] / self.mymodel.variances[j, :]) - (self.mymodel.means[i, :] / self.mymodel.variances[i, :])
        c = np.log(self.mymodel.class_priors[j] / self.mymodel.class_priors[i]) + np.sum([np.log(1. / np.sqrt(2.*np.pi*self.mymodel.variances[j,k])) - ((self.mymodel.means[j,k]**2) / (2.*self.mymodel.variances[j,k])) for k in range(self.mymodel.dim)]) - np.sum([np.log(1. / np.sqrt(2.*np.pi*self.mymodel.variances[i,k])) - ((self.mymodel.means[i,k]**2) / (2.*self.mymodel.variances[i,k])) for k in range(self.mymodel.dim)])

        return [cp.trace(A @ var_X) + b @ var_x + c + self.epsilon <= 0]

    def _build_solve_dcqp(self, x_orig, y_target, regularization, features_whitelist):
        Q0 = np.eye(self.mymodel.dim)   # TODO: Can be ignored if regularization != l2
        Q1 = np.zeros((self.mymodel.dim, self.mymodel.dim))
        q = -2. * x_orig
        c = 0.0

        A0_i = []
        A1_i = []
        b_i = []
        r_i = []
        
        i = y_target
        for j in filter(lambda z: z != y_target, range(len(self.mymodel.means))):
            A0_i.append(np.diag(1. / (2. * self.mymodel.variances[i, :])))
            A1_i.append(np.diag(1. / (2. * self.mymodel.variances[j, :])))
            b_i.append((self.mymodel.means[j, :] / self.mymodel.variances[j, :]) - (self.mymodel.means[i, :] / self.mymodel.variances[i, :]))
            r_i.append(np.log(self.mymodel.class_priors[j] / self.mymodel.class_priors[i]) + np.sum([np.log(1. / np.sqrt(2.*np.pi*self.mymodel.variances[j,k])) - ((self.mymodel.means[j,k]**2) / (2.*self.mymodel.variances[j,k])) for k in range(self.mymodel.dim)]) - np.sum([np.log(1. / np.sqrt(2.*np.pi*self.mymodel.variances[i,k])) - ((self.mymodel.means[i,k]**2) / (2.*self.mymodel.variances[i,k])) for k in range(self.mymodel.dim)]))

        self.build_program(self.model, x_orig, y_target, Q0, Q1, q, c, A0_i, A1_i, b_i, r_i, features_whitelist=features_whitelist, mad=None if regularization != "l1" else np.ones(self.mymodel.dim))
        
        return DCQP.solve(self, x0=self.mymodel.means[i, :], tao=1.2, tao_max=100, mu=1.5)

    def solve(self, x_orig, y_target, regularization, features_whitelist, return_as_dict):
        xcf = None
        if self.mymodel.is_binary:
            xcf = self.build_solve_opt(x_orig, y_target)
        else:
            xcf = self._build_solve_dcqp(x_orig, y_target, regularization, features_whitelist)
        delta = x_orig - xcf

        if self._model_predict([xcf]) != y_target:
            raise Exception("No counterfactual found - Consider changing parameters 'regularization', 'features_whitelist', 'optimizer' and try again")

        if return_as_dict is True:
            return self._SklearnCounterfactual__build_result_dict(xcf, y_target, delta)
        else:
            return xcf, y_target, delta


def gaussiannb_generate_counterfactual(model, x, y_target, features_whitelist=None, regularization="l1", C=1.0, optimizer="auto", return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : a :class:`sklearn.naive_bayes.GaussianNB` instance.
        The gaussian naive bayes model that is used for computing the counterfactual.
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

        Use "auto" if you do not know what optimizer to use - a suitable optimizer is chosen automatically.

        The default is "auto".

        Gaussian naive Bayes supports the use of mathematical programs for computing counterfactuals - set `optimizer` to "mp" for using a semi-definite program (binary classifier) or a DCQP (otherwise) for computing the counterfactual.
        Note that in this case the hyperparameter `C` is ignored.
        Because the DCQP is a non-convex problem, we are not guaranteed to find the best solution (it might even happen that we do not find a solution at all) - we use the penalty convex-concave procedure for approximately solving the DCQP.
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
    cf = GaussianNbCounterfactual(model)

    if optimizer == "auto":
        if cf.mymodel.is_binary:
            optimizer = "mp"
        else:
            optimizer = "nelder-mead"

    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, return_as_dict, done)
