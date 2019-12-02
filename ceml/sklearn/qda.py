# -*- coding: utf-8 -*-
import jax.numpy as npx
import numpy as np
import cvxpy as cp
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from ..backend.jax.layer import log_multivariate_normal, create_tensor
from ..backend.jax.costfunctions import NegLogLikelihoodCost
from ..model import ModelWithLoss
from .counterfactual import SklearnCounterfactual
from ..optim import MathematicalProgram, SDP, DCQP


class Qda(ModelWithLoss):
    """Class for rebuilding/wrapping the :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis` class.

    The :class:`Qda` class rebuilds a lda model from a given parameters.

    Parameters
    ----------
    model : instance of :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
        The qda model.

    Attributes
    ----------
    class_priors : `numpy.ndarray`
        Class dependend priors.
    means : `numpy.ndarray`
        Class dependend means.
    sigma_inv : `numpy.ndarray`
        Class dependend inverted covariance matrices.
    dim : `int`
        Dimensionality of the input data.
    is_binary : `boolean`
        True if `model` is a binary classifier, False otherwise.

    Raises
    ------
    TypeError
        If `model` is not an instance of :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
    """
    def __init__(self, model):
        if not isinstance(model, QuadraticDiscriminantAnalysis):
            raise TypeError(f"model has to be an instance of 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis' but not of {type(model)}")

        self.class_priors = model.priors_
        self.means = model.means_
        self.sigma_inv = [np.linalg.inv(cov) for cov in model.covariance_]

        self.dim = self.means.shape[1]
        self.is_binary = self.means.shape[0] == 2
        
        super(Qda, self).__init__()
    
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
        log_proba = create_tensor([npx.log(self.class_priors[i]) + log_multivariate_normal(x, self.means[i], self.sigma_inv[i], self.dim) for i in range(len(self.class_priors))])
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


class QdaCounterfactual(SklearnCounterfactual, MathematicalProgram, SDP, DCQP):
    """Class for computing a counterfactual of a qda model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model):
        super(QdaCounterfactual, self).__init__(model)
    
    def rebuild_model(self, model):
        """Rebuild a :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis` model.

        Converts a :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis` into a :class:`ceml.sklearn.qda.Qda`.

        Parameters
        ----------
        model : instance of :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
            The `sklearn` qda model - note that `store_covariance` must be set to True. 

        Returns
        -------
        :class:`ceml.sklearn.qda.Qda`
            The wrapped qda model.
        """
        if not isinstance(model, QuadraticDiscriminantAnalysis):
            raise TypeError(f"model has to be an instance of 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis' but not of {type(model)}")

        return Qda(model)
    
    def _build_constraints(self, var_X, var_x, y):
        i = y
        j = 0 if y == 1 else 1

        A = .5 * ( self.mymodel.sigma_inv[i] - self.mymodel.sigma_inv[j])
        b = np.dot(self.mymodel.sigma_inv[j], self.mymodel.means[j]) - np.dot(self.mymodel.sigma_inv[i], self.mymodel.means[i])
        c = np.log(self.mymodel.class_priors[j] / self.mymodel.class_priors[i]) + 0.5 * np.log(np.linalg.det(self.mymodel.sigma_inv[j]) / np.linalg.det(self.mymodel.sigma_inv[i])) + 0.5 * (self.mymodel.means[i].T.dot(self.mymodel.sigma_inv[i]).dot(self.mymodel.means[i]) - self.mymodel.means[j].T.dot(self.mymodel.sigma_inv[j]).dot(self.mymodel.means[j]))

        return [cp.trace(A @ var_X) + var_x.T @ b + c + self.epsilon <= 0]

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
            A0_i.append(.5 * self.mymodel.sigma_inv[i])
            A1_i.append(.5 * self.mymodel.sigma_inv[j])
            b_i.append(np.dot(self.mymodel.sigma_inv[j], self.mymodel.means[j]) - np.dot(self.mymodel.sigma_inv[i], self.mymodel.means[i]))
            r_i.append(np.log(self.mymodel.class_priors[j] / self.mymodel.class_priors[i]) + .5 * np.log(np.linalg.det(self.mymodel.sigma_inv[j]) / np.linalg.det(self.mymodel.sigma_inv[i])) + .5 * (self.mymodel.means[i].T.dot(self.mymodel.sigma_inv[i]).dot(self.mymodel.means[i]) - self.mymodel.means[j].T.dot(self.mymodel.sigma_inv[j]).dot(self.mymodel.means[j])))

        self.build_program(self.model, x_orig, y_target, Q0, Q1, q, c, A0_i, A1_i, b_i, r_i, features_whitelist=features_whitelist, mad=None if regularization != "l1" else np.ones(self.mymodel.dim))
        
        return DCQP.solve(self, x0=self.mymodel.means[i], tao=1.2, tao_max=100, mu=1.5)

    def solve(self, x_orig, y_target, regularization, features_whitelist, return_as_dict):
        xcf = None
        if self.mymodel.is_binary:
            xcf = self.build_solve_opt(x_orig, y_target, features_whitelist)
        else:
            xcf = self._build_solve_dcqp(x_orig, y_target, regularization, features_whitelist)
        delta = x_orig - xcf

        if self.model.predict([xcf]) != y_target:
            raise Exception("No counterfactual found - Consider changing parameters 'regularization', 'features_whitelist', 'optimizer' and try again")

        return self.__build_result_dict(xcf, y_target, delta) if return_as_dict else xcf, y_target, delta


def qda_generate_counterfactual(model, x, y_target, features_whitelist=None, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : a :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis` instance.
        The qda model that is used for computing the counterfactual.
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

        Quadratic discriminant analysis supports the use of mathematical programs for computing counterfactuals - set `optimizer` to "mp" for using a semi-definite program (binary classifier) or a DCQP (otherwise) for computing the counterfactual.
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
    cf = QdaCounterfactual(model)

    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, return_as_dict, done)