# -*- coding: utf-8 -*-
import jax.numpy as npx
import sklearn.naive_bayes
from ..backend.jax.layer import log_normal_distribution, create_tensor
from ..backend.jax.costfunctions import NegLogLikelihoodCost
from ..model import ModelWithLoss
from .counterfactual import SklearnCounterfactual


class GaussianNB(ModelWithLoss):
    """Class for rebuilding/wrapping the :class:`sklearn.naive_bayes.GaussianNB` class

    The :class:`GaussianNB` class rebuilds a gaussian naive bayes model from a given set of parameters (priors, means and variances).

    Parameters
    ----------
    model : instance of :class:`sklearn.naive_bayes.GaussianNB`
        The gaussian naive bayes model.

    Attributes
    ----------
    class_priors : `numpy.array`
        Class dependend priors.
    means : `numpy.array`
        Class and feature dependend means.
    variances : `numpy.array`
        Class and feature dependend variances.
    """
    def __init__(self, model):
        if not isinstance(model, sklearn.naive_bayes.GaussianNB):
            raise TypeError(f"model has to be an instance of 'sklearn.naive_bayes.GaussianNB' but not of {type(model)}")

        self.class_priors = model.class_prior_
        self.means = model.theta_
        self.variances = model.sigma_
        
        super(GaussianNB, self).__init__()

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
        feature_wise_normal = lambda z, mu, v: npx.sum([log_normal_distribution(z[i], mu[i], v[i]) for i in range(z.shape[0])])

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


class GaussianNbCounterfactual(SklearnCounterfactual):
    """Class for computing a counterfactual of a gaussian naive bayes model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model):
        super(GaussianNbCounterfactual, self).__init__(model)
    
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


def gaussiannb_generate_counterfactual(model, x, y_target, features_whitelist=None, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : a :class:`sklearn.naive_bayes.GaussianNB` instance.
        The gaussian naive bayes model that is used for computing the counterfactual.
    x : `numpy.ndarray`
        The input `x` whose prediction has to be explained.
    y_target : `int` or `float` or a callable that returns True if a given prediction is accepted.
        The desired prediction of the counterfactual.
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
    cf = GaussianNbCounterfactual(model)

    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, return_as_dict, done)
