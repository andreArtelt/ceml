# -*- coding: utf-8 -*-
import sklearn.linear_model
import numpy as np

from ..backend.jax.layer import create_tensor, affine, softmax, softmax_binary
from ..backend.jax.costfunctions import NegLogLikelihoodCost
from ..model import ModelWithLoss
from .counterfactual import SklearnCounterfactual
from ..optim import MathematicalProgram, ConvexQuadraticProgram, PlausibleCounterfactualOfHyperplaneClassifier


class SoftmaxRegression(ModelWithLoss):
    """Class for rebuilding/wrapping the :class:`sklearn.linear_model.LogisticRegression` class.

    The :class:`SoftmaxRegression` class rebuilds a softmax regression model from a given weight vector and intercept.

    Parameters
    ----------
    model : instance of :class:`sklearn.linear_model.LogisticRegression`
        The softmax regression model.

    Attributes
    ----------
    w : `numpy.ndarray`
        The weight vector (a matrix if we have more than two classes).
    b : `numpy.ndarray`
        The intercept/bias (a vector if we have more than two classes). 
    dim : `int`
        Dimensionality of the input data.
    is_multiclass : `boolean`
        True if `model` is a binary classifier, False otherwise.

    Raises
    ------
    TypeError
        If `model` is not an instance of :class:`sklearn.linear_model.LogisticRegression`
    """
    def __init__(self, model, **kwds):
        if not isinstance(model, sklearn.linear_model.LogisticRegression):
            raise TypeError(f"model has to be an instance of 'sklearn.linear_model.LogisticRegression' not of {type(model)}")

        self.w = model.coef_
        self.b = model.intercept_
        self.dim = model.coef_.shape[1]

        self.is_multiclass = model.coef_.shape[0] > 1

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
        if self.is_multiclass is True:
            return softmax(affine(x, self.w, self.b))
        else:
            return softmax_binary(affine(x, self.w, self.b))
    
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


class SoftmaxCounterfactual(SklearnCounterfactual, MathematicalProgram, ConvexQuadraticProgram, PlausibleCounterfactualOfHyperplaneClassifier):
    """Class for computing a counterfactual of a softmax regression model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model, **kwds):
        if not isinstance(model, sklearn.linear_model.LogisticRegression):
            raise TypeError(f"model has to be an instance of 'sklearn.linear_model.LogisticRegression' not of {type(model)}")
        if model.multi_class != "multinomial":
            raise ValueError(f"multi_class has to be 'multinomial' not {model.multi_class}")

        super().__init__(model=model, w=model.coef_, b=model.intercept_, n_dims=model.coef_.shape[1], **kwds)
    
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
        return SoftmaxRegression(model)
    
    def _build_constraints(self, var_x, y):
        constraints = []
        
        # If set, a apply an affine preprocessing to x
        var_x_ = self._apply_affine_preprocessing(var_x)

        # Build constraints
        if self.mymodel.is_multiclass is True:  # Multiclass classifier
            i = y
            w_i = self.mymodel.w[i,:]
            b_i = self.mymodel.b[i]

            for j in range(len(self.mymodel.w)):
                if i == j:
                    continue

                w_j = self.mymodel.w[j,:]
                b_j = self.mymodel.b[j]
                
                constraints.append(w_i.T @ var_x_ + b_i >= w_j.T @ var_x_ + b_j + self.epsilon)
        else:   # Binary classifier
            y_ = -1 if y == 0 else 1
            constraints.append(y_ * (var_x_.T @ self.mymodel.w.flatten() + self.mymodel.b.flatten()) >= self.epsilon)

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


def softmaxregression_generate_counterfactual(model, x, y_target, features_whitelist=None, regularization="l1", C=1.0, optimizer="mp", return_as_dict=True, done=None, plausibility=None):
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
        See :func:`ceml.optim.optimizer.prepare_optim` for details.

        Softmax regression supports the use of mathematical programs for computing counterfactuals - set `optimizer` to "mp" for using a convex quadratic program for computing the counterfactual. Note that in this case the hyperparameter `C` is ignored.

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
    cf = SoftmaxCounterfactual(model)

    if optimizer == "auto":
        optimizer = "mp"

    if plausibility is None:
        return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, return_as_dict, done)
    else:
        cf.setup_plausibility_params(plausibility["ellipsoids_r"], plausibility["gmm_weights"], plausibility["gmm_means"], plausibility["gmm_covariances"], plausibility["projection_matrix"], plausibility["projection_mean_sub"], plausibility["use_density_constraints"], plausibility["density_thresholds"])

        return cf.compute_plausible_counterfactual(x, y_target)