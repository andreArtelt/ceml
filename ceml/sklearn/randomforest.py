# -*- coding: utf-8 -*-
import logging
import sklearn.ensemble

from .decisiontree import DecisionTreeCounterfactual
from ..model import ModelWithLoss
from ..costfunctions import CostFunction, RegularizedCost
from .utils import build_regularization_loss
from ..optim import InputWrapper
from .counterfactual import SklearnCounterfactual


class EnsembleVotingCost(CostFunction):
    """Loss function of an ensemble of models.

    The loss is the negative fraction of models that predict the correct output.

    Parameters
    ----------
    models : `list(object)`
        List of models
    y_target : `int`, `float` or a callable that returns True if a given prediction is accepted.
        The requested prediction.
    input_wrapper : `callable`, optional
        Converts the input (e.g. if we want to exclude some features/dimensions, we might have to include these missing features before applying any function to it).
        
        The default is None.
    """
    def __init__(self, models, y_target, input_wrapper=None, epsilon=0):
        self.models = models
        self.num_models = len(models)
        self.y_target = y_target if callable(y_target) else lambda y: y == y_target
        
        super(EnsembleVotingCost, self).__init__(input_wrapper)
    
    def score_impl(self, x):
        """
        Implementation of the loss function.
        """
        return (-1. * sum([1 if self.y_target(model.predict([x])[0]) else 0 for model in self.models])) / self.num_models


class RandomForest(ModelWithLoss):
    """Class for rebuilding/wrapping the :class:`sklearn.ensemble.RandomForestClassifier` or :class:`sklearn.ensemble.RandomForestRegressor` class.

    Parameters
    ----------
    model : instance of :class:`sklearn.ensemble.RandomForestClassifier` or :class:`sklearn.ensemble.RandomForestRegressor`
        The random forest model.
        
    Raises
    ------
    TypeError
        If `model` is not an instance of :class:`sklearn.ensemble.RandomForestClassifier` or :class:`sklearn.ensemble.RandomForestRegressor`
    """
    def __init__(self, model):
        if not isinstance(model, sklearn.ensemble.RandomForestClassifier) and not isinstance(model, sklearn.ensemble.RandomForestRegressor):
            raise TypeError(f"model has to be an instance of 'sklearn.ensemble.RandomForestClassifier' or 'sklearn.ensemble.RandomForestRegressor' not of {type(model)}")

        self.model = model

        super(RandomForest, self).__init__()

    def predict(self, x):
        """Predict the output of a given input.

        Computes the class label of a given input `x`.

        Parameters
        ----------
        x : `numpy.ndarray`
            The input `x` that is going to be classified.
        
        Returns
        -------
        `int` or `float`
            Prediction.
        """
        return self.model.predict([x])[0]
    
    def get_loss(self, y_target, input_wrapper=None):
        """Creates and returns a loss function.

        Parameters
        ----------
        y_target: `int`, `float` or a callable that returns True if a given prediction is accepted.
            The requested prediction.
        input_wrapper : `callable`
            Converts the input (e.g. if we want to exclude some features/dimensions, we might have to include these missing features before applying any function to it).
        
        Returns
        -------
        :class:`ceml.sklearn.randomforest.EnsembleVotingCost`
            Initialized loss function. The target output is `y_target`.
        """
        return EnsembleVotingCost(self.model.estimators_, y_target, input_wrapper)


class RandomForestCounterfactual(SklearnCounterfactual):
    """Class for computing a counterfactual of a random forest model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model):
        super(RandomForestCounterfactual, self).__init__(model)
    
    def build_loss(self, regularization, x_orig, y_target, pred, grad_mask, C, input_wrapper):
        """
        Build the (non-differentiable) cost function: Regularization + Loss
        """
        regularization = build_regularization_loss(regularization, x_orig)

        loss = RegularizedCost(regularization, self.mymodel.get_loss(y_target, input_wrapper), C=C)

        return loss, None

    def rebuild_model(self, model):
        """Rebuilds a :class:`sklearn.ensemble.RandomForestClassifier` or :class:`sklearn.ensemble.RandomForestRegressor` model.

        Converts a :class:`sklearn.ensemble.RandomForestClassifier` or :class:`sklearn.ensemble.RandomForestRegressor` instance into a :class:`ceml.sklearn.randomforest.RandomForest` instance.

        Parameters
        ----------
        model : instance of :class:`sklearn.ensemble.RandomForestClassifier` or :class:`sklearn.ensemble.RandomForestRegressor`
            The `sklearn` random forest model. 

        Returns
        -------
        :class:`ceml.sklearn.randomforest.RandomForest`
            The wrapped random forest model.
        """
        if not isinstance(model, sklearn.ensemble.RandomForestClassifier) and not isinstance(model, sklearn.ensemble.RandomForestRegressor):
            raise TypeError(f"model has to be an instance of 'sklearn.ensemble.RandomForestClassifier', 'sklearn.ensemble.RandomForestRegressor' not of {type(model)}")

        return RandomForest(model)
    
    def __compute_initial_values(self, x, y_target, features_whitelist):
        """
        Compute initial value for the optimizer.
        """
        result = [x]
        
        for m in self.model.estimators_:
            try:
                result += DecisionTreeCounterfactual(m).compute_all_counterfactuals(x, y_target, features_whitelist=features_whitelist)
            except Exception as ex:
                logging.debug(str(ex))
        
        return result

    def compute_counterfactual(self, x, y_target, features_whitelist=None, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=True, done=None):
        """Computes a counterfactual of a given input `x`.

        Parameters
        ----------
        x : `numpy.ndarray`
            The input `x` whose prediction has to be explained.
        y_target : `int` or `float`
            The requested prediction of the counterfactual.
        feature_whitelist : `list(int)`, optional
            List of feature indices (dimensions of the input space) that can be used when computing the counterfactual.
            
            If `feature_whitelist` is None, all features can be used.

            The default is None.
        regularization : `str` or :class:`ceml.costfunctions.costfunctions.CostFunction`, optional
            Regularizer of the counterfactual. Penalty for deviating from the original input `x`.
            Supported values:
            
                - l1: Penalizes the absolute deviation.
                - l2: Penalizes the squared deviation.

            `regularization` can be a description of the regularization, an instance of :class:`ceml.costfunctions.costfunctions.CostFunction` (or :class:`ceml.costfunctions.costfunctions.DifferentiableCostFunction` if the cost function is differentiable) or None if no regularization is requested.

            If `regularization` is None, no regularization is used.

            The default is "l1".
        C : `float` or `list(float)`, optional
            The regularization strength. If `C` is a list, all values in `C` are tried and as soon as a counterfactual is found, this counterfactual is returned and no other values of `C` are tried.

            If no regularization is used (`regularization=None`), `C` is ignored.

            The default is 1.0
        optimizer : `str` or instance of :class:`ceml.optim.optimizer.Optimizer`, optional
            Name/Identifier of the optimizer that is used for computing the counterfactual.
            See :func:`ceml.optim.optimizer.prepare_optim` for details.

            As an alternative, we can use any (custom) optimizer that is derived from the :class:`ceml.optim.optimizer.Optimizer` class.

            The default is "nelder-mead".

            Note
            ----
            The cost function of a random forest model is not differentiable - we can not use a gradient-based optimization algorithm.
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
            In case of a regression it might not always be possible to achieve a given output/prediction exactly.

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
        # Try to compute a counter factual for each of the models and use this counterfactual as a starting point
        x_start = self.__compute_initial_values(x, y_target, features_whitelist)

        # Check if the prediction of the given input is already consistent with y_target
        done = done if done is not None else y_target if callable(y_target) else lambda y: y == y_target
        self.warn_if_already_done(x, done)

        # Repeat for all C
        if not type(C) == list:
            C = [C]

        for x0 in x_start:
            input_wrapper, x_orig, pred, grad_mask = self.wrap_input(features_whitelist, x0, optimizer)

            for c in C:
                # Build loss
                loss, loss_grad = self.build_loss(regularization, x_orig, y_target, pred, grad_mask, c, input_wrapper)

                # Compute counterfactual
                x_cf, y_cf, delta = self.compute_counterfactual_ex(x, loss, x_orig, loss_grad, optimizer, input_wrapper, False)

                if done(y_cf) == True:
                    if return_as_dict is True:
                        return self._SklearnCounterfactual__build_result_dict(x_cf, y_cf, delta)
                    else:
                        return x_cf, y_cf, delta
        
        raise Exception("No counterfactual found - Consider changing parameters 'C', 'regularization', 'features_whitelist', 'optimizer' and try again")


def randomforest_generate_counterfactual(model, x, y_target, features_whitelist=None, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : a :class:`sklearn.ensemble.RandomForestClassifier` or :class:`sklearn.ensemble.RandomForestRegressor` instance.
        The random forest model that is used for computing the counterfactual.
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

        Note
        ----
        The cost function of a random forest model is not differentiable - we can not use a gradient-based optimization algorithm.
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
    cf = RandomForestCounterfactual(model)

    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, return_as_dict)
