# -*- coding: utf-8 -*-
import sklearn
import sklearn_lvq

from .softmaxregression import softmaxregression_generate_counterfactual
from .linearregression import linearregression_generate_counterfactual
from .naivebayes import gaussiannb_generate_counterfactual
from .decisiontree import decisiontree_generate_counterfactual
from .randomforest import randomforest_generate_counterfactual
from .isolationforest import isolationforest_generate_counterfactual
from .knn import knn_generate_counterfactual
from .lvq import lvq_generate_counterfactual
from .lda import lda_generate_counterfactual
from .qda import qda_generate_counterfactual
from .pipeline import pipeline_generate_counterfactual


def generate_counterfactual(model, x, y_target, features_whitelist=None, dist="l2", regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : object
        The sklearn model that is used for computing the counterfactual.
    x : `numpy.ndarray`
        The input `x` whose prediction has to be explained.
    y_target : `int` or `float` or a callable that returns True if a given prediction is accepted.
        The requested prediction of the counterfactual.
    features_whitelist : `list(int)`, optional
        List of feature indices (dimensions of the input space) that can be used when computing the counterfactual.
        
        If `features_whitelist` is None, all features can be used.

        The default is None.
    dist : `str` or callable, optional
        Computes the distance between a prototype and a data point.

        Supported values:

            - l1: Penalizes the absolute deviation.
            - l2: Penalizes the squared deviation.

        You can use your own custom distance function by setting `dist` to a callable that can be called on a data point and returns a scalar.

        The default is "l1".

        **Note:** `dist` must not be None.

        Note
        ----
        Only needed if `model` is a LVQ or KNN model!
    regularization : `str` or callable, optional
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
    ValueError
        If `model` contains an unsupported model.
    """
    if isinstance(model, sklearn.pipeline.Pipeline):
        return pipeline_generate_counterfactual(model, x, y_target, features_whitelist=features_whitelist, regularization=regularization, C=C, optimizer=optimizer, return_as_dict=return_as_dict, done=done)
    elif isinstance(model, sklearn.linear_model.LogisticRegression):
        return softmaxregression_generate_counterfactual(model, x, y_target, features_whitelist=features_whitelist, regularization=regularization, C=C, optimizer=optimizer, return_as_dict=return_as_dict, done=done)
    elif isinstance(model, sklearn.linear_model.base.LinearModel):
        return linearregression_generate_counterfactual(model, x, y_target, features_whitelist=features_whitelist, regularization=regularization, C=C, optimizer=optimizer, return_as_dict=return_as_dict, done=done)
    elif isinstance(model, sklearn.naive_bayes.GaussianNB):
        return gaussiannb_generate_counterfactual(model, x, y_target, features_whitelist=features_whitelist, regularization=regularization, C=C, optimizer=optimizer, return_as_dict=return_as_dict, done=done)
    elif isinstance(model, sklearn.discriminant_analysis.LinearDiscriminantAnalysis):
        return lda_generate_counterfactual(model, x, y_target, features_whitelist=features_whitelist, regularization=regularization, C=C, optimizer=optimizer, return_as_dict=return_as_dict, done=done)
    elif isinstance(model, sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis):
        return qda_generate_counterfactual(model, x, y_target, features_whitelist=features_whitelist, regularization=regularization, C=C, optimizer=optimizer, return_as_dict=return_as_dict, done=done)
    elif isinstance(model, sklearn.tree.DecisionTreeClassifier) or isinstance(model, sklearn.tree.DecisionTreeRegressor):
        return decisiontree_generate_counterfactual(model, x, y_target, features_whitelist=features_whitelist, regularization=regularization, return_as_dict=return_as_dict, done=done)
    elif isinstance(model, sklearn.ensemble.RandomForestClassifier) or isinstance(model, sklearn.ensemble.RandomForestRegressor):
        return randomforest_generate_counterfactual(model, x, y_target, features_whitelist=features_whitelist, regularization=regularization, C=C, optimizer=optimizer, return_as_dict=return_as_dict)
    elif isinstance(model, sklearn.ensemble.IsolationForest):
        return isolationforest_generate_counterfactual(model, x, y_target, features_whitelist=features_whitelist, regularization=regularization, C=C, optimizer=optimizer, return_as_dict=return_as_dict)
    elif isinstance(model, sklearn.neighbors.KNeighborsClassifier) or isinstance(model, sklearn.neighbors.KNeighborsRegressor):
        return knn_generate_counterfactual(model, x, y_target, dist=dist, features_whitelist=features_whitelist, regularization=regularization, C=C, optimizer=optimizer, return_as_dict=return_as_dict, done=done)
    elif any([isinstance(model, t) for t in [sklearn_lvq.GlvqModel, sklearn_lvq.GmlvqModel, sklearn_lvq.LgmlvqModel, sklearn_lvq.RslvqModel, sklearn_lvq.MrslvqModel, sklearn_lvq.LmrslvqModel]]):
        return lvq_generate_counterfactual(model, x, y_target, dist=dist, features_whitelist=features_whitelist, regularization=regularization, C=C, optimizer=optimizer, return_as_dict=return_as_dict, done=done)
    else:
        raise ValueError(f"{type(model)} is not supported")
