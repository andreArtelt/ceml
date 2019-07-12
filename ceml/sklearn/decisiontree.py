# -*- coding: utf-8 -*-
import numpy as np
import sklearn.tree

from ..model import Model
from .tree import get_leafs_from_tree, score_adjustments, apply_adjustment
from .utils import desc_to_regcost
from .counterfactual import SklearnCounterfactual


class DecisionTreeCounterfactual(SklearnCounterfactual):
    """Class for computing a counterfactual of a decision tree model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model):
        super(DecisionTreeCounterfactual, self).__init__(model)
    
    def rebuild_model(self, model):
        """Rebuild a :class:`sklearn.linear_model.LogisticRegression` model.

        Does nothing.

        Parameters
        ----------
        model : instance of :class:`sklearn.tree.DecisionTreeClassifier` or :class:`sklearn.tree.DecisionTreeRegressor`
            The `sklearn` decision tree model. 

        Returns
        -------
            None
        
        Note
        ----
        In contrast to many other :class:`SklearnCounterfactual` instances, we do do not rebuild the model because we do not need/can compute gradients in a decision tree.
        We compute the set of counterfactuals without using a "common" optimization algorithms like Nelder-Mead.
        """
        if not isinstance(model, sklearn.tree.DecisionTreeClassifier) and not isinstance(model, sklearn.tree.DecisionTreeRegressor):
            raise TypeError(f"model has to be an instance of 'sklearn.tree.DecisionTreeClassifier' or 'sklearn.tree.DecisionTreeRegressor' not {type(model)}")

        return None # We do not need to rebuild the model!
    
    def compute_all_counterfactuals(self, x, y_target, features_whitelist=None, regularization="l1"):
        """Computes all counterfactuals of a given input `x`.

        Parameters
        ----------
        model : a :class:`sklearn.tree.DecisionTreeClassifier` or :class:`sklearn.tree.DecisionTreeRegressor` instance.
            The decision tree model that is used for computing the counterfactual.
        x : `numpy.ndarray`
            The input `x` whose prediction is supposed to be explained.
        y_target : `int` or `float` or a callable that returns True if a given prediction is accepted.
            The requested prediction of the counterfactual.
        features_whitelist : `list(int)`, optional
            List of feature indices (dimensions of the input space) that can be used when computing the counterfactual.
            
            If `features_whitelist` is None, all features can be used.

            The default is None.
        regularization : `str` or callable, optional
            Regularizer of the counterfactual. Penalty for deviating from the original input `x`.

            Supported values:
            
                - l1: Penalizes the absolute deviation.
                - l2: Penalizes the squared deviation.
            
            You can use your own custom penalty function by setting `regularization` to a callable that can be called on a potential counterfactual and returns a scalar.
            
            If `regularization` is None, no regularization is used.

            The default is "l1".
        
        Returns
        -------
        `list(np.array)`
            List of all counterfactuals.
        
        Raises
        ------
        TypeError
            If an invalid argument is passed to the function.
        
        ValueError
            If no counterfactual exists.
        """
        is_classifier = None
        if isinstance(self.model, sklearn.tree.DecisionTreeClassifier):
            is_classifier = True
        elif isinstance(self.model, sklearn.tree.DecisionTreeRegressor):
            is_classifier = False

        if isinstance(regularization, str):
            regularization = desc_to_regcost(regularization, x, None)
        elif not callable(regularization):
            raise TypeError("'regularization' has to be either callable or a valid description of a supported regularization")

        # Collect all leafs
        leafs = get_leafs_from_tree(self.model.tree_, classifier=is_classifier)

        # Filter leafs for predictions
        if callable(y_target):
            leafs = list(filter(lambda x: y_target(x[-1][2]), leafs))
        else:
            leafs = list(filter(lambda x: x[-1][2] == y_target, leafs))

        if len(leafs) == 0:
            raise ValueError("Tree does not has a path/leaf yielding the requested outcome specified in 'y_target'")
        
        # Compute path of sample
        path_of_x = list(self.model.decision_path([x]).indices)

        # Score and sort all counterfactuals of the sample
        counterfactuals = score_adjustments(x, path_of_x, leafs, regularization)

        counterfactuals = [apply_adjustment(x, cf[2]) for cf in counterfactuals]

        # Drop all counterfactuals with changes in protected attributes
        if features_whitelist is not None:
            def used_protected_attributes(x, cf, features_whitelist):
                d = x - cf
                for i in range(d.shape[0]):
                    if i not in features_whitelist and d[i] != 0:
                        return True
                return False

            counterfactuals = list(filter(lambda z: not used_protected_attributes(x, z, features_whitelist), counterfactuals))

            if len(counterfactuals) == 0:
                raise ValueError("After filtering for whitelisted features, the tree has no longer a path/leaf yielding the requested outcome specified in 'y_target'.")

        return counterfactuals


    def compute_counterfactual(self, x, y_target, features_whitelist=None, regularization="l1", C=None, optimizer=None, return_as_dict=True):
        """Computes a counterfactual of a given input `x`.

        Parameters
        ----------
        model : a :class:`sklearn.tree.DecisionTreeClassifier` or :class:`sklearn.tree.DecisionTreeRegressor` instance.
            The decision tree model that is used for computing the counterfactual.
        x : `numpy.ndarray`
            The input `x` whose prediction is supposed to be explained.
        y_target : `int` or `float` or a callable that returns True if a given prediction is accepted.
            The requested prediction of the counterfactual.
        features_whitelist : `list(int)`, optional
            List of feature indices (dimensions of the input space) that can be used when computing the counterfactual.
            
            If `features_whitelist` is None, all features can be used.

            The default is None.
        regularization : `str` or callable, optional
            Regularizer of the counterfactual. Penalty for deviating from the original input `x`.

            Supported values:
            
                - l1: Penalizes the absolute deviation.
                - l2: Penalizes the squared deviation.
            
            You can use your own custom penalty function by setting `regularization` to a callable that can be called on a potential counterfactual and returns a scalar.
            
            If `regularization` is None, no regularization is used.

            The default is "l1".
        C : None
            Not used - is always None.
            
            The only reason for including this parameter is to match the signature of other :class:`ceml.sklearn.counterfactual.SklearnCounterfactual` children.
        optimizer : None
            Not used - is always None.
            
            The only reason for including this parameter is to match the signature of other :class:`ceml.sklearn.counterfactual.SklearnCounterfactual` children.
        return_as_dict : `boolean`, optional
            If True, returns the counterfactual, its prediction and the needed changes to the input as dictionary.
            If False, the results are returned as a triple.

            The default is True.

        Returns
        -------
        `dict` or `triple`
            A dictionary where the counterfactual is stored in 'x_cf', its prediction in 'y_cf' and the changes to the original input in 'delta'.

            (x_cf, y_cf, delta) : triple if `return_as_dict` is False
        """
        # Check if the prediction of the given input is already consistent with y_target
        done = y_target if callable(y_target) else lambda y: y == y_target        
        self.warn_if_already_done(x, done)

        # Compute all counterfactual
        counterfactuals = self.compute_all_counterfactuals(x, y_target, features_whitelist, regularization)

        # Select the one with the smallest score
        x_cf = counterfactuals[0]
        delta = x - x_cf
        y_cf = self.model.predict([x_cf])[0]

        if return_as_dict is True:
            return self._SklearnCounterfactual__build_result_dict(x_cf, y_cf, delta)
        else:
            return x_cf, y_cf, delta


def decisiontree_generate_counterfactual(model, x, y_target, features_whitelist=None, regularization="l1", return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : a :class:`sklearn.tree.DecisionTreeClassifier` or :class:`sklearn.tree.DecisionTreeRegressor` instance.
        The decision tree model that is used for computing the counterfactual.
    x : `numpy.ndarray`
        The input `x` whose prediction has to be explained.
    y_target : `int` or `float` or a callable that returns True if a given prediction is accepted.
        The requested prediction of the counterfactual.
    feature_whitelist : `list(int)`, optional
        List of feature indices (dimensions of the input space) that can be used when computing the counterfactual.
        
        If `feature_whitelist` is None, all features can be used.

        The default is None.
    regularization : `str` or callable, optional
        Regularizer of the counterfactual. Penalty for deviating from the original input `x`.
        Supported values:
        
            - l1: Penalizes the absolute deviation.
            - l2: Penalizes the squared deviation.
        
        You can use your own custom penalty function by setting `regularization` to a callable that can be called on a potential counterfactual and returns a scalar.
        
        If `regularization` is None, no regularization is used.

        The default is "l1".
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
    cf = DecisionTreeCounterfactual(model)
    
    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, return_as_dict=return_as_dict)
