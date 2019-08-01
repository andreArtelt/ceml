# -*- coding: utf-8 -*-
import sklearn.neighbors

from ..backend.jax.layer import create_tensor
from ..backend.jax.costfunctions import TopKMinOfListDistCost, RegularizedCost
from ..model import ModelWithLoss
from .utils import desc_to_dist
from .utils import build_regularization_loss
from .counterfactual import SklearnCounterfactual


class KNN(ModelWithLoss):
    """Class for rebuilding/wrapping the :class:`sklearn.neighbors.KNeighborsClassifier` and :class:`sklearn.neighbors.KNeighborsRegressor` classes.

    The :class:`KNN` class rebuilds a `sklearn` knn model.

    Parameters
    ----------
    model : instance of :class:`sklearn.neighbors.KNeighborsClassifier` or :class:`sklearn.neighbors.KNeighborsRegressor`
        The knn model.
    dist : `str` or callable, optional
        Computes the distance between a prototype and a data point.

        Supported values:

            - l1: Penalizes the absolute deviation.
            - l2: Penalizes the squared deviation.

        You can use your own custom distance function by setting `dist` to a callable that can be called on a data point and returns a scalar.

        The default is "l2".

        **Note:** `dist` must not be None.

    Attributes
    ----------
    X : `numpy.array`
        The training data set.
    y : `numpy.array`
        The ground truth of the training data set.
    dist : `callable`
        The distance function.

    Raises
    ------
    TypeError
        If `model` is not an instance of :class:`sklearn.neighbors.KNeighborsClassifier` or :class:`sklearn.neighbors.KNeighborsRegressor`
    """
    def __init__(self, model, dist="l2"):
        if not isinstance(model, sklearn.neighbors.KNeighborsClassifier) and not isinstance(model, sklearn.neighbors.KNeighborsRegressor):
            raise TypeError(f"model has to be an instance of 'sklearn.neighbors.KNeighborsClassifier' or 'sklearn.neighbors.KNeighborsRegressor' but not of {type(model)}")

        self.X = model._fit_X
        self.y = model._y
        self.n_neighbors = model.n_neighbors
        self.dist = dist

        super(KNN, self).__init__()
    
    def predict(self, x):
        """
        Note
        ----
        This function is a placeholder only.

        This function does not predict anything and just returns the given input.
        """
        return x    # Note: Identity function is necessary because our knn loss function works on the input (not on the final classification) 
    
    def get_loss(self, y_target, pred=None):
        """Creates and returns a loss function.

        Builds a cost function where we penalize the minimum distance to the nearest prototype which is consistent with the target `y_target`.

        Parameters
        ----------
        y_target : `int`
            The target class.
        pred : `callable`, optional
            A callable that maps an input to an input. E.g. using the :class:`ceml.optim.input_wrapper.InputWrapper` class.

            If `pred` is None, no transformation is applied to the input before passing it into the loss function.

            The default is None.
        
        Returns
        -------
        :class:`ceml.backend.jax.costfunctions.TopKMinOfListDistCost`
            Initialized cost function. Target label is `y_target`.
        """
        # Collect all prototypes of the target class
        if callable(y_target):
            target_samples = create_tensor(self.X[[y_target(y) for y in self.y], :])
        else:
            target_samples = create_tensor(self.X[self.y == y_target, :])
        
        # Build a loss function that penalize the distance to the nearest prototype
        if not callable(self.dist):
            self.dist = desc_to_dist(self.dist)
        
        return TopKMinOfListDistCost(self.dist, target_samples, self.n_neighbors, pred)


class KnnCounterfactual(SklearnCounterfactual):
    """Class for computing a counterfactual of a knn model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model, dist="l2"):
        self.dist = dist    # TODO: Extract distance from model

        super(KnnCounterfactual, self).__init__(model)

    def rebuild_model(self, model):
        """Rebuilds a :class:`sklearn.neighbors.KNeighborsClassifier` or :class:`sklearn.neighbors.KNeighborsRegressor` model.

        Converts a :class:`sklearn.neighbors.KNeighborsClassifier` or :class:`sklearn.neighbors.KNeighborsRegressor` instance into a :class:`ceml.sklearn.knn.KNN` instance.

        Parameters
        ----------
        model : instace of :class:`sklearn.neighbors.KNeighborsClassifier` or :class:`sklearn.neighbors.KNeighborsRegressor`
            The `sklearn` knn model. 

        Returns
        -------
        :class:`ceml.sklearn.knn.KNN`
            The wrapped knn model.
        """
        if not isinstance(model, sklearn.neighbors.KNeighborsClassifier) and not isinstance(model, sklearn.neighbors.KNeighborsRegressor):
            raise TypeError(f"model has to be an instance of 'sklearn.neighbors.KNeighborsClassifier' or 'sklearn.neighbors.KNeighborsRegressor' but not of {type(model)}")
    
        return KNN(model, self.dist)


def knn_generate_counterfactual(model, x, y_target, features_whitelist=None, dist="l2", regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : a :class:`sklearn.neighbors.KNeighborsClassifier` or :class:`sklearn.neighbors.KNeighborsRegressor` instance.
        The knn model that is used for computing the counterfactual.
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
    """
    cf = KnnCounterfactual(model, dist)

    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, return_as_dict, done)
