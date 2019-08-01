# -*- coding: utf-8 -*-
import numpy as np
import sklearn_lvq

from ..backend.jax.layer import create_tensor
from ..backend.jax.costfunctions import MinOfListDistCost, MinOfListDistExCost, RegularizedCost
from ..model import ModelWithLoss
from .utils import desc_to_dist
from .counterfactual import SklearnCounterfactual


class LVQ(ModelWithLoss):
    """Class for rebuilding/wrapping the :class:`sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` and :class:`sklearn_lvq.LmrslvqModel` classes.

    The :class:`LVQ` class rebuilds a `sklearn-lvq` lvq model.

    Parameters
    ----------
    model : instance of :class:`sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` or :class:`sklearn_lvq.LmrslvqModel`
        The lvq model.
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
    prototypes : `numpy.array`
        The prototypes.
    labels : `numpy.array`
        The labels of the prototypes. 
    dist : `callable`
        The distance function.
    model : `object`
        The original `sklearn-lvq` model.
    model_class : `class`
        The class of the `sklearn-lvq` model.

    Raises
    ------
    TypeError
        If `model` is not an instance of :class:`sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` or :class:`sklearn_lvq.LmrslvqModel`
    """
    def __init__(self, model, dist="l2"):
        if not any([isinstance(model, t) for t in [sklearn_lvq.GlvqModel, sklearn_lvq.GmlvqModel, sklearn_lvq.LgmlvqModel, sklearn_lvq.RslvqModel, sklearn_lvq.MrslvqModel, sklearn_lvq.LmrslvqModel]]):
            raise TypeError(f"model has to be an instance of 'sklearn_lvq.GlvqModel', 'sklearn_lvq.GmlvqModel', 'sklearn_lvq.LgmlvqModel', 'sklearn_lvq.RslvqModel', 'sklearn_lvq.MrslvqModel' or 'sklearn_lvq.LmrslvqModel' but not of {type(model)}")
        
        self.prototypes = model.w_
        self.labels = model.c_w_
        self.dist = dist
        self.model = model
        self.model_class = model.__class__

        # The model might have learned its own distance metric
        if isinstance(model, sklearn_lvq.GmlvqModel) or isinstance(model, sklearn_lvq.MrslvqModel):
            self.dist_mat = create_tensor(np.dot(model.omega_.T, model.omega_))
        elif isinstance(model, sklearn_lvq.LgmlvqModel) or isinstance(model, sklearn_lvq.LmrslvqModel):
            if model.classwise == True:
                self.omegas = [create_tensor(np.dot(omega.T, omega)) for omega in model.omegas_]
                self.classwise = True
                self.dist_mats = None
            else:
                self.dist_mats = [create_tensor(np.dot(omega.T, omega)) for omega in model.omegas_]
                self.classwise = False

        super(LVQ, self).__init__()

    def predict(self, x):
        """
        Note
        ----
        This function is a placeholder only.

        This function does not predict anything and just returns the given input.
        """
        return x    # Note: Identity function is necessary because our lvq loss function works on the input (not on the final classification) 

    def get_loss(self, y_target, pred=None):
        """Creates and returns a loss function.

        Builds a cost function where we penalize the minimum distance to the nearest prototype which is consistent with the target `y_target`.

        Parameters
        ----------
        y_target : `int`
            The target class.
        pred : `callable`, optional
            A callable that maps an input to an input. E.g. using the :class:`ceml.optim.input_wrapper.InputWrapper` class.

            If `pred` is None, no transformation is applied to the input before putting it into the loss function.

            The default is None.
        
        Returns
        -------
        :class:`ceml.backend.jax.costfunctions.MinOfListDistCost`
            Initialized cost function. Target label is `y_target`.
        """
        # Collect all prototypes of the target class
        if callable(y_target):
            target_samples = create_tensor(self.prototypes[[y_target(y) for y in self.labels], :])
        else:
            target_samples = create_tensor(self.prototypes[self.labels == y_target, :])

        # Build a loss function that penalize the distance to the nearest prototype - note that the model might have learned its own distance matrix
        if self.model_class == sklearn_lvq.GmlvqModel or self.model_class == sklearn_lvq.MrslvqModel:
            self.dist_mats = [self.dist_mat for _ in target_samples]

            return MinOfListDistExCost(self.dist_mats, target_samples, pred)
        elif self.model_class == sklearn_lvq.LgmlvqModel or self.model_class == sklearn_lvq.LmrslvqModel:
            if self.classwise == True:
                self.dist_mats = []
                if callable(y_target):
                    for y in self.labels:
                        if y_target(y) is True:
                            self.dist_mats.append(self.omegas[y])
                else:
                    for y in self.labels:
                        if y == y_target:
                            self.dist_mats.append(self.omegas[y])
            
            return MinOfListDistExCost(self.dist_mats, target_samples, pred)
        else:
            if not callable(self.dist):
                self.dist = desc_to_dist(self.dist)

            return MinOfListDistCost(self.dist, target_samples, pred)


class LvqCounterfactual(SklearnCounterfactual):
    """Class for computing a counterfactual of a lvq model.

    See parent class :class:`ceml.sklearn.counterfactual.SklearnCounterfactual`.
    """
    def __init__(self, model, dist="l2"):
        self.dist = dist    # TODO: Extract distance from model

        super(LvqCounterfactual, self).__init__(model)
    
    def rebuild_model(self, model):
        """Rebuilds a :class:`sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` or :class:`sklearn_lvq.LmrslvqModel` model.

        Converts a :class:`sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` or :class:`sklearn_lvq.LmrslvqModel` instance into a :class:`ceml.sklearn.lvq.LVQ` instance.

        Parameters
        ----------
        model : instace of :class:`sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` or :class:`sklearn_lvq.LmrslvqModel`
            The `sklearn-lvq` lvq model. 

        Returns
        -------
        :class:`ceml.sklearn.lvq.LVQ`
            The wrapped lvq model.
        """
        if not any([isinstance(model, t) for t in [sklearn_lvq.GlvqModel, sklearn_lvq.GmlvqModel, sklearn_lvq.LgmlvqModel, sklearn_lvq.RslvqModel, sklearn_lvq.MrslvqModel, sklearn_lvq.LmrslvqModel]]):
            raise TypeError(f"model has to be an instance of 'sklearn_lvq.GlvqModel', 'sklearn_lvq.GmlvqModel', 'sklearn_lvq.LgmlvqModel', 'sklearn_lvq.RslvqModel', 'sklearn_lvq.MrslvqModel' or 'sklearn_lvq.LmrslvqModel' but not of {type(model)}")
    
        return LVQ(model, self.dist)


def lvq_generate_counterfactual(model, x, y_target, features_whitelist=None, dist="l2", regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : a :class:`sklearn.neighbors.sklearn_lvq.GlvqModel`, :class:`sklearn_lvq.GmlvqModel`, :class:`sklearn_lvq.LgmlvqModel`, :class:`sklearn_lvq.RslvqModel`, :class:`sklearn_lvq.MrslvqModel` or :class:`sklearn_lvq.LmrslvqModel` instance.
        The lvq model that is used for computing the counterfactual.

        **Note:** Only lvq models from sklearn-lvq are supported.
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
    cf = LvqCounterfactual(model, dist)

    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, return_as_dict)    
