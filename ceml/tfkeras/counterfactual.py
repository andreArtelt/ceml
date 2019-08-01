# -*- coding: utf-8 -*-
import logging
import tensorflow as tf
import numpy as np

from ..backend.tensorflow.layer import create_tensor, create_mutable_tensor
from ..backend.tensorflow.costfunctions import RegularizedCost
from ..backend.tensorflow.optimizer import prepare_optim
from ..model import ModelWithLoss, Counterfactual
from .utils import build_regularization_loss, wrap_input


class TfCounterfactual(Counterfactual):
    """Class for computing a counterfactual of a tensorflow model.

    See parent class :class:`ceml.model.counterfactual.Counterfactual`.

    Parameters
    ----------
    model : instance of :class:`ceml.model.model.ModelWithLoss`
        The tensorflow model that is used for computing counterfactuals. The model has to be wrapped inside a class that is derived from the :class:`ceml.model.model.ModelWithLoss` class.

    Raises
    ------
    TypeError
        If model is not an instance of :class:`ceml.model.model.ModelWithLoss`.
    Exception
        If eager execution is not enabled.
    """
    def __init__(self, model):
        if not tf.executing_eagerly():
            raise Exception("Eager mode is not enabled - Please enable eager execution")
        if not isinstance(model, ModelWithLoss):
            raise TypeError(f"model has to be an instance of 'ceml.model.ModelWithLoss' not {type(model)}")
        
        self.model = model

        super(TfCounterfactual, self).__init__()
    
    def wrap_input(self, features_whitelist, x, optimizer):
        return wrap_input(features_whitelist, x, self.model, optimizer)

    def build_loss(self, input_wrapper, regularization, x, y_target, C):
        regularization = build_regularization_loss(regularization, create_tensor(x))

        loss = RegularizedCost(regularization, self.model.get_loss(y_target), C=C)
        loss_npy = lambda z: loss(create_tensor(input_wrapper(z))).numpy()
        
        def loss_grad_npy(x):
            z_ = create_mutable_tensor(input_wrapper(x))

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(z_)
                loss_value = loss(z_)
            
                return tape.gradient(loss_value, z_).numpy()
        
        return loss, loss_npy, loss_grad_npy

    def warn_if_already_done(self, x, done):
        if done(self.model.predict(np.array([x]))):
            logging.warning("The prediction of the input 'x' is already consistent with the requested prediction 'y_target' - It might not make sense to search for a counterfactual!")

    def __build_result_dict(self, x_cf, y_cf, delta):
        return {'x_cf': x_cf, 'y_cf': y_cf, 'delta': delta}
    
    def compute_counterfactual_ex(self, input_wrapper, x_orig, loss, loss_npy, loss_grad_npy, optimizer, optimizer_args, grad_mask, return_as_dict):
        tol = None if optimizer_args is None or "tol" not in optimizer_args else optimizer_args["tol"]
        max_iter = None if optimizer_args is None or "max_iter" not in optimizer_args else optimizer_args["max_iter"]
        
        solver = prepare_optim(optimizer, loss, loss_npy, loss_grad_npy, x_orig, self.model, tol, max_iter, grad_mask)

        x_cf = input_wrapper(solver())
        y_cf = self.model.predict(np.array([x_cf]))
        delta = input_wrapper(x_orig) - x_cf

        if return_as_dict is True:
            return self.__build_result_dict(x_cf, y_cf, delta)
        else:
            return x_cf, y_cf, delta

    def compute_counterfactual(self, x, y_target, features_whitelist=None, regularization=None, C=1.0, optimizer="nelder-mead", optimizer_args=None, return_as_dict=True, done=None):
        """Computes a counterfactual of a given input `x`.

        Parameters
        ----------
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
        C : `float` or `list(float)`, optional
            The regularization strength. If `C` is a list, all values in `C` are tried and as soon as a counterfactual is found, this counterfactual is returned and no other values of `C` are tried.

            `C` is ignored if no regularization is used (`regularization=None`).

            The default is 1.0
        optimizer : `str` or instance of :class:`tf.train.Optimizer`, optional
            Name/Identifier of the optimizer that is used for computing the counterfactual.
            See :func:`ceml.optim.optimizer.desc_to_optim` for details.

            As an alternative, any optimizer from tensorflow can be used - `optimizer` must be an an instance of :class:`tf.train.Optimizer`.

            The default is "nelder-mead".
        optimizer_args : `dict`, optional
            Dictionary containing additional parameters for the optimization algorithm.

            Supported parameters (keys):

                - tol: Tolerance for termination
                - max_iter: Maximum number of iterations

            If `optimizer_args` is None or if some parameters are missing, default values are used.

            The default is None.
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
        dict or triple
            A dictionary where the counterfactual is stored in 'x_cf', its prediction in 'y_cf' and the changes to the original input in 'delta'.

            (x_cf, y_cf, delta) : triple if `return_as_dict` is False
        
        Raises
        ------
        Exception
            If no counterfactual was found.
        """
        # Hide the input in a wrapper if we can use a subset of features only
        input_wrapper, x_orig, _, grad_mask = self.wrap_input(features_whitelist, x, optimizer)
        
        # Check if the prediction of the given input is already consistent with y_target
        done = done = done if done is not None else y_target if callable(y_target) else lambda y: y == y_target
        self.warn_if_already_done(x, done)

        # Repeat for all C
        if not type(C) == list:
            C = [C]

        for c in C:
            # Build loss
            loss, loss_npy, loss_grad_npy = self.build_loss(input_wrapper, regularization, x, y_target, c)
            
            # Add gradient mask
            loss_grad_npy_ = loss_grad_npy
            if grad_mask is not None:
                loss_grad_npy_ = lambda z: np.multiply(loss_grad_npy(z), grad_mask)

            # Compute counterfactual
            x_cf, y_cf, delta = self.compute_counterfactual_ex(input_wrapper, x_orig, loss, loss_npy, loss_grad_npy_, optimizer, optimizer_args, grad_mask, False)

            if done(y_cf) == True:
                if return_as_dict is True:
                    return self.__build_result_dict(x_cf, y_cf, delta)
                else:
                    return x_cf, y_cf, delta
        
        raise Exception("No counterfactual found - Consider changing parameters 'C', 'regularization', 'features_whitelist', 'optimizer' and try again")


def generate_counterfactual(model, x, y_target, features_whitelist=None, regularization=None, C=1.0, optimizer="nelder-mead", optimizer_args=None, return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : instance of :class:`ceml.model.model.ModelWithLoss`
        The tensorflow model that is used for computing the counterfactual.
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
    C : `float` or `list(float)`, optional
        The regularization strength. If `C` is a list, all values in `C` are tried and as soon as a counterfactual is found, this counterfactual is returned and no other values of `C` are tried.

        If no regularization is used (`regularization=None`), `C` is ignored.

        The default is 1.0
    optimizer : `str` or instance of :class:`tf.train.Optimizer`, optional
            Name/Identifier of the optimizer that is used for computing the counterfactual.
            See :func:`ceml.optim.optimizer.desc_to_optim` for details.

            As an alternative, any optimizer from tensorflow can be used - `optimizer` must be an an instance of :class:`tf.train.Optimizer`.

            The default is "nelder-mead".
    optimizer_args : `dict`, optional
        Dictionary containing additional parameters for the optimization algorithm.

        Supported parameters (keys):

            - tol: Tolerance for termination
            - max_iter: Maximum number of iterations

        If `optimizer_args` is None or if some parameters are missing, default values are used.

        The default is None.
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
    dict or triple
        A dictionary where the counterfactual is stored in 'x_cf', its prediction in 'y_cf' and the changes to the original input in 'delta'.

        (x_cf, y_cf, delta) : triple if `return_as_dict` is False
    """
    cf = TfCounterfactual(model)

    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, optimizer_args, return_as_dict, done)
