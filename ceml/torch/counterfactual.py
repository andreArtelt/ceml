# -*- coding: utf-8 -*-
import logging
import torch
import numpy as np

from ..backend.torch.layer import create_tensor
from ..backend.torch.costfunctions import RegularizedCost
from ..backend.torch.optimizer import prepare_optim
from ..model import ModelWithLoss, Counterfactual
from .utils import build_regularization_loss, wrap_input


class TorchCounterfactual(Counterfactual):
    """Class for computing a counterfactual of a PyTorch model.

    See parent class :class:`ceml.model.counterfactual.Counterfactual`.

    Parameters
    ----------
    model : instance of :class:`torch.nn.Module` **and** :class:`ceml.model.model.ModelWithLoss`
        The PyTorch model that is used for computing counterfactuals. The model has to be wrapped inside a class that is derived from the classes :class:`torch.nn.Module` and :class:`ceml.model.model.ModelWithLoss`.
    device : :class:`torch.device`
        Specifies the hardware device (e.g. cpu or gpu) we are working on.

        The default is `torch.device("cpu")`.
    
    Raises
    ------
    TypeError
        If model is not an instance of :class:`torch.nn.Module` and :class:`ceml.model.model.ModelWithLoss`.
    """
    def __init__(self, model, device=torch.device("cpu")):
        if not isinstance(model, torch.nn.Module) or not isinstance(model, ModelWithLoss):
            raise TypeError(f"model has to be an instance of 'torch.nn.Module' and of 'ceml.model.ModelWithLoss' not {type(model)}")
        
        self.model = model
        self.device = device

        # Make everything non-differentiable - later on we will make the input differentiable
        for p in self.model.parameters():
            p.requires_grad = False
        
        super(TorchCounterfactual, self).__init__()
    
    def wrap_input(self, features_whitelist, x, optimizer):
        return wrap_input(features_whitelist, x, self.model, optimizer, self.device)

    def build_loss(self, input_wrapper, regularization, x, y_target, C):
        regularization = build_regularization_loss(regularization, create_tensor(x, self.device))

        loss = RegularizedCost(regularization, self.model.get_loss(y_target), C=C)
        loss_npy = lambda z: loss(create_tensor(input_wrapper(z), self.device)).numpy()
        
        def loss_grad_npy(x):
            z_ = create_tensor(input_wrapper(x), self.device)
            z_.requires_grad = True

            self.model.zero_grad()
            loss(z_).backward()

            return z_.grad.numpy()
        
        return loss, loss_npy, loss_grad_npy

    def warn_if_already_done(self, x, done):
        if done(self.model.predict(create_tensor(x, self.device), dim=0).numpy()):
            logging.warning("The prediction of the input 'x' is already consistent with the requested prediction 'y_target' - It might not make sense to search for a counterfactual!")

    def __build_result_dict(self, x_cf, y_cf, delta):
        return {'x_cf': x_cf, 'y_cf': y_cf, 'delta': delta}

    def compute_counterfactual_ex(self, input_wrapper, x_orig, loss, loss_npy, loss_grad_npy, optimizer, optimizer_args, grad_mask, return_as_dict):
        lr_scheduler = None if optimizer_args is None or "lr_scheduler" not in optimizer_args else optimizer_args["lr_scheduler"]
        lr_scheduler_args = None if optimizer_args is None or "lr_scheduler_args" not in optimizer_args else optimizer_args["lr_scheduler_args"]
        tol = None if optimizer_args is None or "tol" not in optimizer_args else optimizer_args["tol"]
        max_iter = None if optimizer_args is None or "max_iter" not in optimizer_args else optimizer_args["max_iter"]
        optimizer_args = None if optimizer_args is None or "args" not in optimizer_args else optimizer_args["args"]
        
        solver = prepare_optim(optimizer, optimizer_args, lr_scheduler, lr_scheduler_args, loss, loss_npy, loss_grad_npy, x_orig, self.model, tol, max_iter, grad_mask, self.device)

        x_cf = input_wrapper(solver())
        y_cf = self.model.predict(create_tensor(x_cf, self.device), dim=0).numpy()
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
        y_target : `int` or `float`
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
        optimizer : `str` or class that is derived from :class:`torch.optim.Optimizer`, optional
            Name/Identifier of the optimizer that is used for computing the counterfactual.
            See :func:`ceml.optim.optimizer.desc_to_optim` for details.

            As an alternative, any optimizer from PyTorch can be used - `optimizer` must be class that is derived from :class:`torch.optim.Optimizer`.

            The default is "nelder-mead".
        optimizer_args : `dict`, optional
            Dictionary containing additional parameters for the optimization algorithm.

            Supported parameters (keys):

                - args: Arguments of the optimization algorithm (e.g. learning rate, momentum, ...)
                - lr_scheduler: Learning rate scheduler (see :class:`torch.optim.lr_scheduler`)
                - lr_scheduler_args: Arguments of the learning rate scheduler
                - tol: Tolerance for termination
                - max_iter: Maximum number of iterations

            If `optimizer_args` is None or if some parameters are missing, default values are used.

            The default is None.

            Note
            ----
            The parameters `tol` and `max_iter` are passed to all optimization algorithms. Whereas the other parameters are only passed to PyTorch optimizers.
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


def generate_counterfactual(model, x, y_target, device=torch.device('cpu'), features_whitelist=None, regularization=None, C=1.0, optimizer="nelder-mead", optimizer_args=None, return_as_dict=True, done=None):
    """Computes a counterfactual of a given input `x`.

    Parameters
    ----------
    model : instance of :class:`torch.nn.Module` and :class:`ceml.model.model.ModelWithLoss`
        The PyTorch model that is used for computing the counterfactual.
    x : `numpy.ndarray`
        The input `x` whose prediction has to be explained.
    y_target : `int` or `float`
        The requested prediction of the counterfactual.
    device : :class:`torch.device`
        Specifies the hardware device (e.g. cpu or gpu) we are working on.

        The default is `torch.device("cpu")`.
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
    optimizer : `str` or class that is derived from :class:`torch.optim.Optimizer`, optional
            Name/Identifier of the optimizer that is used for computing the counterfactual.
            See :func:`ceml.optim.optimizer.desc_to_optim` for details.

            As an alternative, any optimizer from PyTorch can be used - `optimizer` must be class that is derived from :class:`torch.optim.Optimizer`.

            The default is "nelder-mead".
    optimizer_args : `dict`, optional
        Dictionary containing additional parameters for the optimization algorithm.

        Supported parameters (keys):

            - args: Arguments of the optimization algorithm (e.g. learning rate, momentum, ...)
            - lr_scheduler: Learning rate scheduler (see :class:`torch.optim.lr_scheduler`)
            - lr_scheduler_args: Arguments of the learning rate scheduler
            - tol: Tolerance for termination
            - max_iter: Maximum number of iterations

        If `optimizer_args` is None or if some parameters are missing, default values are used.

        The default is None.

        Note
        ----
        The parameters `tol` and `max_iter` are passed to all optimization algorithms. Whereas the other parameters are only passed to PyTorch optimizers.
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
    cf = TorchCounterfactual(model, device)

    return cf.compute_counterfactual(x, y_target, features_whitelist, regularization, C, optimizer, optimizer_args, return_as_dict, done)
