# -*- coding: utf-8 -*-
import torch
import numpy as np
import inspect
from ..layer import create_tensor
from ....optim import Optimizer
from ....optim import prepare_optim as desc_to_optim_scipy


class TorchOptimizer(Optimizer):
    """Wrapper for a PyTorch optimization algorithm.

    The :class:`TorchOptimizer` provides an interface for wrapping an arbitrary PyTorch optimization algorithm (see :class:`torch.optim`) and minimizing a given loss function.
    """
    def __init__(self):
        self.model = None
        self.loss = None
        self.tol = None
        self.max_iter = None
        self.device = None

        self.x = None

        self.optim = None
        self.lr_scheduler = None

        self.grad_mask = None

        super(TorchOptimizer, self).__init__()
    
    def init(self, model, loss, x, optim, optim_args, lr_scheduler=None, lr_scheduler_args=None, tol=None, max_iter=1, grad_mask=None, device=torch.device("cpu")):
        """
        Initializes all parameters.

        Parameters
        ----------
        model : instance of :class:`torch.nn.Module`
            The model that is to be used.
        loss : instance of :class:`ceml.backend.torch.costfunctions.RegularizedCost`
            The loss that has to be minimized.
        x : `numpy.ndarray`
            The starting value of `x` - usually this is the original input whose prediction has to be explained..
        optim : instance of `torch.optim.Optimizer`
            Optimizer for minimizing the loss.
        optim_args : `dict`
            Arguments of the optimization algorithm (e.g. learning rate, momentum, ...)
        lr_scheduler : Learning rate scheduler (see :class:`torch.optim.lr_scheduler`)
            Learning rate scheduler (see :class:`torch.optim.lr_scheduler`).

            The default is None.
        lr_scheduler_args : `dict`, optional
            Arguments of the learning rate scheduler.

            The default is None.
        tol : `float`, optional
            Tolerance for termination.

            The default is 0.0
        max_iter : `int`, optional
            Maximum number of iterations.

            The default is 1.
        grad_mask : `numpy.array`, optional
            Mask that is multiplied element wise on top of the gradient - can be used to hold some dimensions constant. 

            If `grad_mask` is None, no gradient mask is used.

            The default is None.
        device : :class:`torch.device`
            Specifies the hardware device (e.g. cpu or gpu) we are working on.

            The default is `torch.device("cpu")`.
        
        Raises
        ------
        TypeError
            If the type of `loss` or `model` is not correct.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model must be an instance of torch.nn.Module not of {type(model)}")
        if not callable(loss):
            raise TypeError("loss has to be callable")
        
        self.model = model
        self.loss = loss
        self.tol = 0.0 if tol is None else tol
        self.max_iter = max_iter
        self.device = device

        self.x = create_tensor(np.copy(x), device)
        self.x.requires_grad = True

        self.optim = optim([self.x], **optim_args)
        self.lr_scheduler = None
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optim, **lr_scheduler_args)

        if grad_mask is not None:
            self.grad_mask = create_tensor(grad_mask)
        else:
            self.grad_mask = self.x.new_ones(self.x.shape)

    def is_grad_based(self):
        return True

    def minimize(self):
        old_loss = float("inf")
        for _ in range(self.max_iter):
            self.optim.zero_grad()
            l = self.loss(self.x)
            l.backward()

            self.x.grad *= self.grad_mask

            self.optim.step()

            if np.abs(old_loss - l.detach().numpy()) <= self.tol:
                break
            old_loss = l.detach().numpy()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
        return self.x.detach().numpy()

    def __call__(self):
        return self.minimize()


def prepare_optim(optimizer, optimizer_args, lr_scheduler, lr_scheduler_args, loss, loss_npy, loss_grad_npy, x_orig, model, tol, max_iter, grad_mask, device):
    if isinstance(optimizer, str) or isinstance(optimizer, Optimizer):
        return desc_to_optim_scipy(optimizer, loss_npy, x_orig, loss_grad_npy, tol, max_iter)
    elif inspect.isclass(optimizer) == True:
        optim = TorchOptimizer()
        optim.init(model, loss, x_orig, optimizer, optimizer_args, lr_scheduler, lr_scheduler_args, tol, max_iter, grad_mask, device)
        return optim
    else:
        raise TypeError("Invalid type of argument 'optimizer'.\n'optimizer' has to be a string or the name of a class that is derived from torch.optim.Optimizer")
