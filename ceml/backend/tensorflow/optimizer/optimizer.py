# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import inspect
from ..layer import create_tensor, create_mutable_tensor
from ....optim import Optimizer
from ....optim import prepare_optim as desc_to_optim_scipy


class TfOptimizer(Optimizer):
    """Wrapper for a tensorflow optimization algorithm.

    The :class:`TfOptimizer` provides an interface for wrapping an arbitrary tensorflow optimization algorithm (see :class:`tf.train.Optimizer`) and minimizing a given loss function.
    """
    def __init__(self):
        self.model = None
        self.loss = None
        self.tol = None
        self.max_iter = None

        self.x = None

        self.optim = None

        self.grad_mask = None

        super(TfOptimizer, self).__init__()
    
    def init(self, model, loss, x, optim, tol=None, max_iter=1, grad_mask=None):
        """
        Initializes all parameters.

        Parameters
        ----------
        model : `callable` or instance of :class:`tf.keras.Model`
            The model that is to be used.
        loss : instance of :class:`ceml.backend.tensorflow.costfunctions.RegularizedCost`
            The loss that has to be minimized.
        x : `numpy.ndarray`
            The starting value of `x` - usually this is the original input whose prediction has to be explained..
        optim : instance of :class:`tf.train.Optimizer`
            Optimizer for minimizing the loss.
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
        
        Raises
        ------
        TypeError
            If the type of `loss` or `model` is not correct.
        """
        if not callable(model):
            raise TypeError("model must be callable")
        if not callable(loss):
            raise TypeError("loss must be callable")
        
        self.model = model
        self.loss = loss
        self.tol = 0.0 if tol is None else tol
        self.max_iter = max_iter

        self.x = create_mutable_tensor(np.copy(x))

        self.optim = optim

        if grad_mask is not None:
            self.grad_mask = create_tensor(grad_mask)
        else:
            self.grad_mask = create_tensor(np.ones_like(x))
    
    def is_grad_based(self):
        return True

    def __loss_grad(self):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.x)
            loss_value = self.loss(self.x)
            
            return loss_value, tape.gradient(loss_value, self.x)
        
    def __process_gradient(self, grad):
        return self.grad_mask * grad

    def minimize(self):
        old_loss = float("inf")
        for _ in range(self.max_iter):
            loss, grad = self.__loss_grad()

            self.optim.apply_gradients([(self.__process_gradient(grad), self.x)])

            if np.abs(old_loss - loss.numpy()) <= self.tol:
                break
            old_loss = loss.numpy()
        
        return self.x.numpy()

    def __call__(self):
        return self.minimize()


def prepare_optim(optimizer, loss, loss_npy, loss_grad_npy, x_orig, model, tol, max_iter, grad_mask):
    if isinstance(optimizer, str) or isinstance(optimizer, Optimizer):
        return desc_to_optim_scipy(optimizer, loss_npy, x_orig, loss_grad_npy, tol, max_iter)
    elif isinstance(optimizer, tf.compat.v1.train.Optimizer):
        optim = TfOptimizer()
        optim.init(model, loss, x_orig, optimizer, tol, max_iter, grad_mask)
        return optim
    else:
        raise TypeError("Invalid type of argument 'optimizer'.\n'optimizer' has to be a string or the name of a class that is derived from tf.train.Optimizer")
