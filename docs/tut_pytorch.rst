PyTorch
=======

Computing a counterfactual of a PyTorch model is done by using the :func:`ceml.torch.counterfactual.generate_counterfactual` function.

We must provide the PyTorch model within a class that is derived from :class:`torch.nn.Module` and :class:`ceml.model.model.ModelWithLoss`.
In this class, we must overwrite the `predict` function and the `get_loss` function which returns a loss that we want to use - a couple of differentiable loss functions are implemented in :class:`ceml.backend.torch.costfunctions`.

Besides the model, we must specify the input whose prediction we want to explain and the desired target prediction (prediction of the counterfactual).
In addition we can restrict the features that can be used for computing a counterfactual, specify a regularization of the counterfactual and specifying the optimization algorithm used for computing a counterfactual.

A complete example of a softmax regression model using the negative-log-likelihood is given below:

.. literalinclude:: examples/torch.py
    :linenos: