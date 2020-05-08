Tensorflow & Keras
==================

Since keras is a higher-lever interface for tensorflow and nowadays part of tensorflow , we do not need to distinguish between keras and tensorflow models when using ceml.

Computing a counterfactual of a tensorflow/keras model is done by using the :func:`ceml.tfkeras.counterfactual.generate_counterfactual` function.

.. note::
    We have to run in *eager execution mode* when computing a counterfactual! Since tensorflow 2, eager execution is enabled by default.

We must provide the tensorflow/keras model within a class that is derived from the :class:`ceml.model.model.ModelWithLoss` class.
In this class, we must overwrite the `predict` function and `get_loss` function which returns a loss that we want to use - a couple of differentiable loss functions are implemented in :class:`ceml.backend.tensorflow.costfunctions`.

Besides the model, we must specify the input whose prediction we want to explain and the desired target prediction (prediction of the counterfactual).
In addition we can restrict the features that can be used for computing a counterfactual, specify a regularization of the counterfactual and specifying the optimization algorithm used for computing a counterfactual.

A complete example of a softmax regression model using the negative-log-likelihood is given below:

.. literalinclude:: examples/keras_tensorflow.py
    :linenos: