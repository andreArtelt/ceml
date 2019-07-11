*********
Advanced
*********

ceml can be easily extended and all major components can be customized to fit the users needs.

Below is a (non-exhaustive) list of some (common) use cases:

Custom regularization
+++++++++++++++++++++

Instead of using one of the predefined regularizations, we can pass a custom regularization to :func:`ceml.sklearn.models.generate_counterfactual`.

All regularization implementations must be classes derived from :class:`ceml.costfunctions.costfunctions.CostFunction`. In case of scikit-learn, if we want to use a gradient based optimization algorithm, we must derive from :class:`ceml.backend.jax.costfunctions.costfunctions.CostFunctionDifferentiableJax` - note that :class:`ceml.backend.jax.costfunctions.costfunctions.CostFunctionDifferentiableJax` is already dervied from :class:`ceml.costfunctions.costfunctions.CostFunction`.

.. note::
    For tensorflow/keras or PyTorch models the base classes are :class:`ceml.backend.tensorflow.costfunctions.costfunctions.CostFunctionDifferentiableTf` and :class:`ceml.backend.torch.costfunctions.costfunctions.CostFunctionDifferentiableTorch`. 

The computation of the regularization itself must be implemented in the `score_impl` function.

A complete example of a re-implementation of the l2-regularization is given below:

.. literalinclude:: examples/sklearn_customregularization.py
    :linenos:


Custom loss function
++++++++++++++++++++

In order to use a custom loss function we have to do three things:

    1. Implement the loss function. This is the same as implementing a custom regularization - a regularization is a loss function that works on the input rather than on the output.
    2. Derive a child class from the model class and overwrite the `get_loss` function to use our custom loss function.
    3. Derive a child class from the counterfactual class of the model and overwrite the `rebuild_model` function to use our model from the previous step.

A complete example of using a custom loss for a linear regression model is given below:

.. literalinclude:: examples/sklearn_customloss.py
    :linenos:

Add a custom model to the sklearn pipeline
++++++++++++++++++++++++++++++++++++++++++

TODO

Add a custom optimizer
++++++++++++++++++++++

TODO