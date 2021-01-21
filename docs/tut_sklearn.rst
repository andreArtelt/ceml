scikit-learn
============

Classification
++++++++++++++

Computing a counterfactual of a sklearn classifier is done by using the :func:`ceml.sklearn.models.generate_counterfactual` function.

We must specify the model we want to use, the input whose prediction we want to explain and the requested target prediction (prediction of the counterfactual).
In addition we can restrict the features that can be used for computing a counterfactual, specify a regularization of the counterfactual and specifying the optimization algorithm used for computing a counterfactual.

A complete example of a classification task is given below:

.. literalinclude:: examples/sklearn_classifier.py
    :linenos:

Regression
+++++++++++

The interface for computing a counterfactual of a regression model is exactly the same.

But because it might be very difficult or even impossible (e.g. knn or decision tree) to achieve a requested prediction exactly, we can specify a tolerance range in which the prediction is accepted.

We can so by defining a function that takes a prediction as an input and returns `True` if the predictions is accepted (it is in the range of tolerated predictions) and `False` otherwise.
For instance, if our target value is `25.0` but we are also happy if it deviates not more than 0.5, we could come up with the following function:

.. code-block:: python
    :linenos:

    done = lambda z: np.abs(z - 25.0) <= 0.5

This function can be passed as a value of the optional argument `done` to the :func:`ceml.sklearn.models.generate_counterfactual` function.

A complete example of a regression task is given below:

.. literalinclude:: examples/sklearn_regression.py
    :linenos:

Pipeline
++++++++

Often our machine learning pipeline contains more than one model. E.g. we first scale the input and/or reduce the dimensionality before classifying it.

The interface for computing a counterfactual when using a pipeline is identical to the one when using a single model only. We can simply pass a :class:`sklearn.pipeline.Pipeline` instance as the value of the parameter `model` to the function :func:`ceml.sklearn.models.generate_counterfactual`.

Take a look at the :class:`ceml.sklearn.pipeline.PipelineCounterfactual` class to see which preprocessings are supported.

A complete example of a classification pipeline with the standard scaler :class:`skelarn.preprocessing.StandardScaler` and logistic regression :class:`sklearn.linear_model.LogisticRegression` is given below:

.. literalinclude:: examples/sklearn_pipeline.py
    :linenos:


Change optimization parameters
++++++++++++++++++++++++++++++

Sometimes it might become necessary to change to default parameters of the optimization methods - e.g. changing the solver, the maximum number of iterations, etc.
This can be done by passing the optional argument `optimizer_args` to the :func:`ceml.sklearn.models.generate_counterfactual` function.
The value of `optimizer_args` must be a dictionary where some parameters like verbosity, solver, maximum number of iterations, tolerance thresholds, etc. can be changed - note that not all parameters are used by every optimization algorithm (e.g. "epsilon", "solver" and "solver_verbosity" are only used if `optimizer="mp"`).

A short code snippet demonstrating how to change some optimization parameters is given below:

.. literalinclude:: examples/sklearn_opt_args.py
    :linenos: