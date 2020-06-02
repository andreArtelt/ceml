Plausible counterfactuals
=========================

In `Convex Density Constraints for Computing Plausible Counterfactual Explanations (Artelt et al. 2020) <https://github.com/andreArtelt/ConvexDensityConstraintsForPlausibleCounterfactuals>`_ a general framework for computing plausible counterfactuals was proposed.
CEML currently implements these methods for softmax regression and decision tree classifiers.

In order to compute plausible counterfactual explanations, some preparations are required:

Use the :func:`ceml.sklearn.plausibility.prepare_computation_of_plausible_counterfactuals` function for creating a dictionary that can be passed to functions for generating counterfactuals.
You have to provide class dependent fitted Gaussian Mixture Models (GMMs) and the training data itself. In addition, you can also provide an affine preprocessing and a requested density/plausibility threshold (if you do not specify any, a suitable threshold will be selected automatically).

A complete example for computing a plausible counterfactual of a digit classifier (logistic regression) is given below:

.. literalinclude:: examples/sklearn_plausible_counterfactual.py
    :linenos:
