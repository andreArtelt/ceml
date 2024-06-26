Welcome to CEML's documentation!
================================

Counterfactuals for Explaining Machine Learning models - CEML
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CEML is a Python toolbox for computing counterfactuals. Counterfactuals can be used to explain the predictions of machine learing models.

It supports many common machine learning frameworks:

    - scikit-learn (1.5.0)
    - PyTorch (2.0.1)
    - Keras & Tensorflow (2.13.1)

Furthermore, CEML is easy to use and can be extended very easily. See the following user guide for more information on how to use and extend ceml.


.. toctree::
    :maxdepth: 2
    :caption: User Guide

    installation
    tutorial
    advanced
    theory_background
    faq


.. toctree::
    :maxdepth: 2
    :caption: API Reference

    ceml
    ceml.sklearn
    ceml.tfkeras
    ceml.torch
    ceml.costfunctions
    ceml.model
    ceml.optim
    ceml.backend.jax
    ceml.backend.torch
    ceml.backend.tensorflow


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
