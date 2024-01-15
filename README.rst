****
CEML
****
--------------------------------------------------------
Counterfactuals for Explaining Machine Learning models
--------------------------------------------------------

CEML is a Python toolbox for computing counterfactuals. Counterfactuals can be used to explain the predictions of machine learing models.

It supports many common machine learning frameworks:

    - scikit-learn (1.3.1)
    - PyTorch (2.0.1)
    - Keras & Tensorflow (2.13.1)

Furthermore, CEML is easy to use and can be extended very easily. See the following user guide for more information on how to use and extend CEML.

Installation
------------

**Note: Python 3.8 is required!**

Tested on Ubuntu -- note that some people reported problems with some dependencies on Windows!

PyPI
++++

.. code-block:: bash

    pip install ceml

**Note**: The package hosted on PyPI uses the cpu only. If you want to use the gpu, you have to install CEML manually - see next section.

Git
+++
Download or clone the repository:

.. code:: bash

    git clone https://github.com/andreArtelt/ceml.git
    cd ceml

Install all requirements (listed in ``requirements.txt``):

.. code:: bash

    pip install -r requirements.txt

**Note**: If you want to use a gpu/tpu, you have to install the gpu version of jax, tensorflow and pytorch manually. Do not use ``pip install -r requirements.txt``.

Install the toolbox itself:

.. code:: bash

    pip install .


Quick example
-------------

.. code-block:: python

    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    from ceml.sklearn import generate_counterfactual


    if __name__ == "__main__":
        # Load data
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

        # Whitelist of features - list of features we can change/use when computing a counterfactual 
        features_whitelist = None   # We can use all features

        # Create and fit model
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)

        # Select data point for explaining its prediction
        x = X_test[1,:]
        print("Prediction on x: {0}".format(model.predict([x])))

        # Compute counterfactual
        print("\nCompute counterfactual ....")
        print(generate_counterfactual(model, x, y_target=0, features_whitelist=features_whitelist))

Documentation
-------------

Documentation is available on readthedocs:`https://ceml.readthedocs.io/en/latest/ <https://ceml.readthedocs.io/en/latest/>`_

License
-------

MIT license - See `LICENSE <LICENSE>`_

How to cite?
------------
    You can cite CEML by using the following BibTeX entry:

    .. code-block::

        @misc{ceml,
                author = {André Artelt},
                title = {CEML: Counterfactuals for Explaining Machine Learning models - A Python toolbox},
                year = {2019 - 2023},
                publisher = {GitHub},
                journal = {GitHub repository},
                howpublished = {\url{https://www.github.com/andreArtelt/ceml}}
            }


Third party components
----------------------

    - `numpy <https://github.com/numpy/numpy>`_
    - `scipy <https://github.com/scipy/scipy>`_
    - `jax <https://github.com/google/jax>`_
    - `cvxpy <https://github.com/cvxgrp/cvxpy>`_
    - `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_
    - `sklearn-lvq <https://github.com/MrNuggelz/sklearn-lvq>`_
    - `PyTorch <https://github.com/pytorch/pytorch>`_
    - `tensorflow <https://github.com/tensorflow>`_
