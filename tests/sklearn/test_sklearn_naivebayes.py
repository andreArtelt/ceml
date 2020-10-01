# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'..')

import numpy as np
np.random.seed(42)
import pytest
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from ceml.sklearn import generate_counterfactual


def test_gaussiannaivebayes():
    # Load data
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    assert model.predict([x_orig]) == 2

    # Compute counterfactual
    features_whitelist = None

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", optimizer="mp", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    
    cf = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", optimizer="mp", return_as_dict=True)
    assert cf["y_cf"] == 0
    assert model.predict(np.array([cf["x_cf"]])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l2", optimizer="mp", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="bfgs", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization=None, optimizer="bfgs", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization=None, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0


    features_whitelist = [0, 1, 2]
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", optimizer="mp", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] <= 1e-4 for i in range(x_orig.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l2", optimizer="mp", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] <= 1e-5 for i in range(x_orig.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="bfgs", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization=None, optimizer="bfgs", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization=None, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    # Test binary case
    X, y = load_iris(return_X_y=True)
    idx = y != 2
    X, y = X[idx, :], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    model = GaussianNB()
    model.fit(X_train, y_train)

    x_orig = X_test[1:4][0,:]
    assert model.predict([x_orig]) == 0

    features_whitelist = None

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=1, features_whitelist=features_whitelist, optimizer="mp", return_as_dict=False)
    assert y_cf == 1
    assert model.predict(np.array([x_cf])) == 1

    cf = generate_counterfactual(model, x_orig, y_target=1, features_whitelist=features_whitelist, optimizer="mp", return_as_dict=True)
    assert cf["y_cf"] == 1
    assert model.predict(np.array([cf["x_cf"]])) == 1

    x_orig = X_test[0,:]
    print(model.predict_proba(np.array([x_orig])))
    assert model.predict([x_orig]) == 1

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, optimizer="mp", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    # Other stuff
    from ceml.sklearn import GaussianNbCounterfactual
    with pytest.raises(TypeError):
        GaussianNbCounterfactual(sklearn.linear_model.LogisticRegression())