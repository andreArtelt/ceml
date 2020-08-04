# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'..')

import numpy as np
np.random.seed(42)
import pytest
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

from ceml.sklearn import generate_counterfactual


def test_linearregression():
    # Load data
    X, y = load_boston(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    model = Lasso()
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    y_orig_pred = model.predict([x_orig])
    assert y_orig_pred >= 19. and y_orig_pred < 20.

    # Compute counterfactual
    y_target = 25.
    y_target_done = lambda z: np.abs(z - y_target) < 1.

    features_whitelist = None

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target, done=y_target_done, features_whitelist=features_whitelist, regularization="l1", optimizer="mp", return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))

    cf = generate_counterfactual(model, x_orig, y_target, done=y_target_done, features_whitelist=features_whitelist, regularization="l1", optimizer="mp", return_as_dict=True)
    assert y_target_done(cf["y_cf"])
    assert y_target_done(model.predict(np.array([cf["x_cf"]])))

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target, done=y_target_done, features_whitelist=features_whitelist, regularization="l2", optimizer="mp", return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target, done=y_target_done, features_whitelist=features_whitelist, regularization="l1", C=[1.e10, 1.0], optimizer="bfgs", return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target, done=y_target_done, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target, done=y_target_done, features_whitelist=features_whitelist, regularization=None, optimizer="bfgs", return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target, done=y_target_done, features_whitelist=features_whitelist, regularization=None, optimizer="nelder-mead", return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))


    features_whitelist = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12]
    #x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target, done=y_target_done, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="bfgs", return_as_dict=False)
    #assert y_target_done(y_cf)
    #assert y_target_done(model.predict(np.array([x_cf])))
    #assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target, done=y_target_done, features_whitelist=features_whitelist, regularization="l1", optimizer="mp", return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))
    assert all([True if i in features_whitelist else delta[i] <= 1e-5 for i in range(x_orig.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target, done=y_target_done, features_whitelist=features_whitelist, regularization="l2", optimizer="mp", return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))
    assert all([True if i in features_whitelist else delta[i] <= 1e-2 for i in range(x_orig.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target, done=y_target_done, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    #x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target, done=y_target_done, features_whitelist=features_whitelist, regularization=None, optimizer="bfgs", return_as_dict=False)
    #assert y_target_done(y_cf)
    #assert y_target_done(model.predict(np.array([x_cf])))
    #assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target, done=y_target_done, features_whitelist=features_whitelist, regularization=None, optimizer="nelder-mead", return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    # Other stuff
    from ceml.sklearn import LinearRegressionCounterfactual
    with pytest.raises(TypeError):
        LinearRegressionCounterfactual(sklearn.naive_bayes.GaussianNB())