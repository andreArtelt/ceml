# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'..')

import numpy as np
np.random.seed(42)
import sklearn
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ceml.sklearn import generate_counterfactual


def test_randomforest_classifier():
    # Load data
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    assert model.predict([x_orig]) == 2

    # Compute counterfactual
    features_whitelist = None

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", C=0.01, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization=None, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    features_whitelist = [0, 2]

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization=None, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])


def test_randomforest_regressor():
    # Load data
    X, y = load_boston(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    y_orig_pred = model.predict([x_orig])
    assert y_orig_pred >= 19. and y_orig_pred < 21.

    # Compute counterfactual
    y_target = 25.
    y_target_done = lambda z: np.abs(z - y_target) < 1.

    features_whitelist = None

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target_done, features_whitelist=features_whitelist, regularization="l1", C=1.0, return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target_done, features_whitelist=features_whitelist, regularization=None, return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))

    features_whitelist = [0, 2, 4, 5, 7, 8, 9, 12]
    
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target_done, features_whitelist=features_whitelist, regularization="l1", C=1.0, return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target_done, features_whitelist=features_whitelist, regularization=None, return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])