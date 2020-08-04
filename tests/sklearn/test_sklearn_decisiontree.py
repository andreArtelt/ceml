# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'..')

import numpy as np
import random
random.seed(424242)
import sklearn
from sklearn.datasets import load_iris, load_boston
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle

from ceml.sklearn import generate_counterfactual
from ceml.sklearn.decisiontree import decisiontree_generate_counterfactual
from ceml.sklearn.plausibility import prepare_computation_of_plausible_counterfactuals


def test_plausible_counterfactual():
    # Load data
    X, y = load_iris(return_X_y=True)

    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Choose target labels
    y_test_target = []
    labels = np.unique(y)
    for i in range(X_test.shape[0]):
        y_test_target.append(random.choice(list(filter(lambda l: l != y_test[i], labels))))
    y_test_target = np.array(y_test_target)

    # Reduce dimensionality
    X_train_orig = np.copy(X_train)
    X_test_orig = np.copy(X_test)
    projection_matrix = None
    projection_mean_sub = None

    # Fit classifier
    model = DecisionTreeClassifier(max_depth=7, random_state=42)
    model.fit(X_train, y_train)

    # For each class, fit density estimators
    density_estimators = {}
    kernel_density_estimators = {}
    labels = np.unique(y)
    for label in labels:
        # Get all samples with the 'correct' label
        idx = y_train == label
        X_ = X_train[idx, :]

        # Optimize hyperparameters
        cv = GridSearchCV(estimator=KernelDensity(), param_grid={'bandwidth': np.arange(0.1, 10.0, 0.05)}, n_jobs=-1, cv=5)
        cv.fit(X_)
        bandwidth = cv.best_params_["bandwidth"]

        cv = GridSearchCV(estimator=GaussianMixture(covariance_type='full'), param_grid={'n_components': range(2, 10)}, n_jobs=-1, cv=5)
        cv.fit(X_)
        n_components = cv.best_params_["n_components"]

        # Build density estimators
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(X_)

        de = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        de.fit(X_)

        density_estimators[label] = de
        kernel_density_estimators[label] = kde

    # Build dictionary for ceml
    plausibility_stuff = prepare_computation_of_plausible_counterfactuals(X_train_orig, y_train, density_estimators, projection_mean_sub, projection_matrix)

    # Compute and plot counterfactual with vs. without density constraints
    i = 0

    x_orig = X_test[i,:]
    x_orig_orig = X_test_orig[i,:]
    y_orig = y_test[i]
    y_target = y_test_target[i]

    assert model.predict([x_orig]) == y_orig # Model already predicts target label!

    # Compute plausible counterfactual
    x_cf_plausible = decisiontree_generate_counterfactual(model, x_orig_orig, y_target, plausibility=plausibility_stuff)
    print("Predictec label of plausible countrefactual: {0}".format(model.predict([x_cf_plausible])))
    assert model.predict([x_cf_plausible]) == y_target

    # Compute closest counterfactual     
    plausibility_stuff["use_density_constraints"] = False   
    x_cf = decisiontree_generate_counterfactual(model, x_orig_orig, y_target, plausibility=plausibility_stuff)
    assert model.predict([x_cf]) == y_target


def test_decisiontree_classifier():
    # Load data
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    assert model.predict([x_orig]) == 2

    # Compute counterfactual
    features_whitelist = None

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    features_whitelist = [0, 2]
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])


def test_decisiontree_regressor():
    # Load data
    X, y = load_boston(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    y_orig_pred = model.predict([x_orig])
    assert y_orig_pred >= 19. and y_orig_pred < 21.

    # Compute counterfactual
    y_target = 25.
    y_target_done = lambda z: np.abs(z - y_target) < 1.

    features_whitelist = None

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target_done, features_whitelist=features_whitelist, regularization="l1", return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))

    features_whitelist = [0, 2, 4, 5, 7, 8, 9, 12]
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target_done, features_whitelist=features_whitelist, regularization="l1", return_as_dict=False)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])