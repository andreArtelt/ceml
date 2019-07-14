# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'..')

import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, Normalizer, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from ceml.sklearn import generate_counterfactual


def compute_counterfactuals(model, x, y):
    features_whitelist = None
    
    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="bfgs", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == y

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization=None, optimizer="bfgs", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization=None, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y


    features_whitelist = [1, 2]
    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="bfgs", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])

    features_whitelist = [0, 1, 2]
    x_cf, y_cf, delta = generate_counterfactual(model, x, 0, features_whitelist=features_whitelist, regularization=None, optimizer="bfgs", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x, 0, features_whitelist=features_whitelist, regularization=None, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])


def compute_counterfactuals_poly(model, x, y):
    features_whitelist = None

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="bfgs", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization=None, optimizer="bfgs", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization=None, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y


    features_whitelist = [0, 1, 2]
    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="bfgs", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization=None, optimizer="bfgs", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x, 0, features_whitelist=features_whitelist, regularization=None, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])


def compute_counterfactuals_2(model, x, y):
    features_whitelist = None

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization=None, optimizer="bfgs", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization=None, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization=None, optimizer="powell", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l1", C=0.001, optimizer="bfgs", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l1", C=0.001, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l1", C=0.001, optimizer="powell", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y


    features_whitelist = [0, 1, 2]
    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization=None, optimizer="bfgs", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization=None, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization=None, optimizer="powell", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l2", C=0.001, optimizer="bfgs", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l2", C=0.001, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x, y, features_whitelist=features_whitelist, regularization="l2", C=0.001, optimizer="powell", return_as_dict=False)
    assert y_cf == y
    assert model.predict(np.array([x_cf])) == y
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x.shape[0])])


def test_pipeline_scaler_softmaxregression():
    # Load data
    X, y = load_iris(True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    scaler = StandardScaler()

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model = make_pipeline(scaler, model)
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    assert model.predict([x_orig]) == 2

    # Compute counterfactual
    compute_counterfactuals(model, x_orig, 0)


def test_pipeline_robustscaler_softmaxregression():
    # Load data
    X, y = load_iris(True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    scaler = RobustScaler()

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model = make_pipeline(scaler, model)
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    assert model.predict([x_orig]) == 2

    # Compute counterfactual
    compute_counterfactuals(model, x_orig, 0)


def test_pipeline_maxabsscaler_softmaxregression():
    # Load data
    X, y = load_iris(True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    scaler = MaxAbsScaler()

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model = make_pipeline(scaler, model)
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    assert model.predict([x_orig]) == 2

    # Compute counterfactual
    compute_counterfactuals_2(model, x_orig, 0)


def test_pipeline_minmaxscaler_softmaxregression():
    # Load data
    X, y = load_iris(True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    scaler = MinMaxScaler()

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model = make_pipeline(scaler, model)
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    assert model.predict([x_orig]) == 2

    # Compute counterfactual
    compute_counterfactuals_2(model, x_orig, 0)


def test_pipeline_normalizer_softmaxregression():
    # Load data
    X, y = load_iris(True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    scaler = Normalizer()

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model = make_pipeline(scaler, model)
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    assert model.predict([x_orig]) == 2

    # Compute counterfactual
    compute_counterfactuals_2(model, x_orig, 0)


def test_pipeline_poly_softmaxregression():
    # Load data
    X, y = load_iris(True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    poly = PolynomialFeatures(degree=2)

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model = make_pipeline(poly, model)
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    assert model.predict([x_orig]) == 2

    # Compute counterfactual
    compute_counterfactuals_poly(model, x_orig, 0)


def test_pipeline_scaler_poly_softmaxregression():
    # Load data
    X, y = load_iris(True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    poly = PolynomialFeatures(degree=2)
    scaler = StandardScaler()

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model = make_pipeline(poly, scaler, model)
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    assert model.predict([x_orig]) == 2

    # Compute counterfactual
    compute_counterfactuals_poly(model, x_orig, 0)