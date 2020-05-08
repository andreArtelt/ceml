# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'..')

import numpy as np
np.random.seed(42)
import pytest
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

from ceml.sklearn import generate_counterfactual
from ceml.optim import Optimizer
from ceml.backend.jax.costfunctions import LMadCost 


# Custom optimization method that simply calls the BFGS optimizer from scipy
class MyOptimizer(Optimizer):
    def __init__(self):
        self.f = None
        self.f_grad = None
        self.x0 = None
        self.tol = None
        self.max_iter = None

        super(MyOptimizer, self).__init__()
    
    def init(self, f, f_grad, x0, tol=None, max_iter=None):
        self.f = f
        self.f_grad = f_grad
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter

    def is_grad_based(self):
        return True
    
    def __call__(self):
        optimum = minimize(fun=self.f, x0=self.x0, jac=self.f_grad, tol=self.tol, options={'maxiter': self.max_iter}, method="BFGS")
        return np.array(optimum["x"])


def test_softmaxregression():
    # Load data
    X, y = load_iris(True)

    # Binary classification problem
    idx = y > 1 # Convert data into a binary problem
    X_, y_ = X[idx,:], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)    # Split data into training and test set

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')   # Create and fit model
    model.fit(X_train, y_train)

    x_orig = X_test[1:4][0,:]   # Select data point for explaining its prediction
    assert model.predict([x_orig]) == 2

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, return_as_dict=False) # Compute counterfactual explanation
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    cf = generate_counterfactual(model, x_orig, 0, return_as_dict=True) # Compute counterfactual explanation
    assert cf["y_cf"]== 0
    assert model.predict(np.array([cf["x_cf"]])) == 0

    # Multiclass classification problem
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    assert model.predict([x_orig]) == 2

    # Create weighted manhattan distance cost function
    md = np.median(X_train, axis=0)
    mad = np.median(np.abs(X_train - md), axis=0)
    regularization_mad = LMadCost(x_orig, mad)

    # Compute counterfactual
    features_whitelist = None

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    cf = generate_counterfactual(model, x_orig, 0, return_as_dict=True)
    assert cf["y_cf"] == 0
    assert model.predict(np.array([cf["x_cf"]])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", optimizer="mp", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l2", optimizer="mp", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="bfgs", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer=MyOptimizer(), return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization=regularization_mad, C=1.0, optimizer="bfgs", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", C=1.0, optimizer="cg", return_as_dict=False)
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


    features_whitelist = [1, 2]
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", optimizer="mp", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] <= 1e-5 for i in range(x_orig.shape[0])])

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

    features_whitelist = [0, 1, 2]
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization=None, optimizer="bfgs", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization=None, optimizer="nelder-mead", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    # Test binary case
    X, y = load_iris(True)
    idx = y != 2
    X, y = X[idx, :], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, y_train)

    x_orig = X_test[1:4][0,:]
    assert model.predict([x_orig]) == 0

    features_whitelist = None

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 1, features_whitelist=features_whitelist, regularization="l1", optimizer="mp", return_as_dict=False)
    assert y_cf == 1
    assert model.predict(np.array([x_cf])) == 1

    x_orig = X_test[0,:]
    assert model.predict([x_orig]) == 1

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", optimizer="mp", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    # Other stuff
    from ceml.sklearn import SoftmaxCounterfactual
    with pytest.raises(TypeError):
        SoftmaxCounterfactual(sklearn.linear_model.LinearRegression())
    
    with pytest.raises(ValueError):
        SoftmaxCounterfactual(LogisticRegression(multi_class="ovr"))