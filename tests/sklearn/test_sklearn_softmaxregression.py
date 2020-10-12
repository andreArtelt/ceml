# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'..')

import numpy as np
np.random.seed(42)
import random
random.seed(424242)
from sklearn.utils import shuffle
import pytest
import sklearn
from sklearn.datasets import load_iris
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

from ceml.sklearn import generate_counterfactual
from ceml.sklearn.softmaxregression import softmaxregression_generate_counterfactual
from ceml.sklearn.plausibility import prepare_computation_of_plausible_counterfactuals
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
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
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
    x_cf_plausible = softmaxregression_generate_counterfactual(model, x_orig_orig, y_target, plausibility=plausibility_stuff)
    print("Predictec label of plausible countrefactual: {0}".format(model.predict([x_cf_plausible])))
    assert model.predict([x_cf_plausible]) == y_target

    # Compute closest counterfactual     
    plausibility_stuff["use_density_constraints"] = False   
    x_cf = softmaxregression_generate_counterfactual(model, x_orig_orig, y_target, plausibility=plausibility_stuff)
    assert model.predict([x_cf]) == y_target


def test_softmaxregression():
    # Load data
    X, y = load_iris(return_X_y=True)

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

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", C=0.1, optimizer="bfgs", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", C=0.1, optimizer=MyOptimizer(), return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization=regularization_mad, C=0.1, optimizer="bfgs", return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, 0, features_whitelist=features_whitelist, regularization="l1", C=0.1, optimizer="cg", return_as_dict=False)
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
    X, y = load_iris(return_X_y=True)
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