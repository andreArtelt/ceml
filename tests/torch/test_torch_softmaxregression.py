# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'..')

import torch
torch.manual_seed(42)
import numpy as np
np.random.seed(42)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize

from ceml.torch import generate_counterfactual
from ceml.backend.torch.costfunctions import NegLogLikelihoodCost, LMadCost
from ceml.model import ModelWithLoss
from ceml.optim import Optimizer


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
    # Neural network - Softmax regression
    class Model(torch.nn.Module, ModelWithLoss):
        def __init__(self, input_size, num_classes):
            super(Model, self).__init__()

            self.linear = torch.nn.Linear(input_size, num_classes)
            self.softmax = torch.nn.Softmax(dim=0)
        
        def forward(self, x):
            return self.linear(x)   # NOTE: Softmax is build into CrossEntropyLoss
        
        def predict_proba(self, x):
            return self.softmax(self.forward(x))
        
        def predict(self, x, dim=1):
            return torch.argmax(self.forward(x), dim=dim)
        
        def get_loss(self, y_target, pred=None):
            return NegLogLikelihoodCost(input_to_output=self.predict_proba, y_target=y_target)

    # Load data
    X, y = load_iris(return_X_y=True)
    X = X.astype(np.dtype(np.float32))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # numpy -> torch tensor
    x = torch.from_numpy(X_train)
    labels = torch.from_numpy(y_train)

    x_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    # Create and fit model
    model = Model(4, 3)

    learning_rate = 0.001
    momentum = 0.9
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  

    num_epochs = 800
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    y_pred = model.predict(x_test).detach().numpy()
    assert accuracy_score(y_test, y_pred) >= 0.8

    # Select data point for explaining its prediction
    x_orig = X_test[1,:]
    assert model.predict(torch.from_numpy(np.array([x_orig]))).detach().numpy() == 1

    # Create weighted manhattan distance cost function
    md = np.median(X_train, axis=0)
    mad = np.median(np.abs(X_train - md), axis=0)
    regularization_mad = LMadCost(torch.from_numpy(x_orig), torch.from_numpy(mad))

    # Compute counterfactual
    features_whitelist = None

    optimizer = "bfgs"
    optimizer_args = {"max_iter": 1000, "args": {"lr": 0.9, "momentum": 0.9}}
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l2", C=0.001, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))).detach().numpy() == 0

    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l2", C=0.001, optimizer=MyOptimizer(), optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))).detach().numpy() == 0

    optimizer = "bfgs"
    optimizer_args = {"max_iter": 1000, "args": {"lr": 0.9, "momentum": 0.9}}
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization=regularization_mad, C=0.001, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))).detach().numpy() == 0

    optimizer = "nelder-mead"
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l1", C=0.001, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))).detach().numpy() == 0

    optimizer = torch.optim.SGD
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l2", C=0.001, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))).detach().numpy() == 0

    optimizer = "bfgs"
    optimizer_args = {"max_iter": 1000, "args": {"lr": 0.9, "momentum": 0.9}}
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization=None, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))).detach().numpy() == 0

    optimizer = torch.optim.SGD
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization=None, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))).detach().numpy() == 0


    features_whitelist = [0, 2]

    optimizer = "bfgs"
    optimizer_args = {"max_iter": 1000, "args": {"lr": 0.9, "momentum": 0.9}}
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l1", C=0.001, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))).detach().numpy() == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    optimizer = "nelder-mead"
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l2", C=0.001, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))).detach().numpy() == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    optimizer = torch.optim.SGD
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l2", C=0.0001, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))).detach().numpy() == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    optimizer = "bfgs"
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization=None, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))).detach().numpy() == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    optimizer = torch.optim.SGD
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization=None, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))).detach().numpy() == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])
