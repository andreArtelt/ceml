# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'..')

import torch
torch.manual_seed(42)
import numpy as np
np.random.seed(42)
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from ceml.torch import generate_counterfactual
from ceml.backend.torch.costfunctions import SquaredError
from ceml.model import ModelWithLoss


def test_linearregression():
    # Neural network - Linear regression
    class Model(torch.nn.Module, ModelWithLoss):
        def __init__(self, input_size):
            super(Model, self).__init__()

            self.linear = torch.nn.Linear(input_size, 1)
        
        def forward(self, x):
            return self.linear(x)
        
        def predict(self, x, dim=None): # Note: In contrast to classification, the parameter `dim` is not used!
            return self.forward(x)
        
        def get_loss(self, y_target, pred=None):
            return SquaredError(input_to_output=self.predict, y_target=y_target)

    # Load data
    X, y = load_boston(return_X_y=True)
    X = X.astype(np.dtype(np.float32))
    y = y.astype(np.dtype(np.float32))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # numpy -> torch tensor
    x = torch.from_numpy(X_train)
    labels = torch.from_numpy(y_train.reshape(y_train.shape[0], -1))

    x_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    # Create and fit model
    model = Model(X_train.shape[1])

    learning_rate = 0.1
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01) 

    num_epochs = 30000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    y_pred = model.predict(x_test).detach().numpy()
    assert r2_score(y_test, y_pred) >= 0.6

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][1,:]
    y_orig_pred = model.predict(torch.from_numpy(np.array([x_orig], dtype=np.float32)))
    assert y_orig_pred >= 16. and y_orig_pred < 20.

    # Compute counterfactual
    features_whitelist = None
    y_target = 30.
    y_target_done = lambda z: np.abs(z - y_target) < 6.

    optimizer = "bfgs"
    optimizer_args = {"max_iter": 1000, "args": {"lr": 0.01}}
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=y_target, features_whitelist=features_whitelist, regularization="l1", C=35., optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False, done=y_target_done)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))))

    optimizer = "nelder-mead"
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=y_target, features_whitelist=features_whitelist, regularization="l2", C=8., optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False, done=y_target_done)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))))

    optimizer = torch.optim.Adam
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=y_target, features_whitelist=features_whitelist, regularization="l2", C=5., optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False, done=y_target_done)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))))

    features_whitelist = features_whitelist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    optimizer = "bfgs"
    optimizer_args = {"max_iter": 5000, "args": {"lr": 0.01}}
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=y_target, features_whitelist=features_whitelist, regularization="l1", C=33., optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False, done=y_target_done)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))))
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    optimizer = "nelder-mead"
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=y_target, features_whitelist=features_whitelist, regularization="l2", C=6., optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False, done=y_target_done)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(torch.from_numpy(np.array([x_cf], dtype=np.float32))))
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])
