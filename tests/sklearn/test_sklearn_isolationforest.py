# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(42)
import pytest
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

from ceml.sklearn import generate_counterfactual


def test_isolationforest():
    # Load data
    X, _ = make_blobs(n_samples=400, centers=[[0, 0], [0, 0]], cluster_std=0.5, n_features=2, random_state=42)
    X_outlier = np.random.RandomState(42).uniform(low=-6, high=6, size=(50, 2))

    # Create and fit model
    model = IsolationForest(random_state=42)
    model.fit(X)

    # Compute counterfactuals
    x = X[0,:]
    y_target = -1
    assert model.predict([x]) == 1

    x_cf, y_cf, _ = generate_counterfactual(model, x, y_target=y_target, return_as_dict=False)
    assert y_cf == y_target
    assert model.predict(np.array([x_cf])) == y_target

    x = X_outlier[1,:]
    y_target = 1
    assert model.predict([x]) == -1

    x_cf, y_cf, _ = generate_counterfactual(model, x, y_target=y_target, return_as_dict=False)
    assert y_cf == y_target
    assert model.predict(np.array([x_cf])) == y_target

    cf = generate_counterfactual(model, x, y_target=y_target, return_as_dict=True)
    assert cf["y_cf"] == y_target
    assert model.predict(np.array([cf["x_cf"]])) == y_target

    # Other stuff
    from ceml.sklearn import IsolationForest as IsolationForestCf
    model_cf = IsolationForestCf(model)
    assert model.predict([x]) == model_cf.predict(x)

    with pytest.raises(TypeError):
        IsolationForestCf(sklearn.linear_model.LogisticRegression())