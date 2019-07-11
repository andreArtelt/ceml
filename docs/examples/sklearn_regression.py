#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge

from ceml.sklearn import generate_counterfactual


if __name__ == "__main__":
    # Load data
    X, y = load_boston(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Whitelist of features - list of features we can change/use when computing a counterfactual 
    features_whitelist = [0, 1, 2, 3, 4]    # Use the first five features only

    # Create and fit model
    model = Ridge()
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x = X_test[1,:]
    print("Prediction on x: {0}".format(model.predict([x])))

    # Compute counterfactual
    print("\nCompute counterfactual ....")
    y_target = 25.0
    done = lambda z: np.abs(y_target - z) <= 0.5     # Since we might not be able to achieve `y_target` exactly, we tell ceml that we are happy if we do not deviate more than 0.5 from it.
    print(generate_counterfactual(model, x, y_target=y_target, features_whitelist=features_whitelist, C=1.0, regularization="l2", optimizer="bfgs", done=done))