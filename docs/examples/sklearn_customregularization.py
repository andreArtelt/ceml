#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import jax.numpy as npx
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from ceml.sklearn import generate_counterfactual
from ceml.backend.jax.costfunctions import CostFunctionDifferentiableJax


# Custom implementation of the l2-regularization. Note that this regularization is differentiable
class MyRegularization(CostFunctionDifferentiableJax):
    def __init__(self, x_orig):
        self.x_orig = x_orig

        super(MyRegularization, self).__init__()
    
    def score_impl(self, x):
        return npx.sum(npx.square(x - self.x_orig)) # Note: This expression must be written in jax and it must be differentiable!


if __name__ == "__main__":
    # Load data
    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Whitelist of features - list of features we can change/use when computing a counterfactual 
    features_whitelist = None   # All features can be used.

    # Create and fit the model
    model = GaussianNB()   # Note that ceml requires: multi_class='multinomial'
    model.fit(X_train, y_train)
    
    # Select data point for explaining its prediction
    x = X_test[1,:]
    print("Prediction on x: {0}".format(model.predict([x])))

    # Create custom regularization function
    regularization = MyRegularization(x)

    # Compute counterfactual
    print("\nCompute counterfactual ....")
    print(generate_counterfactual(model, x, y_target=0, features_whitelist=features_whitelist, regularization=regularization, optimizer="bfgs"))