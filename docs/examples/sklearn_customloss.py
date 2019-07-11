#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import jax.numpy as npx
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge


from ceml.sklearn import generate_counterfactual
from ceml.sklearn import LinearRegression, LinearRegressionCounterfactual
from ceml.backend.jax.costfunctions import CostFunctionDifferentiableJax


# Custom implementation of the l2-regularization. Note that this regularization is differentiable.
class MyLoss(CostFunctionDifferentiableJax):
    def __init__(self, input_to_output, y_target):
        self.y_target = y_target

        super(MyLoss, self).__init__(input_to_output)
    
    def score_impl(self, y):
        return npx.abs(y - y_target)**4

# Derive a new class from ceml.sklearn.linearregression.LinearRegression and overwrite the get_loss method to use our custom loss MyLoss
class LinearRegressionWithMyLoss(LinearRegression):
    def __init__(self, model):
        super(LinearRegressionWithMyLoss, self).__init__(model)

    def get_loss(self, y_target, pred=None):
        if pred is None:
            return MyLoss(self.predict, y_target)
        else:
            return MyLoss(pred, y_target)

# Derive a new class from ceml.sklearn.linearregression.LinearRegressionCounterfactual that uses our new linear regression wrapper LinearRegressionWithMyLoss for computing counterfactuals
class MyLinearRegressionCounterfactual(LinearRegressionCounterfactual):
    def __init__(self, model):
        super(MyLinearRegressionCounterfactual, self).__init__(model)

    def rebuild_model(self, model):
        return LinearRegressionWithMyLoss(model)


if __name__ == "__main__":
    # Load data
    X, y = load_boston(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Whitelist of features - list of features we can change/use when computing a counterfactual 
    features_whitelist = None   # All features can be used.

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
    
    cf = MyLinearRegressionCounterfactual(model)    # Since we are using our own loss function, we can no longer use standard method generate_counterfactual 
    print(cf.compute_counterfactual(x, y_target=y_target, features_whitelist=features_whitelist, regularization="l2", C=1.0, optimizer="bfgs", done=done))