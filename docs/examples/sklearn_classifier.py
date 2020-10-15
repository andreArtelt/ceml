#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from ceml.sklearn import generate_counterfactual


if __name__ == "__main__":
    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Whitelist of features - list of features we can change/use when computing a counterfactual 
    features_whitelist = None   # We can use all features

    # Create and fit model
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x = X_test[1,:]
    print("Prediction on x: {0}".format(model.predict([x])))

    # Compute counterfactual
    print("\nCompute counterfactual ....")
    print(generate_counterfactual(model, x, y_target=0, features_whitelist=features_whitelist))