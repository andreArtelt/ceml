#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
random.seed(424242)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from ceml.sklearn.softmaxregression import softmaxregression_generate_counterfactual
from ceml.sklearn.plausibility import prepare_computation_of_plausible_counterfactuals


if __name__ == "__main__":
    # Load data set
    X, y = load_digits(return_X_y=True);pca_dim=40

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

    pca = PCA(n_components=pca_dim)
    pca.fit(X_train)

    projection_matrix = pca.components_ # Projection matrix
    projection_mean_sub = pca.mean_

    X_train = np.dot(X_train - projection_mean_sub, projection_matrix.T)
    X_test = np.dot(X_test - projection_mean_sub, projection_matrix.T)

    # Fit classifier
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
    model.fit(X_train, y_train)

    # Compute accuracy on test set
    print("Accuracy: {0}".format(accuracy_score(y_test, model.predict(X_test))))

    # For each class, fit density estimators
    density_estimators = {}
    kernel_density_estimators = {}
    labels = np.unique(y)
    for label in labels:
        # Get all samples with the 'correct' label
        idx = y_train == label
        X_ = X_train[idx, :]

        # Optimize hyperparameters
        cv = GridSearchCV(estimator=GaussianMixture(covariance_type='full'), param_grid={'n_components': range(2, 10)}, n_jobs=-1, cv=5)
        cv.fit(X_)
        n_components = cv.best_params_["n_components"]

        # Build density estimators
        de = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        de.fit(X_)

        density_estimators[label] = de

    # Build dictionary for ceml
    plausibility_stuff = prepare_computation_of_plausible_counterfactuals(X_train_orig, y_train, density_estimators, projection_mean_sub, projection_matrix)

    # Compute and plot counterfactual with vs. without density constraints
    i = 0

    x_orig = X_test[i,:]
    x_orig_orig = X_test_orig[i,:]
    y_orig = y_test[i]
    y_target = y_test_target[i]
    print("Original label: {0}".format(y_orig))
    print("Target label: {0}".format(y_target))

    if(model.predict([x_orig]) == y_target):  # Model already predicts target label!
        raise ValueError("Requested prediction already satisfied")

    # Compute plausible counterfactual
    x_cf_plausible = softmaxregression_generate_counterfactual(model, x_orig_orig, y_target, plausibility=plausibility_stuff)
    x_cf_plausible_projected = np.dot(x_cf_plausible - projection_mean_sub, projection_matrix.T)
    print("Predictec label of plausible countrefactual: {0}".format(model.predict([x_cf_plausible_projected])))

    # Compute closest counterfactual     
    plausibility_stuff["use_density_constraints"] = False   
    x_cf = softmaxregression_generate_counterfactual(model, x_orig_orig, y_target, plausibility=plausibility_stuff)
    x_cf_projected = np.dot(x_cf - projection_mean_sub, projection_matrix.T)
    print("Predicted label of closest counterfactual: {0}".format(model.predict([x_cf_projected])))

    # Plot results
    fig, axes = plt.subplots(3, 1)
    axes[0].imshow(x_orig_orig.reshape(8, 8))    # Original sample
    axes[1].imshow(x_cf.reshape(8, 8))           # Closest counterfactual
    axes[2].imshow(x_cf_plausible.reshape(8, 8)) # Plausible counterfactual
    plt.show()