# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'..')

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.random.set_random_seed(42)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ceml.tfkeras import generate_counterfactual
from ceml.backend.tensorflow.costfunctions import NegLogLikelihoodCost
from ceml.model import ModelWithLoss


def test_softmaxregression():
    # Neural network - Softmax regression
    class Model(ModelWithLoss):
        def __init__(self, input_size, num_classes):
            super(Model, self).__init__()

            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(num_classes, activation='softmax', input_shape=(input_size,))
            ])
        
        def fit(self, x_train, y_train, num_epochs=800):
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            self.model.fit(X_train, y_train, epochs=num_epochs, verbose=False)

        def predict(self, x):
            return np.argmax(self.model(x), axis=1)
        
        def predict_proba(self, x):
            return self.model(x)
        
        def __call__(self, x):
            return self.predict(x)

        def get_loss(self, y_target, pred=None):
            return NegLogLikelihoodCost(self.model.predict_proba, y_target)

    # Load data
    X, y = load_iris(True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # Create and fit model
    model = Model(4, 3)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    assert accuracy_score(y_test, y_pred) >= 0.8

    # Select data point for explaining its prediction
    x_orig = X_test[1,:]
    assert model.predict(np.array([x_orig])) == 1

    # Compute counterfactual
    features_whitelist = None

    optimizer = "bfgs"
    optimizer_args = {"max_iter": 1000}
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l1", C=0.01, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    optimizer = "nelder-mead"
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l1", C=0.01, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l1", C=0.01, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    optimizer = "bfgs"
    optimizer_args = {"max_iter": 1000}
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization=None, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization=None, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0


    features_whitelist = [0, 2]

    optimizer = "bfgs"
    optimizer_args = {"max_iter": 1000}
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l1", C=0.01, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    optimizer = "nelder-mead"
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l1", C=0.01, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l1", C=0.01, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    optimizer = "bfgs"
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization=None, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization=None, optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False)
    assert y_cf == 0
    assert model.predict(np.array([x_cf])) == 0
    assert all([True if i in features_whitelist else delta[i] == 0. for i in range(x_orig.shape[0])])