# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'..')

import tensorflow as tf
tf.random.set_seed(42)

import numpy as np
np.random.seed(42)
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from ceml.tfkeras import generate_counterfactual
from ceml.backend.tensorflow.costfunctions import SquaredError
from ceml.model import ModelWithLoss


def test_linearregression():
    # Neural network - Linear regression
    class Model(ModelWithLoss):
        def __init__(self, input_size):
            super(Model, self).__init__()

            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(1, input_shape=(input_size,), kernel_regularizer=tf.keras.regularizers.l2(0.0001))
            ])
        
        def fit(self, x_train, y_train, num_epochs=800):
            self.model.compile(optimizer='adam', loss='mse')

            self.model.fit(x_train, y_train, epochs=num_epochs, verbose=False)

        def predict(self, x):
            return self.model(x)
        
        def __call__(self, x):
            return self.predict(x)

        def get_loss(self, y_target, pred=None):
            return SquaredError(input_to_output=self.model.predict, y_target=y_target)

    # Load data
    X, y = load_boston(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # Create and fit model
    model = Model(X.shape[1])
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    assert r2_score(y_test, y_pred) >= 0.6

    # Select data point for explaining its prediction
    x_orig = X_test[3,:]
    y_orig_pred = model.predict(np.array([x_orig]))
    assert y_orig_pred >= 16. and y_orig_pred < 22.

    # Compute counterfactual
    features_whitelist = None
    y_target = 30.
    y_target_done = lambda z: np.abs(z - y_target) < 6.

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    optimizer_args = {"max_iter": 1000}
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=y_target, features_whitelist=features_whitelist, regularization="l2", C=10., optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False, done=y_target_done)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))

    optimizer = "bfgs"
    optimizer_args = {"max_iter": 1000}
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=y_target, features_whitelist=features_whitelist, regularization="l2", C=10., optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False, done=y_target_done)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))

    optimizer = "nelder-mead"
    optimizer_args = {"max_iter": 1000}
    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=y_target, features_whitelist=features_whitelist, regularization="l2", C=[0.1, 1.0, 10., 20.], optimizer=optimizer, optimizer_args=optimizer_args, return_as_dict=False, done=y_target_done)
    assert y_target_done(y_cf)
    assert y_target_done(model.predict(np.array([x_cf])))