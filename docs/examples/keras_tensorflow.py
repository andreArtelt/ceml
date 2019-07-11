#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ceml.tfkeras import generate_counterfactual
from ceml.backend.tensorflow.costfunctions import NegLogLikelihoodCost
from ceml.model import ModelWithLoss


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


if __name__ == "__main__":
    # IMPORTANT: Enable eager execution
    tf.compat.v1.enable_eager_execution()

    # Load data
    X, y = load_iris(True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # Create and fit model
    model = Model(4, 3)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("Accuracy: {0}".format(accuracy_score(y_test, y_pred)))

    # Select a data point whose prediction has to be explained
    x_orig = X_test[1,:]
    print("Prediction on x: {0}".format(model.predict(np.array([x_orig]))))

    # Whitelist of features we can use/change when computing the counterfactual
    features_whitelist = None

    # Compute counterfactual
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)    # Init optimization algorithm
    optimizer_args = {"max_iter": 1000}

    print("\nCompute counterfactual ....") 
    print(generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l1", C=0.01, optimizer=optimizer, optimizer_args=optimizer_args))
