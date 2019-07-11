#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ceml.torch import generate_counterfactual
from ceml.backend.torch.costfunctions import NegLogLikelihoodCost
from ceml.model import ModelWithLoss


# Neural network - Softmax regression
class Model(torch.nn.Module, ModelWithLoss):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()

        self.linear = torch.nn.Linear(input_size, num_classes)
        self.softmax = torch.nn.Softmax(dim=0)
    
    def forward(self, x):
        return self.linear(x)   # NOTE: Softmax is build into CrossEntropyLoss
    
    def predict_proba(self, x):
        return self.softmax(self.forward(x))
    
    def predict(self, x, dim=1):
        return torch.argmax(self.forward(x), dim=dim)
    
    def get_loss(self, y_target, pred=None):
        return NegLogLikelihoodCost(self.predict_proba, y_target)


if __name__ == "__main__":
    # Load data
    X, y = load_iris(True)
    X = X.astype(np.dtype(np.float32))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # numpy -> torch tensor
    x = torch.from_numpy(X_train)
    labels = torch.from_numpy(y_train)

    x_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    # Create and fit model
    model = Model(4, 3)

    learning_rate = 0.001
    momentum = 0.9
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  

    num_epochs = 800
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    y_pred = model.predict(x_test).numpy()
    print("Accuracy: {0}".format(accuracy_score(y_test, y_pred)))

    # Select a data point whose prediction has to be explained
    x_orig = X_test[1,:]
    print("Prediction on x: {0}".format(model.predict(torch.from_numpy(np.array([x_orig])))))

    # Whitelist of features we can use/change when computing the counterfactual
    features_whitelist = [0, 2] # Use the first and third feature only

    # Compute counterfactual
    print("\nCompute counterfactual ....")
    print(generate_counterfactual(model, x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l1", C=0.1, optimizer="nelder-mead"))