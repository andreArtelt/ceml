# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class Model(ABC):
    """Base class of a model.

    Note
    ----
    The class :class:`Model` can not be instantiated because it contains an abstract method. 
    """
    def __init__(self):
        super().__init__()
    
    def __call__(self, x):
        return self.predict(x)
    
    @abstractmethod
    def predict(self, x):
        """Predict the output of a given input.

        Abstract method for computing a prediction.

        Note
        ----
        All derived classes must implement this method.
        """
        raise NotImplementedError()


class ModelWithLoss(Model):
    """Base class of a model that comes with its own loss function.

    Note
    ----
    The class :class:`ModelWithLoss` can not be instantiated because it contains an abstract method. 
    """
    def __init__(self):
        super(ModelWithLoss, self).__init__()

    @abstractmethod
    def get_loss(self, y_target, pred=None):
        """Creates and returns a loss function.

        Builds a cost function where the target is `y_target`.

        Returns
        -------
        :class:`ceml.costfunctions.costfunction.Cost`
            The cost function.

        Note
        ----
        All derived classes must implement this method.
        """
        raise NotImplementedError()

