# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class Counterfactual(ABC):
    """Base class for computing a counterfactual.

    Note
    ----
    The class :class:`Counterfactual` can not be instantiated because it contains an abstract method. 
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def compute_counterfactual(self):
        """Compute a counterfactual.

        Abstract method for computing a counterfactual.

        Note
        ----
        All derived classes must implement this method.
        """
        raise NotImplementedError()
