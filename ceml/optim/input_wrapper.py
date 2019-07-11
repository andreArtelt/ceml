# -*- coding: utf-8 -*-
import numpy as np


class InputWrapper():
    """Class for wrapping an input.

    The :class:`InputWrapper` class wraps an inputs to hide some of its dimensions/features to subsequent methods.

    Parameters
    ----------
    features_whitelist : `list(int)`
        A non-empty list of feature indices (dimensions of the input space) that can be used when computing the counterfactual.
        
        If `feature_whitelist` is None, all features can be used.
    x_orig : `numpy.array`
        The original input that is going to be wrapped - this is the input whose prediction has to be explained.
        
    Raises
    ------
    ValueError
        If `features_whitelist` is an empty list.
    """
    def __init__(self, features_whitelist, x_orig):
        self.x_orig = x_orig
        self.features_whitelist = features_whitelist
        if self.features_whitelist == []:
            raise ValueError("'features_whitelist' does not contain any features. If you do not want to restrict the features, use 'features_whitelist=None' instead of '[]'")
        
        super(InputWrapper, self).__init__()
    
    def complete(self, x):
        """Completing a given input.

        Adds the fixed/hidden dimensions from the original input to the given input.

        Parameters
        ----------
        x : `array_like`:
            The input to be completed.
        
        Returns
        -------
        `numpy.array`
            The completed input.
        """
        if self.features_whitelist is None:
            return x
        else:
            return np.array([x[self.features_whitelist.index(i)] if i in self.features_whitelist else self.x_orig[i] for i in range(self.x_orig.shape[0])])

    def extract_from(self, x):
        """Extracts the whitelisted dimensions from a given input.

        Parameters
        ----------
        x : `array_like`:
            The input to be processed.
        
        Returns
        -------
        `numpy.array`
            The extracted input - only whitelisted features/dimensions are kept.
        """
        if self.features_whitelist is None:
            return x
        else:
            return x[self.features_whitelist]

    def __call__(self, x):
        return self.complete(x)
