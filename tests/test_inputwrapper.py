# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'..')

import numpy as np

from ceml.optim import InputWrapper


def test_inputwrapper():
    x_orig = np.array([0, 1, 2, 3, 4, 5])

    wrapper = InputWrapper(features_whitelist=None, x_orig=x_orig)
    assert all(wrapper.extract_from(x_orig) == x_orig)
    assert all(wrapper.complete(x_orig) == x_orig)

    features_whitelist = [0, 2, 4]
    wrapper = InputWrapper(features_whitelist=features_whitelist, x_orig=x_orig)
    assert all(wrapper.extract_from(x_orig) == np.array([0, 2, 4]))
    assert all(wrapper.complete(np.array([42, 94, 27])) == np.array([42, 1, 94, 3, 27, 5]))