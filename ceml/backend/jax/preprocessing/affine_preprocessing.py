# -*- coding: utf-8 -*-
import numpy as np
from functools import reduce


class AffinePreprocessing():
    """
    Wrapper for an affine mapping (preprocessing)
    """
    def __init__(self, A, b):
        self.A = A
        self.b = b


def concatenate_affine_mappings(mappings):
    A = reduce(np.matmul, [m.A for m in mappings])
    b = mappings[-1].b
    if len(mappings) > 1:
        b += reduce(np.matmul, [np.dot(mappings[i].A, mappings[i-1].b) for i in range(1, len(mappings))])

    return A, b
