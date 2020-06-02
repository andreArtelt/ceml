# -*- coding: utf-8 -*-
import numpy as np
from functools import reduce


class AffinePreprocessing():
    """
    Wrapper for an affine mapping (preprocessing)
    """
    def __init__(self, A, b, **kwds):
        self.A = A
        self.b = b

        super().__init__(**kwds)

def concatenate_affine_mappings(mappings):
    A = reduce(np.matmul, reversed([m.A for m in mappings]))
    b = mappings[-1].b
    if len(mappings) > 1:
        b = np.dot(mappings[1].A, mappings[0].b) + mappings[1].b
        for i in range(2, len(mappings)):
            b = np.dot(mappings[i].A, b) + mappings[i].b

    return A, b
