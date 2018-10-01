# Implementation of various likeness measures for images

import numpy as np
from SmulException import SmulException

def compare(I1, I2):
    """
    Highest-level interface, through which image
    comparisons should be conducted.
    """
    return meanSquaredError(I1, I2)

def meanSquaredError(I1, I2):
    """
    Compute the mean-squared-error of the two images, i.e.

       err = 1/(m*n) Sum_i (Sum_j ( I1_ij - I2_ij )^2 )
    """
    if I1.shape != I2.shape:
        raise SmulException("Images are not of the same size")

    err = np.sum((I1 - I2)**2)
    err /= I1.shape[0] * I1.shape[1]

    return err

