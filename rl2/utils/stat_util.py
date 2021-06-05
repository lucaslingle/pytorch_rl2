"""
Utility module containing basic statistical operations.
"""

import numpy as np


def standardize(arr: np.ndarray) -> np.ndarray:
    """
    Computes the empirical z-scores of an array.

    Args:
        arr: a numpy array

    Returns:
        a numpy array of z-scores.
    """
    eps = 1e-8
    mu = arr.mean()
    sigma = arr.std()
    standardized = (arr - mu) / (eps + sigma)
    return standardized


def explained_variance(ypred: np.ndarray, y: np.ndarray) -> np.float32:
    """
    Computes the explained variance.
    See https://en.wikipedia.org/wiki/Explained_variation

    Args:
        ypred: predicted values.
        y: actual values.

    Returns:
        The explained variance, a number between 0 and 1.
    """
    vary = y.var()
    return 1 - (y-ypred).var()/(1e-8 + vary)