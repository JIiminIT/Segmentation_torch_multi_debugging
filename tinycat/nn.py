"""
Tinycat's numpy implementations of tf.nn,
necessary neural network numpy operations for imaging
"""


import numpy as np


__all__ = ["softmax"]


def softmax(x, axis=-1):
    """convert n-dimensional array object into softmax

    Args:
        x (np.ndarray): target array object
        axis (int, optional): Defaults to -1. 

    Returns:
        np.ndarray: softmax array
    """

    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
