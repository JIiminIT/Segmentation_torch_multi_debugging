import numpy as np
from tinycat import lut


def mm_to_ml(mm):
    """convert mm^3 unit to ml
    
    Args:
        mm (int): millimeter
    
    Returns:
        float: milliliter
    """

    return mm * 0.001


def estimate_volumes(array):
    uniques, counts = np.unique(array, return_counts=True)
    volumes = {}

    for unique, count in zip(uniques, counts):
        volumes[unique] = count

    return volumes


def get_nifti_with_volumes(nifti_items):
    for i in range(len(nifti_items)):
        nifti_items[i].volume = estimate_volumes(nifti_items[i].dataobj)
    return nifti_items
