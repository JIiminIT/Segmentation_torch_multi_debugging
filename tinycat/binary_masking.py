# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import numpy as np
import scipy.ndimage as ndimg
from scipy.ndimage.morphology import binary_fill_holes as fill_holes

from tinycat.util_common import look_up_operations, otsu_threshold

"""
This class defines methods to generate a binary image from an input image.
The binary image can be used as an automatic foreground selector, so that later
processing layers can only operate on the `True` locations within the image.
"""
SUPPORTED_MASK_TYPES = set(
    ["threshold_plus", "threshold_minus", "otsu_plus", "otsu_minus", "mean_plus"]
)

SUPPORTED_MULTIMOD_MASK_TYPES = set(["or", "and", "multi"])


class BinaryMaskingLayer(object):
    def __init__(self, type_str="otsu_plus", multimod_fusion="or", threshold=0.0):
        self.type_str = look_up_operations(type_str.lower(), SUPPORTED_MASK_TYPES)
        self.multimod_fusion = look_up_operations(
            multimod_fusion.lower(), SUPPORTED_MULTIMOD_MASK_TYPES
        )
        self.threshold = threshold

    def __make_mask_3d(self, image):
        assert image.ndim == 3
        image_shape = image.shape
        image = image.reshape(-1)
        mask = np.zeros_like(image, dtype=np.bool)
        thr = self.threshold
        if self.type_str == "threshold_plus":
            mask[image > thr] = True
        elif self.type_str == "threshold_minus":
            mask[image < thr] = True
        elif self.type_str == "otsu_plus":
            thr = otsu_threshold(image) if np.any(image) else thr
            mask[image > thr] = True
        elif self.type_str == "otsu_minus":
            thr = otsu_threshold(image) if np.any(image) else thr
            mask[image < thr] = True
        elif self.type_str == "mean_plus":
            thr = np.mean(image)
            mask[image > thr] = True
        mask = mask.reshape(image_shape)
        mask = ndimg.binary_dilation(mask, iterations=2)
        mask = fill_holes(mask)
        # foreground should not be empty
        assert np.any(mask == True), (
            "no foreground based on the specified combination parameters, "
            "please change choose another `mask_type` or double-check all "
            "input images"
        )
        return mask

    def layer_op(self, image):
        mask = self.__make_mask_3d(image)

        return mask
