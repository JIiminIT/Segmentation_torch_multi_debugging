"""
tinycat: neurophet nifti & neural network package
==================================================

Contents
--------
Tinycat imports and wraps several functions from the nibabel namespace,
Implements various numpy-based matrix calculation & neuroimaging functional APIs
"""

from __future__ import absolute_import
from nibabel import Nifti2Image

from tinycat.about import *
from tinycat.nifti_utils import *

# pylint: disable=redefined-builtin
from tinycat import (
    base,
    elastix,
    eval,
    freesurfer,
    gpu,
    label,
    lut,
    multicore,
    nifti,
    nn,
    normalization,
    np,
    patch,
    plot,
    volume,
)

del absolute_import
