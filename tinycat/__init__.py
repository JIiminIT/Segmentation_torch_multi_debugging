"""
tinycat: neurophet nifti & neural network package
==================================================

Contents
--------
Tinycat imports and wraps several functions from the nibabel namespace,
Implements various numpy-based matrix calculation & neuroimaging functional APIs
"""

# pylint: disable=redefined-builtin
from tinycat import (
    thirdparty,
    evaluation,
    lut,
    data,
    plot,
    label,
    patch,
    nifti,
    system,
    volume,
    metrics,
    multicore,
    normalization,
    binary_masking,
    util_common,
)
# 버젼정보 등 import
from tinycat.about import *
# nifti 로딩 및 관련 함수들은 최상위 이름공간에서 사용할 수 있습니다.
from tinycat.nifti_utils import *
