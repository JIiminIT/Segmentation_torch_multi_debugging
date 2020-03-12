"""Efficient data structures and lookup tables"""

import os
from tinycat._base import namedtuple_with_defaults as namedtuple

__all__ = [
    "ValueOrientedLut",
    "NameOrientedLut",
    "FREESURFER_NAME_LUT",
    "FREESURFER_VALUE_LUT",
    "ASEG_WH_LUT",
    "ASEG_CC_LUT",
    "ASEG_SP_LUT",
    "SEGENGINE_V1_LUT",
    "SEGENGINE_STROKE_LUT",
]

# Classes for lookup table
ValueOrientedLut = namedtuple("LookupTable", ["name", "r", "g", "b"])
NameOrientedLut = namedtuple("LookupTable", ["value", "r", "g", "b"])


def _parse_freesurfer_colorlut():
    """Parse freesurfer colorlut from raw textfile
    
    Returns:
        dict: dictionary of lookuptable
    """

    try:
        filepath = __file__
    except NameError:
        filepath = "tinycat/"

    value_lut = {}
    name_lut = {}

    lut_file = os.path.join(
        os.path.dirname(filepath), "resources/FreeSurferColorLUT.txt"
    )

    try:
        with open(lut_file, mode="r", encoding="utf-8") as lut:
            lut_contents = lut.readlines()
            lut_contents = [
                c.replace("\n", "").replace("-", "_")
                for c in lut_contents
                if c and not c.startswith("#")
            ]
    except FileNotFoundError:
        lut = "lut file not found"
        return lut, lut

    for content in lut_contents:
        if content:
            attr = content.split()
            value_lut[int(attr[0])] = ValueOrientedLut(
                name=attr[1], r=int(attr[2]), g=int(attr[3]), b=int(attr[4])
            )
            name_lut[attr[1]] = NameOrientedLut(
                value=attr[0], r=int(attr[2]), g=int(attr[3]), b=int(attr[4])
            )

    return value_lut, name_lut


FREESURFER_VALUE_LUT, FREESURFER_NAME_LUT = _parse_freesurfer_colorlut()

# Ordinary brain, from FreeSurfer
# https://surfer.nmr.mgh.harvard.edu/ftp/articles/fischl02-labeling.pdf
ASEG_BRAIN_LUT = {
    "Unknown",
    "Left Cerebral White Matter",
    "Left Cerebral Cortex",  # APARC
    "Left Lateral Ventricle",
    "Left Inferior Lateral Ventricle",
    "Left Cerebellum White Matter",
    "Left Cerebellum Cortex",
    "Left Thalamus",
    "Left Caudate",
    "Left Putamen",
    "Left Pallidum",
    "Left Hippocampus",
    "Left Amygdala",
    "Left Lesion",
    "Left Accumbens area",
    "Left Ventral Diencephalon",
    "Left Vessel (non-specific)",
    "Right Cerebral White Matter",
    "Right Cerebral Cortex",  # APARC
    "Right Lateral Ventricle",
    "Right Inferior Lateral Ventricle",
    "Right Cerebellum White Matter",
    "Right Cerebellum Cortex",
    "Right Thalamus",
    "Right Caudate",
    "Right Putamen",
    "Right Pallidum",
    "Right Hippocampus",
    "Right Amygdala",
    "Right Lesion",
    "Right Accumbens area",
    "Right Ventral Diencephalon",
    "Right Vessel (non-specific)",
    "Third Ventricle",
    "Fourth Ventricle",
    "Brain Stem",
    "Cerebrospinal Fluid",
}

# Corpus Callosum
ASEG_CC_LUT = {
    "CC_Posterior",
    "CC_Mid_Posterior",
    "CC_Central",
    "CC_Mid_Anterior",
    "CC_Anterior",
}

# Optic & Hypointensity
ASEG_SP_LUT = {"WH-Hypointensities", "non-WM-hypointensities", "Optic-Chiasm"}

# 72  5th-Ventricle                           120 190 150 0

# 77  WM-hypointensities                      200 70  255 0
# 78  Left-WM-hypointensities                 255 148 10  0
# 79  Right-WM-hypointensities                255 148 10  0
# 80  non-WM-hypointensities                  164 108 226 0
# 81  Left-non-WM-hypointensities             164 108 226 0
# 82  Right-non-WM-hypointensities            164 108 226 0

# update aseg lut
ASEG_LUT = {*ASEG_BRAIN_LUT, *ASEG_CC_LUT, *ASEG_SP_LUT}

# corresponding label classes used in segengine
SEGENGINE_V1_LUT = {
    0: "space",
    1: "cerebral gray matter",
    2: "cerebral white matter",
    3: "cerebellar gray matter",
    4: "cerebellar white matter",
    5: "lateral ventricles",
    6: "cerebrospinal fluid",
    7: "skull",
    8: "scalp",
}

# corresponding label classes for future stroke segmentation
SEGENGINE_STROKE_LUT = {0: "non-stroke", 1: "ischemic stroke", 2: "hemorrhagic stroke"}

# aparc+aseg.mgz based aqua label indexes
AQUA_LABEL_V2 = [
    0,
    2,
    4,
    5,
    7,
    8,
    10,
    11,
    12,
    13,
    14,
    15,
    17,
    18,
    26,
    28,
    31,
    41,
    43,
    44,
    46,
    47,
    49,
    50,
    51,
    52,
    53,
    54,
    58,
    60,
    63,
    77,
    251,
    252,
    253,
    254,
    255,
    1001,
    1002,
    1003,
    1005,
    1006,
    1007,
    1008,
    1009,
    1010,
    1011,
    1012,
    1013,
    1014,
    1015,
    1016,
    1017,
    1018,
    1019,
    1020,
    1021,
    1022,
    1023,
    1024,
    1025,
    1026,
    1027,
    1028,
    1029,
    1030,
    1031,
    1032,
    1033,
    1034,
    1035,
    2001,
    2002,
    2003,
    2005,
    2006,
    2007,
    2008,
    2009,
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
    2027,
    2028,
    2029,
    2030,
    2031,
    2032,
    2033,
    2034,
    2035,
]
