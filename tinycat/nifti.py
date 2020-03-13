"""nifti 영상 preprocessing 등에 이용할 수 있는 함수들을 모은 모듈"""
import copy
import warnings
import numpy as np
import nibabel as nib
from scipy import ndimage

import tinycat as cat


__all__ = [
    "rough_crop_3d",
    "resample_mri",
    "affine_correction",
    "get_coordinate_system",
    "reverse",
    "coordinate_converter",
    "convert_image_coordinates",
]

# Ignore scipy warning.
warnings.filterwarnings("ignore", ".*the output shape of zoom.*")


def rough_crop_3d(mri_data, return_coordinate=False):
    """Roughly crop Region of interest from the original image.

    :param mri_data: array of mri data.
    :param return_coordinate: return the coordinates information used for cropping
    :returns: cropped image
    """
    thresholded = mri_data > mri_data.mean() * 0.5
    shape = thresholded.shape
    pad = 24
    padded = np.pad(thresholded, pad, mode="constant", constant_values=0)
    mask = cat.label.gen_mask(padded, 10)
    mask = mask[
        0 + pad : shape[0] + pad, 0 + pad : shape[1] + pad, 0 + pad : shape[2] + pad
    ]

    x, y, z = np.where(mask)
    mri_data = mri_data[
        np.min(x) : np.max(x), np.min(y) : np.max(y), np.min(z) : np.max(z)
    ]

    if return_coordinate:
        return mri_data, (x, y, z)

    return mri_data


def resample_mri(
    target,
    header=None,
    image_spacing=None,
    template_spacing=(1, 1, 1),
    dim_init=None,
    dim_result=None,
    spline_order=3,
):
    """ Resample mri data into standard 1mm spacing or wanted dimension

    :param target: nibabel type data
    :param header: nifti1 header data
    :param tuple image_spacing: provide image spacing
    :param tuple template_spacing: provide template image spacing
    :param tuple dim_init: Initial dimension
    :param tuple dim_result: Targeted dimension
    :param int spline_order: spline order of resampling
    :returns: resampled nibabel type data
    """

    # we use deepcopy of an object to prevent pointer overwriting
    mri = copy.deepcopy(target)

    data = mri.get_data()

    if image_spacing is None:
        image_spacing = mri.header.get_zooms()

    if dim_init is not None:
        if dim_result is not None:
            image_spacing = np.divide(dim_result, dim_init)
            if template_spacing is None:
                template_spacing = mri.header.get_zooms() / image_spacing

    image_resized = ndimage.interpolation.zoom(
        data, zoom=image_spacing, order=spline_order
    )
    affine = affine_correction(mri.affine, image_spacing)

    if header is not None:
        resampled = cat.Nifti1Image(image_resized, affine, header=header)
        for letter in ["b", "c", "d"]:
            resampled.header["quatern_{}".format(letter)] = np.copy(
                header["quatern_{}".format(letter)]
            )
        resampled.header["qform_code"] = np.copy(header["qform_code"])

    else:
        resampled = cat.Nifti1Image(image_resized, affine, header=mri.header)

    resampled.header.set_zooms(template_spacing)
    return resampled


def affine_correction(affine, image_spacing):
    """ Affine transformation for image spacing.

    :param affine: 4x4 array of affine transformation
    :param image_spacing: iterable of x y z image spacing
    :returns: adjusted affine transformation
    """
    for i in range(3):
        for j in range(3):
            affine[i][j] = affine[i][j] / image_spacing[j]
    return affine


def get_coordinate_system(affine):
    """ Get coordinate system of the mri by looking its affine transformation.

    :param affine: 4x4 array of affine transformation
    :returns: string included list type coordinate system
    """
    coordinates = affine[:3, :3]
    coordinates = np.linalg.inv(coordinates)
    absolutes = np.absolute(coordinates)
    ras_system = ["R", "A", "S"]
    lpi_system = ["L", "P", "I"]
    system = []
    for i in range(3):
        for j in range(3):
            indices = np.where(absolutes[i] == absolutes[i].max())
            if indices[0] == j:
                if coordinates[i][indices[0]] > 0:
                    system.append(ras_system[j])
                else:
                    system.append(lpi_system[j])
    return system


def reverse(system):
    """ Simply reverse the coordinate system letter.

    :param system: 1 letter of coordinate system.
    :returns: reversed coordinate system letter.
    """
    if system is "L":
        system = "R"
    elif system is "R":
        system = "L"
    elif system is "S":
        system = "I"
    elif system is "I":
        system = "S"
    elif system is "A":
        system = "P"
    elif system is "P":
        system = "A"
    return system


def _index_changer(goal, coordinates):
    """ Change the index of coordinates.

    :param goal: coordinate system that you want to make
    :param coordinates: current coordinate
    :returns: list that can be used for transposition.
    """
    index = []
    changed = []
    for i in range(3):
        for j in range(3):
            if (coordinates[i] == goal[j]) | (coordinates[i] == reverse(goal[j])):
                if i == j:
                    index.append(i)
                else:
                    index.append(j)
        changed.append(coordinates[index[i]])
    return index, changed


def coordinate_converter(data, coordinates, goal):
    """ convert 3-dimensional data into goal coordinate system

    :param data: 3-dimensional array data
    :param coordinates: coordinates of the input data
    :param goal: 3-letter included iterable, goal coordinate system
    :returns: converted data.
    """
    index, changed = _index_changer(goal, coordinates)
    data = np.transpose(data, axes=index)
    for i in range(3):
        if changed[i] is goal[i]:
            continue
        else:
            data = np.flip(data, i)
    return data


def convert_image_coordinates(image, out_axcodes):
    """convert image coordinates based on nibabel orientations

    Args:
        image (nib.Nifti1Image): nibabel image
        out_axacodes (tuple): tuple of axcode

    Returns:
        nib.Nifti1Image: transformed image
    """

    in_axcodes = nib.aff2axcodes(image.affine)
    in_ornt = nib.orientations.axcodes2ornt(in_axcodes)
    out_ornt = nib.orientations.axcodes2ornt(out_axcodes)
    transformed_ornt = nib.orientations.ornt_transform(in_ornt, out_ornt)
    transformed_array = nib.orientations.apply_orientation(
        image.dataobj, transformed_ornt
    )
    transformed_image = cat.Nifti1Image(
        transformed_array, image.affine, header=image.header
    )
    return transformed_image
