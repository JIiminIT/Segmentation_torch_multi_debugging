from __future__ import division
from __future__ import absolute_import

import numpy as np
import tinycat as cat

from scipy import ndimage


__all__ = [
    "percentile_normalization",
    "z_score_norm",
    "min_max_normalization",
    "histogram_equalization",
    "n4_bias_correction",
]


def percentile_normalization(data: np.ndarray, percentile: int = 1) -> np.ndarray:
    """percentile min-max normalization from mricron

    Args:
        data (np.ndarray): input array
        percentile (int, optional): Defaults to 1.

    Returns:
        np.array: normalized data
    """

    min_percentile = np.percentile(data, percentile)
    max_percentile = np.percentile(data, 100 - percentile)

    # limit maximum intensity of data by max_percentile
    data[data >= max_percentile] = max_percentile

    # limit minimum intensity of data by min_percentile
    data[data <= min_percentile] = min_percentile

    return data


def z_score_norm(data: np.ndarray) -> np.ndarray:
    """ z_score normalization or zero-mean normalization

    Args:
        data (np.ndarray): n-dimensional array

    Returns:
        np.ndarray: normalized data
    """
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data - mean) / std

    # Z-score normalization's result has zero-mean.
    # nan values should be replaced as zero
    normalized[np.isnan(normalized)] = 0
    return normalized


def min_max_normalization(
    data: np.ndarray, top: int = 255, floor: bool = True
) -> np.ndarray:
    """ min-max normalization with top

    :param array data: array with any size and dimension
    :param int top: top number of normalized data
    :param bool floor: choose if floor normalized value
    :returns: intensity normalized data
    """
    # Converting dtype can prevent Overflow Error
    data = data.astype(float)

    lmin = data.min()
    lmax = data.max()
    if floor:
        return np.floor((data - lmin) / (lmax - lmin) * top)
    else:
        return (data - lmin) / (lmax - lmin)


def histogram_equalization(data: np.ndarray) -> np.ndarray:
    """ Histogram equalization

    :param array data: input array data
    :returns: histogram equalized data
    """
    # @TODO Validation
    histogram = np.histogram(data, bins=np.arange(len(data) + 1))[0]
    histograms = np.cumsum(histogram) / float(np.sum(histogram))
    e = np.floor(histograms[data.flatten().astype("int")] * 255)
    return e.reshape(data.shape)


def n4_bias_correction(mri, mask_image=None, shrink_factor=(4, 4, 4)):
    """ process n4 bias-field correction to mri subject.

    :param mri: nibabel mri instance
    :param mask_image: array of mri mask
    :param shrink_factor: factor of shrink
    :returns: returns bias-field corrected nibabel instance
    """
    from tinycat.label import gen_mask
    import SimpleITK as sitk

    mri_data = mri.get_data()
    mri_image = sitk.GetImageFromArray(mri_data)
    mri_image = sitk.Cast(mri_image, sitk.sitkFloat32)

    if mask_image is None:
        mask_image = sitk.OtsuThreshold(mri_image, 1)
    else:
        mask_image = sitk.GetImageFromArray(mask_image)

    # Shrink image to minimize computation cost
    mri_image_sh = sitk.Shrink(mri_image, shrink_factor)
    mask_image_sh = sitk.Shrink(mask_image, shrink_factor)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    # Default parameters for slicer 3D
    corrector.SetSplineOrder = 3
    corrector.SetConvergenceThreshold = 0.0001
    corrector.SetMaximumNumberOfIterations = [50, 50, 50]
    corrector.SetWienerFilterNoise = 0
    corrector.SetNumberOfHistogramBins = 0
    corrector.SetBiasFieldFullWidthAtHalfMaximum = 0.15

    # Calculate bias-field filter
    n4_output = corrector.Execute(mri_image_sh, mask_image_sh)
    n4_filter = sitk.Subtract(n4_output, mri_image_sh)

    # Apply bias-field filter to masked original data
    n4_array = ndimage.interpolation.zoom(
        sitk.GetArrayFromImage(n4_filter), zoom=shrink_factor, order=3
    )
    mri_data = sitk.GetArrayFromImage(mri_image)
    semi_mask = mri_data >= mri_data.mean()
    mask = gen_mask(semi_mask)
    mri_data[mask] = mri_data[mask] - n4_array[mask]

    return cat.Nifti1Image(mri_data, mri.affine, mri.header)
