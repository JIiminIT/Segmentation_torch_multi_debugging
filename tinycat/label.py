from __future__ import print_function
from __future__ import division

from scipy import ndimage
from typing import List, Optional
import numpy as np


__all__ = [
    "fill_and_clean_label",
    "get_single_labeled",
    "get_largest_label",
    "gen_mask",
    "argmax_of_neighbor",
    "connected_components_analysis",
    "in_out_correction",
    "border_cleaning",
    "kernel_correction",
    "np_correction",
]


def fill_and_clean_label(array: np.ndarray, label: int = 1) -> np.ndarray:
    """Fill holes and detach isolated points using
    connected components analysis

    Args:
        array (np.ndarray): target array
        label (int, optional): integer for label, Defaults to 1

    Returns:
        result: filled and cleaned label
    """

    result = array == label
    result = ndimage.binary_fill_holes(result)
    result = get_largest_label(result) * label
    return result


def get_single_labeled(data: np.ndarray) -> List:
    """get unique arrays for unique labels from input array

    Args:
        data (np.ndarray): input array

    Raises:
        TypeError: when data dtype is float

    Returns:
        List: list of unique labels
    """

    if data.dtype == "float":
        raise TypeError("Input dtype must be an int or uint")

    images = []
    labels = np.delete(np.unique(data), 0)
    for i in labels:
        single_labeled = np.copy(data)
        np.putmask(single_labeled, data != i, 0)
        images.append(single_labeled)
    return images


def get_largest_label(
    label: np.ndarray, structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """Find biggest connected components

    Args:
        label (np.ndarray): argmaxed label
        structure (Optional[np.ndarray]): optional ndarray

    Returns:
        clump_mask (np.ndarray): biggest component.
    """

    labels, _ = ndimage.label(label, structure=structure)
    size = np.bincount(labels.ravel())
    try:
        biggest_label = size[1:].argmax() + 1
    except ValueError:
        return label
    clump_mask = labels == biggest_label

    return clump_mask


def gen_mask(data: np.ndarray, iteration: int = 10) -> np.ndarray:
    """morphological close with custom iterations.

    Args:
        data (np.ndarray): input array
        iteration (int, optional): Defaults to 10. iterations for closing

    Returns:
        np.ndarray: mask data
    """

    struct = ndimage.morphology.generate_binary_structure(3, 3)
    th_vol = ndimage.morphology.binary_dilation(
        data, structure=struct, iterations=iteration
    )
    th_vol = ndimage.morphology.binary_erosion(
        th_vol, structure=struct, iterations=iteration
    )
    return th_vol


def argmax_of_neighbor(
    data: np.ndarray,
    masked_data: np.ndarray,
    kernel_size: int,
    without_zero: bool = False,
) -> np.ndarray:
    """ find adjusted value of masked data by looking its neighboring 3D kernel

    Args:
        data (np.ndarray): 3-dimensional array
        masked_data (np.ndarray): position mask to adjust
        kernel_size (int): size of kernel
        without_zero (bool, optional): Defaults to False. bincount from 1

    Returns:
        np.ndarray: filter adjusted data
    """

    i, j, k = np.where(masked_data)
    kernel_hf = kernel_size // 2
    for idx in range(len(i)):
        _min = [i[idx] - kernel_hf, j[idx] - kernel_hf, k[idx] - kernel_hf]
        _max = [i[idx] + kernel_hf, j[idx] + kernel_hf, k[idx] + kernel_hf]
        kernel = data[_min[0] : _max[0], _min[1] : _max[1], _min[2] : _max[2]].flatten()
        try:
            bincount = np.bincount(kernel)
            if without_zero:
                bincount = bincount[1:]
                argmax = bincount.argmax() + 1
            else:
                argmax = bincount.argmax()
            data[i[idx], j[idx], k[idx]] = argmax
        except ValueError:
            continue
    return data


def connected_components_analysis(
    data: np.ndarray,
    label: int = 1,
    theta: float = 1,
    sigma: float = 0.05,
    structure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """ detach unconnected components with thresholded size.

    Args:
        data (np.ndarray): n-dimensional array
        label (int, optional): Defaults to 1. number of label
        theta (float, optional): Defaults to 1. level of gaussian filter
        sigma (float, optional): Defaults to 0.05. threshold rate of connected components
        structure (Optional[np.ndarray], optional): Defaults to None.

    Returns:
        np.ndarray: connected components
    """

    # If input data is only consists of zeros, return raw input data
    if not np.any(data):
        return data

    if sigma:
        image = ndimage.gaussian_filter(data, sigma=sigma)
    else:
        image = data

    thresholded = image >= image.mean()
    connected_component, n_labels = ndimage.label(thresholded, structure=structure)
    connected_component = (connected_component > 0) * 1
    sizes = ndimage.sum(thresholded, connected_component, range(n_labels + 1))
    mask_size = sizes < max(sizes) * theta
    remove_pixel = mask_size[connected_component]
    connected_component[remove_pixel] = 0
    connected_component *= label

    return connected_component


def in_out_correction(
    ex_region: np.ndarray, in_region: np.ndarray, iterations: int = 2
) -> np.ndarray:
    """correct outter region to cover inner region

    Args:
        ex_region (np.ndarray): outter region
        in_region (np.ndarray): inner region
        iterations (int, optional): Defaults to 2. iteration of dilations

    Returns:
        np.ndarray: corrected data
    """
    if iterations:
        data = (
            ndimage.binary_dilation(in_region, iterations=iterations) ^ in_region
        ) | ex_region
        return data
    else:
        return ex_region


def out_in_correction(
    ex_region: np.ndarray, in_region: np.ndarray, iterations: int = 1
) -> np.ndarray:
    """remove exposed surface of inner region by
    referencing outter region

    Args:
        ex_region (np.ndarray): outter region
        in_region (np.ndarray): inner region
        iterations (int, optional): Defaults to 1. iteration number of erosion

    Returns:
        np.ndarray: corrected data
    """

    # 바깥 영역에서 1만큼 줄어든 만큼의 영역
    exposed_surface = (
        ndimage.binary_erosion(ex_region, iterations=iterations) ^ ex_region
    )
    # 안쪽 영역에서 노출된 부분을 제거
    in_region[exposed_surface] = 0
    return in_region


def slicewise_fill_holes(data: np.ndarray) -> np.ndarray:
    """Fill the holes in every 2d slices of 3d label array

    Args:
        data (np.ndarray): 3d label array

    Returns:
        np.ndarray: Transformation of the input data 
            where holes have been filled
    """

    for i in range(data.shape[0]):
        data[i, :, :] = ndimage.binary_fill_holes(data[i, :, :])
    for j in range(data.shape[1]):
        data[:, j, :] = ndimage.binary_fill_holes(data[:, j, :])
    for k in range(data.shape[2]):
        data[:, :, k] = ndimage.binary_fill_holes(data[:, :, k])
    return data


def border_cleaning(data, divide_by=10):
    """ fill calculated border with minimum.

    :param data: 3-dimensional array
    :param int divide_by: parameter for border size.
    :returns: border cleaned data.
    """
    shape = np.array(data.shape)
    i, j, k = shape // divide_by
    min_val = np.min(data)
    data[0:i, :, :] = min_val
    data[:, 0:j, :] = min_val
    data[:, :, 0:k] = min_val
    data[shape[0] - i :, :, :] = min_val
    data[:, shape[1] - j :, :] = min_val
    data[:, :, shape[2] - k :] = min_val
    return data


def kernel_correction(labeled_data, theta=0.3, sigma=0, filter_size=5):
    """connected component analysis and kernel-argmax based segmentation correction

    Args:
        labeled_data (np.ndarray): array of label data
        theta (float, optional): Defaults to 0.3. threshold rate of connected components
        sigma (int, optional): Defaults to 0. level of gaussian filter
        filter_size (int, optional): Defaults to 5. kernel window size

    Returns:
        np.array: corrected data
    """

    connected_components = np.zeros_like(labeled_data, dtype=np.int32)
    labeled_data = labeled_data.astype(np.int32)

    # analyze every unique labels except air
    unique = np.unique(labeled_data)[1:]
    for i in unique:
        component = labeled_data == i
        connected_components += connected_components_analysis(
            component, i, theta=theta, sigma=sigma
        )

    unconnected_labels = labeled_data - connected_components
    filled = ndimage.binary_fill_holes(connected_components)

    inner_zeros = filled - connected_components > 0

    # 2D-based binary fill holes
    for i in range(filled.shape[0]):
        filled[i, :, :] = ndimage.binary_fill_holes(filled[i, :, :])
    for j in range(filled.shape[1]):
        filled[:, j, :] = ndimage.binary_fill_holes(filled[:, j, :])
    for k in range(filled.shape[2]):
        filled[:, :, k] = ndimage.binary_fill_holes(filled[:, :, k])

    # kernel-argmax correction
    kernel_filtered_data = argmax_of_neighbor(
        connected_components, unconnected_labels, filter_size
    )
    inner_zero_filled = argmax_of_neighbor(
        kernel_filtered_data, inner_zeros, filter_size * 2, without_zero=True
    )

    return inner_zero_filled


def np_correction(np_label):
    """correction method for predefined label protocol

    :param np_label: predefined label data
    :returns: corrected data
    """
    cerebral_gm = np_label == 1
    cerebral_wm = np_label == 2
    cerebellar_gm = np_label == 3
    cerebellar_wm = np_label == 4
    ventricles = np_label == 5
    csf = np_label == 6
    skull = np_label == 7
    skin = np_label == 8

    cerebellum = cerebellar_gm | cerebellar_wm

    # enforce upper cerebral gm to cover cerebral wm
    cgm_dil = ndimage.binary_dilation(cerebral_gm, iterations=2)
    cwm_dil = ndimage.binary_dilation(cerebral_wm, iterations=2)
    cerebral_gm = ((csf | cerebellum) & cgm_dil & cwm_dil) | cerebral_gm

    csf = in_out_correction(csf, (cerebral_gm | cerebral_wm), iterations=2)
    csf = in_out_correction(csf, cerebellum, iterations=2)
    skull = in_out_correction(skull, csf, iterations=2)
    skin = in_out_correction(skin, skull, iterations=2)
    np_corrected = np.zeros_like(np_label, dtype=np.uint8)

    # reversed regions
    regions = np.array(
        [
            skin,
            skull,
            csf,
            ventricles,
            cerebellar_wm,
            cerebellar_gm,
            cerebral_wm,
            cerebral_gm,
        ],
        dtype=np.uint8,
    )

    rev_range = range(8, -0, -1)

    for idx, region in zip(rev_range, regions):
        np_corrected += region * idx
        np.putmask(np_corrected, np.logical_and(region, True), idx)

    return np_corrected
