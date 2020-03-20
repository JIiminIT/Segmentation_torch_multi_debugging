"""3d multilabel 영상 처리 알고리즘들을 모아 둔 모듈"""
# pylint: disable=unsupported-assignment-operation, no-member, bad-continuation
from typing import List, Optional
import numpy as np
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation, binary_closing, binary_fill_holes


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
    "n_connections",
    "PostProcessingLayer",
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


def detach_percentile_particles(label: np.ndarray, theta: float = 0.01) -> np.ndarray:
    """detach certain sizes of particles from aqua segmentation results

    Args:
        label (np.ndarray): aqua 104 label segmentation results
        theta (float): percentage of size to allow from biggest label, Defaults to 0.01

    Returns:
        np.ndarray: particle detached results
    """
    detached_reigon = connected_components_analysis(label, label=1, theta=theta)
    label[(label > 0) & (detached_reigon == 0)] = 0
    return label.astype(np.uint8)


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
    theta: float = 0.05,
    sigma: float = 0,
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
        thresholded = image >= image.mean()
    else:
        thresholded = data

    connected_component, n_labels = ndimage.label(thresholded, structure=structure)
    sizes = ndimage.sum(thresholded, connected_component, range(n_labels + 1))
    mask_size = sizes < max(sizes) * theta
    remove_pixel = mask_size[connected_component]
    connected_component[remove_pixel] = 0
    connected_component[connected_component > 0] = label

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


def n_connections(array: np.array) -> List:
    """measures how many connected components are included in array

    Args:
        array (iterable): numpy array or iterable object

    Returns:
        n_connections (list): n_connections of the input data
    """

    uniques = np.unique(array)

    if len(uniques) > 1:
        connections = []
        for unique in uniques:
            connections.append(ndimage.label(array == unique)[1])
    else:
        connections = ndimage.label(array)[1]
    return n_connections


class PostProcessingLayer(object):
    """PostProcessingLayer for SegEngine segmentation

    Args:
        object (class)
    """

    def __init__(self, mode: str = "9-label"):
        self.mode = mode

    def run(self, *args, **kwargs):
        if self.mode == "simplified":
            return self._sequential_postprocess_simplified(*args, **kwargs)
        if self.mode == "9-label":
            return self._sequential_postprocess_v1(*args, **kwargs)
        if self.mode == "10-label":
            return self._sequential_postprocess_v1(*args, **kwargs)
        if self.mode == "9-label+stroke":
            return self._stroke_postprocess(*args, **kwargs)

    def _stroke_postprocess(
        self,
        label: np.array,
        stroke: np.array,
        edited: bool = False,
        layer_gap: int = 2,
    ) -> np.array:
        """merges 9-label segmentation result and
        ischemic stroke segmentation result

        Args:
            label (np.ndarray): label array
            stroke (np.ndarray): stroke array
            edited (bool): edited label should not be processed dramatically
            layer_gap (int): number of iterations for in_out_correction to
                cover internal tissues with external tissues

        Returns:
            result (np.ndarray): result array

        Procedure:
            1. CSF 라벨의 binary closing을 통해 CSF 라벨의 안에
            외부 라벨 (Skull, Skin)이 포함되지 않도록 한다.
            2. 속을 채운 CSF 라벨의 바깥쪽에 존재하는 Stroke는 모두 제거된다.
            3. Stroke label 중 가장 큰 덩어리의 5% 이하 크기의 Stroke는 노이즈로 간주하고 모두 없앤다.
            4. CSF는 Stroke를 감싸도록 Stroke가 빠져나간 부분만큼 dilation된다.
            5. Skull은 CSF를 감싸도를 CSF가 빠져나간 부분만큼 dilation된다.
            6. Skin은 Skull을 감싸도를 Skull이 빠져나간 부분만큼 dilation된다.
        """
        # extract brain from label
        brain_label = (label < 6) & (label != 0)
        brain = label[brain_label]

        # csf has to cover stroke in order to use meshengine
        csf = (label != 0) & (label < 7)

        # prevent skull or scalp included inside of csf
        if not edited:
            csf = binary_closing(csf, iterations=15)

        csf = binary_fill_holes(csf)

        # remove stroke out of csf
        # assume that stroke does not included in skull and skin location
        # skull and skin location is redefined while postprocessing using csf's binary closing
        stroke[csf == 0] = 0

        if not edited:
            stroke = connected_components_analysis(stroke, theta=0.05)

        # csf should cover stroke
        stroke = out_in_correction(csf, stroke)

        # if stroke label is empty, return input label
        if not np.any(stroke):
            return label

        skull = (label != 0) & (label != 8)
        csf = out_in_correction(skull, csf)

        skin = label > 0 | skull
        skull = out_in_correction(skin, skull)

        # csf has to cover the border of brain and stroke
        csf = in_out_correction(csf, (stroke | brain_label), iterations=layer_gap)
        # skull has to cover the border of csf
        skull = in_out_correction(skull, csf, iterations=layer_gap)
        # skin has to cover the border of skull
        skin = in_out_correction(skin, skull, iterations=layer_gap)

        # craft labels
        result = np.zeros_like(label)
        result[skin > 0] = 8
        result[skull > 0] = 7
        result[csf > 0] = 6
        result[brain_label] = brain

        # finalize label with stroke
        result[stroke > 0] = 13

        return result

    @staticmethod
    def _brain_processing(label: np.array, cover_wm: bool = True) -> np.array:
        """process brain label consists of 5 labels by detach particles
        and unconnected labels to make standard cerebrum and cerebellum

        Args:
            label (np.ndarray): label array
            cover_wm (bool): If specified, try to cover white matter with gray matter

        Returns:
            brain (np.ndarray): processed brain
        """

        # Layering order:
        # Cerebral WM -> Cerebral GM -> Cerebellar WM -> Cerebellar GM -> Ventricles ->
        cerebral_wm = fill_and_clean_label(label, 2)
        cerebral_gm = label == 1
        cerebral_gm = get_largest_label(cerebral_gm)
        cerebral_gm = cerebral_gm | (cerebral_wm)
        cerebral_gm = fill_and_clean_label(cerebral_gm, 1)

        cerebellar_wm = fill_and_clean_label(label, 4)
        cerebellar_gm = label == 3
        cerebellar_gm = get_largest_label(cerebellar_gm)
        cerebellar_gm = (cerebellar_gm | (cerebellar_wm)) * 3
        cerebellar_gm = fill_and_clean_label(cerebellar_gm, 3)

        lateral_ventricles = slicewise_fill_holes(label == 5)
        lateral_ventricles = (
            connected_components_analysis(lateral_ventricles, theta=0.3) * 5
        )

        cerebrum = np.where(cerebral_wm > 0, 2, cerebral_gm)

        if cover_wm:
            corrected_gm = in_out_correction(cerebrum == 1, cerebrum == 2)
            dilated_gm = binary_dilation(cerebrum == 1, iterations=3)
            corrected_gm = dilated_gm & corrected_gm
            cerebrum[corrected_gm] = 1

        # Assemble Brains include Ventricle, exclude CSF
        # Recover the disconnection after inserting ventricles
        cerebrum = np.where(lateral_ventricles > 0, 2, cerebrum)
        cerebrum = np.where(binary_fill_holes(cerebrum == 2), 2, cerebrum)
        cerebrum = np.where(lateral_ventricles > 0, 5, cerebrum)

        # Detach isolated cerebrum
        isolated_cerebrum = get_largest_label(cerebrum > 0) ^ (cerebrum > 0)
        cerebrum[isolated_cerebrum] = 0

        # Initialize cerebellum
        cerebellum = np.where(cerebellar_wm > 0, 3, cerebellar_gm)
        cerebellum = np.where(binary_fill_holes(cerebellum == 3), 3, cerebellum)
        cerebellum = np.where(cerebellar_wm > 0, 4, cerebellum)

        # Brain -> cerebrum + cerebellum
        brain = np.where(cerebellum > 0, cerebellum, cerebrum)
        return brain

    def _sequential_postprocess_v1(
        self,
        label: np.array,
        cover_wm: bool = True,
        simplify: bool = False,
        layer_gap: int = 2,
    ) -> np.array:
        """Sequential postprocessing of mri segmentation results.

        Args:
            label (np.ndarray): argmaxed segmentation results in NEUROPHET's protocol
            cover_wm (bool): If specified, try to cover white matter with gray matter
            simplify (bool): If specified, simplify label
            layer_gap (int): number of iterations for in_out_correction to 
                cover internal tissues with external tissues

        Returns:
            label: post processed result in  NEUROPHET's protocol.
        """
        # Layering Order:
        # Brain -> CSF -> Skull -> Scalp
        brain = self._brain_processing(label, cover_wm)

        # Extract CSF covering brain
        csf = (label != 0) & (label < 7)
        csf = binary_fill_holes(csf)
        csf = get_largest_label(csf)

        # Extract Skull covering CSF
        skull = (label != 0) & (label != 8)
        skull = get_largest_label(skull)
        skull = binary_fill_holes(skull)
        csf = out_in_correction(skull, csf)

        # Skin Process
        skin = label > 0
        skin = get_largest_label(skin)
        skin = binary_fill_holes(skin)
        skin = binary_closing(skin)

        # remove skin particles
        skull = out_in_correction(skin, skull)

        # Cover inner region with outter region
        csf = in_out_correction(csf, brain, iterations=layer_gap)
        skull = in_out_correction(skull, csf, iterations=layer_gap)
        skin = in_out_correction(skin, skull, iterations=layer_gap)

        # Start Assembling all Brain, Skull and Skin
        csf_brain = np.where(brain > 0, brain, csf * 6)
        skull_brain = np.where(csf_brain > 0, csf_brain, skull * 7)
        head = np.where(skull_brain > 0, skull_brain, skin * 8)

        # Final argmax-based kernel correction
        result = kernel_correction(head, sigma=0.05, theta=0.3, filter_size=5)

        # tES LAB 소프트웨어에서 사용하는 고유 라벨 값
        if simplify:
            result[result == 1] = 9  # gray matter
            result[result == 2] = 10  # white matter
            result[result == 3] = 9
            result[result == 4] = 10
            result[result == 5] = 11
            result[result == 6] = 11  # csf + ventricles

        return result

    def _sequential_postprocess_simplified(
        self,
        label: np.array,
        cover_wm: bool = True,
        simplify: bool = False,
        layer_gap: int = 2,
    ) -> np.array:
        """Sequential postprocessing of mri segmentation results.
            with simplified 6-labels (without skull, skin)

        Args:
            label (np.ndarray): argmaxed segmentation results in NEUROPHET's protocol
            cover_wm (bool): If specified, try to cover white matter with gray matter
            simplify (bool): If specified, simplify label
            layer_gap (int): number of iterations for in_out_correction to
                cover internal tissues with external tissues

        Returns:
            label: post processed result in  NEUROPHET's protocol.
        """
        # remove skull and skin
        label[label > 6] = 0

        # Layering Order:
        # Brain -> CSF -> Skull -> Scalp
        brain = self._brain_processing(label, cover_wm)

        # Extract CSF covering brain
        csf = (label != 0) & (label < 7)
        csf = binary_fill_holes(csf)
        csf = get_largest_label(csf)

        # Cover inner region with outter region
        csf = in_out_correction(csf, brain, iterations=layer_gap)

        # Start Assembling all Brain, Skull and Skin
        csf_brain = np.where(brain > 0, brain, csf * 6)

        # Final argmax-based kernel correction
        result = kernel_correction(csf_brain, sigma=0.05, theta=0.3, filter_size=5)

        if simplify:
            result[result == 3] = 1
            result[result == 4] = 2
            result[result == 5] = 3
            result[result == 6] = 3  # csf + ventricles

        return result
