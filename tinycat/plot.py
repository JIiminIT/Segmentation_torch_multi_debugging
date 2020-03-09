"""Image plotting interface using matplotlib.pyplot """


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


__all__ = ["NEUROPHET_COLORMAP_10", "convert_lbl_to_nph_colormap", "plot_volume_2d"]


# http://neurophet.iptime.org:30002/issues/884
COLORMAP_10 = {
    0: [0, 0, 0],
    1: [76, 43, 255],  # Cerebral GM
    2: [85, 170, 255],  # Cerebral WM
    3: [120, 26, 227],  # Cerebellar GM
    4: [199, 151, 255],  # Cerebellar WM
    5: [170, 255, 255],  # Ventricles
    6: [44, 45, 55],  # CSF
    7: [227, 227, 227],  # Skull
    8: [90, 104, 116],  # Skin
    13: [255, 0, 0],  # Stroke infarct
}

NEUROPHET_COLORMAP_10 = mpl.colors.ListedColormap(
    np.array(list(COLORMAP_10.values())) / 255.0
)


def convert_lbl_to_nph_colormap(lbl):
    """convert label data to nph colormap

    Args:
        lbl (np.ndarray): array including label data
    
    Raises:
        KeyError: label is in compatible
    
    Returns:
        ListedColormap: matplotlib colormap object
        zeros (np.ndarray): corrected label
    """

    uniques = np.unique(lbl)
    lbl_list = []
    zeros = np.zeros_like(lbl)

    try:
        for i, unique in enumerate(uniques):
            lbl_list.append(COLORMAP_10[unique])
            zeros[lbl == unique] = i
    except KeyError:
        raise KeyError("label provided is incompatible")

    return mpl.colors.ListedColormap(np.array(lbl_list) / 255.0), zeros


def plot_volume_2d(
    x,
    index=None,
    index_type="median",
    plain="x",
    colormap="nipy_spectral",
    normalize=True,
    filename=None,
):
    """plot volume to 2d slices

    Args:
        x (np.ndarray): 3-dimensional image array
        index (int, optional): Defaults to None. slice index for array indexing
        index_type (str, optional): Defaults to 'median'. Choose between ['median']
        plain (str, optional): Defaults to 'x'. set plain to visualize
        colormap (str, optional): Defaults to 'nipy_spectral'. colormap for intensity visualization
        normalize (bool, optional): Defaults to True, normalize data if specified
        filename ([type], optional): Defaults to None. If specified, save figure to directory
    """
    assert len(x.shape) == 3

    if index is None:
        if index_type == "median":
            index = np.array(x.shape) // 2

    if isinstance(index, int):
        index = (index, index, index)

    if plain == "x":
        scene = x[index[0], :, :]
    if plain == "y":
        scene = x[:, index[1], :]
    if plain == "z":
        scene = x[:, :, index[2]]

    if colormap == "neurophet":
        try:
            colormap, scene = convert_lbl_to_nph_colormap(scene)
        except KeyError:
            colormap = "gray"

    if filename is not None:
        plt.imsave(filename, scene, cmap=colormap)
    else:
        plt.imshow(scene, cmap=colormap)
        plt.show()
