"""영상 패치 크롭 함수를 모은 모듈"""
import numpy as np


__all__ = [
    "crop_center_3d",
    "crop_roi_3d",
    "median_crop_roi_3d",
    "get_gridded_patch_3d",
    "assemble_softmax_3d",
]


def crop_center_3d(img, cropx, cropy, cropz):
    """
    Crop 3-dimensional image from center.

    :param img: 3-dimensional image
    :param int cropx: size of cropped x
    :param int cropy: size of cropped y
    :param int cropz: size of cropped z
    :returns: cropped image
    """
    z, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    startz = z // 2 - (cropz // 2)
    return img[
        startz : startz + cropz, starty : starty + cropy, startx : startx + cropx
    ]


def crop_foreground_3d(img, edge_only=False):
    """ Crop 3-dimensional image into non-zero foreground

    Args:
        img (np.array):  3-dimensional array
        edge_only (bool, optional): only returns top and bottom edges

    Returns:
        img (np.array): cropped img
        edges (tuple(np.array, np.array)): top and bottom edge of img

    """
    true_points = np.argwhere(img)
    top_edge = true_points.min(axis=0)
    bottom_edge = true_points.max(axis=0)
    if edge_only:
        return top_edge, bottom_edge
    return img[
        top_edge[0] : bottom_edge[0] + 1,
        top_edge[1] : bottom_edge[1] + 1,
        top_edge[2] : bottom_edge[2] + 1,
    ]


def crop_roi_3d(img, length_only=False):
    """
    Crop 3-dimensional image into region-of-interests.

    :param img: 3-dimensional array
    :param bool length_only: only returns the length of each coordinates.
    :return: center cropped ROI image or length
    """
    z, y, x = np.where(img)
    x_len = np.max(x) - np.min(x)
    y_len = np.max(y) - np.min(y)
    z_len = np.max(z) - np.min(z)
    if length_only:
        return x_len, y_len, z_len
    return crop_center_3d(img, x_len, y_len, z_len)


def median_crop_roi_3d(img, x=None, y=None, z=None, pad=0, pad_constant=0, border=0):
    """Crop 3-dimensional image from the median of region-of-interest

    :param img: 3-dimensional array
    :param x: np.where(img)[2]
    :param y: np.where(img)[1]
    :param z: np.where(img)[0]
    :param int pad: size of pad
    :param int pad_constant: value of pad constant
    :param int border: width of border
    :return: median cropped image
    """

    img = np.pad(img, pad // 2, mode="constant", constant_values=pad_constant)

    if x is None:
        z, y, x = np.where(img)

    x_len = np.max(x) - np.min(x) + border
    y_len = np.max(y) - np.min(y) + border
    z_len = np.max(z) - np.min(z) + border
    x_med = np.median(x).astype(np.int)
    y_med = np.median(y).astype(np.int)
    z_med = np.median(z).astype(np.int)
    startx = x_med - (x_len // 2)
    starty = y_med - (y_len // 2)
    startz = z_med - (z_len // 2)
    return img[
        startz : startz + z_len, starty : starty + y_len, startx : startx + x_len
    ]


def get_gridded_patch_3d(
    data,
    patch_shape=(96, 96, 96),
    grid=None,
    grid_rate=1.5,
    border=(16, 16, 16),
    method="sum",
):
    """3-dimenisional grid patch generation with full size input.
    
    Args:
        data (np.ndarray): 3-dimensional array data
        patch_shape (tuple): shape of 3-dimensional array
        grid (np.array, optional): Defaults to None. grid array for linspacing
        grid_rate (float, optional): Defaults to 1.5. grid rate for linspacing
        border (tuple, optional): Defaults to 16. patch border to crop
        method (str, optional): Defaults to 'crop'. assemble method between ['crop', 'sum']
    
    Returns:
        np.array, np.array: patches and spacing
    """

    edge = data.shape
    patches = []

    if method == "crop":
        grid_patch_shape = np.array(patch_shape) - np.array(border) * 2
        grid = np.max(data.shape) // np.array(grid_patch_shape) * grid_rate

        # calucate spacing for each coordinate
        x_spacing = np.linspace(
            0, edge[0] - grid_patch_shape[0], grid[0], endpoint=True, dtype=np.int32
        )
        y_spacing = np.linspace(
            0, edge[1] - grid_patch_shape[1], grid[1], endpoint=True, dtype=np.int32
        )
        z_spacing = np.linspace(
            0, edge[2] - grid_patch_shape[2], grid[2], endpoint=True, dtype=np.int32
        )

        if border:
            x_width = -(edge[0] - x_spacing[-1] - patch_shape[0]) // 2
            y_width = -(edge[1] - y_spacing[-1] - patch_shape[1]) // 2
            z_width = -(edge[2] - z_spacing[-1] - patch_shape[2]) // 2

            if len(edge) == 4:
                pad_width = (
                    (x_width, x_width),
                    (y_width, y_width),
                    (z_width, z_width),
                    (0, 0),
                )

            if len(edge) == 3:
                pad_width = ((x_width, x_width), (y_width, y_width), (z_width, z_width))

            data = np.pad(data, pad_width, mode="constant")

    if method == "sum":
        # calucate spacing for each coordinate
        x_spacing = np.linspace(
            0, edge[0] - patch_shape[0], grid[0], endpoint=True, dtype=np.int32
        )
        y_spacing = np.linspace(
            0, edge[1] - patch_shape[1], grid[1], endpoint=True, dtype=np.int32
        )
        z_spacing = np.linspace(
            0, edge[2] - patch_shape[2], grid[2], endpoint=True, dtype=np.int32
        )

    # append 3-dimensional patches into list
    for i in x_spacing:
        for j in y_spacing:
            for k in z_spacing:
                patches.append(
                    data[
                        i : i + patch_shape[0],
                        j : j + patch_shape[1],
                        k : k + patch_shape[2],
                    ]
                )

    spacing = (x_spacing, y_spacing, z_spacing)  # set as tuple
    return patches, spacing


def assemble_softmax_3d(
    patches, spacing, orig_shape, n_classes, method="crop", border=(16, 16, 16)
):
    """assemble softmax array into original shape
    
    Args:
        patches (np.ndarray): iterable of each coordinate's spacing
        spacing (np.ndarray): iterable of softmax [[x, y, z, n_classes]]
        orig_shape (np.ndarray): shape of original data
        n_classes (int): number of classes
        method (str, optional): Defaults to 'crop'. assemble method between ['crop', 'sum']
        border (tuple, optional): Defaults to (16, 16, 16). crop border if method is crop
    
    Returns:
        np.array: assembled data
    """

    initial_array = np.zeros([orig_shape[0], orig_shape[1], orig_shape[2], n_classes])
    x_spacing = spacing[0]
    y_spacing = spacing[1]
    z_spacing = spacing[2]
    x_spacing[1:] += 1
    y_spacing[1:] += 1
    z_spacing[1:] += 1

    idx = 0

    if method == "sum":
        for x in x_spacing:
            for y in y_spacing:
                for z in z_spacing:
                    patch = patches[idx]
                    idx += 1

                    if x == x_spacing[-1]:
                        patch = patch[:-1, :, :, :]
                    if y == y_spacing[-1]:
                        patch = patch[:, :-1, :, :]
                    if z == z_spacing[-1]:
                        patch = patch[:, :, :-1, :]

                    initial_array[
                        x : x + patch.shape[0],
                        y : y + patch.shape[1],
                        z : z + patch.shape[2],
                    ] += patch
    if method == "crop":
        assert border

        for x in x_spacing:
            for y in y_spacing:
                for z in z_spacing:
                    patch = patches[idx]
                    idx += 1
                    patch = patch[
                        border[0] : -border[0],
                        border[1] : -border[1],
                        border[2] : -border[2],
                    ]

                    if x == x_spacing[-1]:
                        patch = patch[:-1, :, :, :]
                    if y == y_spacing[-1]:
                        patch = patch[:, :-1, :, :]
                    if z == z_spacing[-1]:
                        patch = patch[:, :, :-1, :]

                    initial_array[
                        x : x + patch.shape[0],
                        y : y + patch.shape[1],
                        z : z + patch.shape[2],
                    ] += patch

    return initial_array
