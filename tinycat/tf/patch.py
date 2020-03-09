import tensorflow as tf


__all__ = ["random_crop_3d"]


def random_crop_3d(image, label, patch_size, shape):
    """
    Randomly crops 3-dimensional data within uniform distribution

    :param image: image data to crop
    :param label: label data to crop
    :param size: crop window size
    :param shape: shape of the original image
    :returns: cropped image and label
    """

    if shape is None:
        shape = []
        shape.append(image.shape[0].value)
        shape.append(image.shape[1].value)
        shape.append(image.shape[2].value)

    x = tf.random_uniform([], minval=0, maxval=shape[0] - patch_size[0], dtype=tf.int64)
    y = tf.random_uniform([], minval=0, maxval=shape[1] - patch_size[1], dtype=tf.int64)
    z = tf.random_uniform([], minval=0, maxval=shape[2] - patch_size[2], dtype=tf.int64)
    image = tf.slice(image, [x, y, z], patch_size)
    label = tf.slice(label, [x, y, z], patch_size)
    return image, label
