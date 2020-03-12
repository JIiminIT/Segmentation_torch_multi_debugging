import tensorflow as tf


def conv2d_block(input_, n_filters, training, name, pool=True, activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F,
                (3, 3),
                activation=None,
                padding="SAME",
                name="conv_{}".format(i + 1),
            )
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1), fused=True
            )
            net = activation(net, name="activ{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name)
        )

        return net, pool


def upsampling2d(tensor, name, size=(2, 2)):
    """Upsample/Rescale `tensor` by size

    Args:
        tensor (4-D Tensor): (N, H, W, C)
        name (str): name of upsampling operations
        size (tuple, optional): (height_multiplier, width_multiplier)
            (default: (2, 2))

    Returns:
        output (4-D Tensor): (N, h_multiplier * H, w_multiplier * W, C)
    """
    h, w, _ = tensor.get_shape().as_list()[1:]

    h_multi, w_multi = size
    target_h = h * h_multi
    target_w = w * h_multi

    return tf.image.resize_nearest_neighbor(
        tensor, (target_h, target_w), name="upsample_{}".format(name)
    )


def upsample2d_concat(net1, net2, name):
    """Upsample `inputA` and concat with `input_B`

    Args:
        net1 (4-D Tensor): (N, H, W, C)
        net2 (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation

    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    upsample = upsampling2d(net1, size=(2, 2), name=name)

    return tf.concat([upsample, net2], axis=-1, name="concat_{}".format(name))


def conv3d_block(input_, n_filters, training, name, pool=True, activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv3d(
                net,
                F,
                (3, 3, 3),
                activation=None,
                padding="SAME",
                name="conv_{}".format(i + 1),
            )
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1)
            )
            net = activation(net, name="activ{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling3d(
            net, (2, 2, 2), strides=(2, 2, 2), name="pool_{}".format(name)
        )

        return net, pool


def upsampling3d(tensor, name, size=(2, 2, 2)):
    """Upsample/Rescale `tensor` by size

    Args:
        tensor (5-D Tensor): (N, H, W, D, C)
        name (str): name of upsampling operations
        size (tuple, optional): (height_multiplier, width_multiplier)
            (default: (2, 2, 2))

    Returns:
        output (5-D Tensor): (N, h_multiplier * H, w_multiplier * W, C)
    """
    h, w, d, _ = tensor.get_shape().as_list()[1:]

    h_multi, w_multi, d_multi = size
    target_h = h * h_multi
    target_w = w * h_multi
    target_d = d * d_multi

    return tf.image.resize_nearest_neighbor(
        tensor, (target_h, target_w, target_d), name="upsample_{}".format(name)
    )


def upsample3d_concat(net1, net2, name):
    """Upsample `inputA` and concat with `input_B`

    Args:
        net1 (5-D Tensor): (N, H, W, D, C)
        net2 (5-D Tensor): (N, 2*H, 2*H, 2*D, C2)
        name (str): name of the concat operation

    Returns:
        output (5-D Tensor): (N, 2*H, 2*W, 2*D, C + C2)
    """

    upsample = upsampling_3d(net1, size=(2, 2, 2), name=name)

    return tf.concat([upsample, net2], axis=4, name="concat_{}".format(name))
