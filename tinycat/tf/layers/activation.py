from __future__ import division
import tensorflow as tf


def ncrelu(x, name="NCRelu"):
    """
    Activation function from DiracNets: 
    Training Very Deep Neural Networks Without Skip-Connections

    https://github.com/szagoruyko/diracnets
    https://arxiv.org/abs/1706.00388
    
    Args:
        x (tensor): tensor object
        name (str, optional): Defaults to "NCRelu".
    """

    return tf.concat(
        [tf.clip_by_value(x, clip_value_min=0), tf.clip_by_value(x, clip_value_max=0)],
        axis=-1,
        name=name,
    )


def swish(x, name="Swish"):
    """ Swish: A Self-gated activation function

    Original Paper:
        https://arxiv.org/pdf/1710.05941.pdf
    """
    with tf.name_scope(name):
        return x * tf.sigmoid(x, "Sigmoid")


def prelu(x, name="PRelu"):
    """Parametric ReLU"""

    with tf.name_scope(name):
        alpha = tf.get_variable(
            "alpha",
            x.get_shape()[-1],
            initializer=tf.constant_initializer(0.01),
            dtype=tf.float32,
        )
        return tf.maximum(0.0, x) + tf.minimum(0.0, alpha * x)


def maxout(x, num_param=5, name="maxout"):
    with tf.name_scope(name):
        output = []
        for i in range(num_param):
            name = "w_%d" % i
            w = tf.get_variable(
                name,
                x.get_shape()[-1],
                initializer=tf.constant_initializer(1.0 * (i - num_param / 2)),
            )
            name = "b_%d" % i
            b = tf.get_variable(
                name,
                x.get_shape()[-1],
                initializer=tf.constant_initializer(i - num_param / 2),
            )
            out = x * w + b
            output.append(out)

        ret = tf.reduce_max(output, 0)
        print(ret.get_shape())
        return ret
