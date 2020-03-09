from __future__ import absolute_import

import tensorflow as tf


def accuracy(ground_truth, predictions):
    with tf.variable_scope("calc_accuracy"):
        return 100 * tf.reduce_mean(
            tf.cast(tf.equal(predictions, ground_truth), tf.float32)
        )


def confusion_matrix(logits, y):
    logits = tf.one_hot(tf.argmax(logits, axis=3), depth=y.get_shape().as_list()[3])
    tp = tf.reduce_sum(tf.multiply(logits, y))
    fp = tf.reduce_sum(tf.multiply(logits, 1 - y))
    fn = tf.reduce_sum(tf.multiply(1 - logits, y))
    tn = tf.reduce_sum(tf.multiply(1 - logits, 1 - y))
    return tp, fp, fn, tn


def dice_coef(logits, y, name="dice_coef"):
    with tf.variable_scope(name):
        eps = 1e-8
        tp, fp, fn, tn = confusion_matrix(logits, y)
        return tf.divide(2 * tp + eps, fp + fn + 2 * tp + eps)


def jaccard_coef(logits, y, name="jaccard_coef"):
    with tf.variable_scope(name):
        eps = 1e-8
        tp, fp, fn, tn = confusion_matrix(logits, y)
        return tf.divide(tp + eps, fp + fn + tp + eps)


def iou(y_pred, y_true):
    """Returns a (approx) IOU score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: IOU score
    """
    h, w, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, h * w])
    true_flat = tf.reshape(y_true, [-1, h * w])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = (
        tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-7
    )

    return tf.reduce_mean(intersection / denominator)
