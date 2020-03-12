import tensorflow as tf


def l1_loss(input_, target_, lamb=1.0, name="l1_loss"):
    with tf.name_scope(name):
        lamb = tf.convert_to_tensor(lamb)
        loss = tf.multiply(tf.reduce_mean(tf.abs(input_ - target_)), lamb, name="loss")
        return loss


def l2_loss(input_, target_, lamb=1.0, name="l2_loss"):
    with tf.name_scope(name):
        lamb = tf.convert_to_tensor(lamb)
        loss = tf.multiply(
            tf.reduce_mean(tf.square(input_ - target_)), lamb, name="loss"
        )
        return loss


def cross_entropy_loss(logits, labels, lamb=1.0, name="ce_loss"):
    with tf.name_scope(name):
        lamb = tf.convert_to_tensor(lamb)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        loss = tf.multiply(lamb, tf.reduce_mean(cross_entropy), name="loss")
        return loss


def _weight_decay(_weight_decay_rate=0.0001):
    costs = []
    for var in tf.trainable_variables():
        costs.append(tf.nn.l2_loss(var))
    return tf.multiply(_weight_decay_rate, tf.add_n(costs))


def pixel_wise_l1_loss(input_, target_, lamb=1.0, name="pixel_l1"):
    return l1_loss(input_, target_, lamb, name)


def pixel_wise_l2_loss(input_, target_, lamb=1.0, name="pixel_l2"):
    return l2_loss(input_, target_, lamb, name)


def _flatten(input_, name="flatten"):
    vec_dim = input_.get_shape()[1:]
    n = vec_dim.num_elements()
    with tf.name_scope(name):
        return tf.reshape(input_, [-1, n])


def pixel_wise_cross_entropy(input_, target_, lamb=1.0, name="pixel_ce"):
    flat = _flatten(input_)
    return cross_entropy_loss(flat, target_, lamb, name)


def pixel_wise_sparse_softmax_entropy(logits, y):
    _y = tf.argmax(y, axis=3)
    _xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=_y, logits=logits)
    _cost = tf.reduce_mean(_xent) + _weight_decay()
    return _cost


def _pixel_wise_softmax(output_map):
    dim = len(output_map.get_shape().as_list()) - 1
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, dim, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1] * dim + [tf.shape(output_map)[dim]]))
    return tf.div(exponential_map, tensor_sum_exp)


# Dice Series
def loss_dice_coef(logits, y):
    eps = 1e-5
    prediction = _pixel_wise_softmax(logits)
    prediction = tf.clip_by_value(prediction, eps, 9.999)
    intersection = tf.reduce_sum(prediction * y)
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y)
    dice = -(2 * intersection / union)
    return dice


def log_dice_coef(logits, y):
    return tf.log(loss_dice_coef(logits=logits, y=y))


def loss_dice_coef_weighted(logits, y, weight=2):
    eps = 1e-10
    prediction = _pixel_wise_softmax(logits)
    _, _, split_3 = tf.split(prediction, num_or_size_splits=3, axis=3)
    zeros_wt = tf.ones(
        [
            prediction.get_shape().as_list()[0],
            prediction.get_shape().as_list()[1],
            prediction.get_shape().as_list()[2],
            prediction.get_shape().as_list()[3] - 1,
        ]
    )
    weighted = tf.concat([zeros_wt, tf.multiply(split_3, weight)], axis=3)
    prediction = tf.multiply(prediction, weighted)
    prediction = tf.clip_by_value(prediction, eps, 0.9999999)
    intersection = tf.reduce_sum(prediction * y)
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y)
    dice = -(2 * intersection / union)
    return dice


def loss_dice_coef_ent(logits, y):
    eps = 1e-10
    prediction = _pixel_wise_softmax(logits)
    prediction = tf.clip_by_value(prediction, eps, 1.0)
    prediction = tf.log(prediction)
    intersection = tf.reduce_sum(prediction * y)
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y)
    dice = -(2 * intersection / tf.log(union))
    return dice


def loss_jaccard_coef(logits, y):
    eps = 1e-10
    prediction = _pixel_wise_softmax(logits)
    prediction = tf.clip_by_value(prediction, eps, 0.9999999)
    intersection = tf.reduce_sum(prediction * y)
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y)
    jaccard = -(intersection / union)
    return jaccard


def weighted_jaccard_coef(logits, y, weights):
    eps = 1e-10
    logits = weights * logits
    prediction = _pixel_wise_softmax(logits)
    prediction = tf.clip_by_value(prediction, eps, 0.9999999)
    intersection = tf.reduce_sum(prediction * y)
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y)
    jaccard = -(intersection / union)
    return jaccard


def soft_dice_loss(y_pred, y_true):
    """
    :param y_pred: softmax output of shape (num_samples, num_classes)
    :param y_true: one hot encoding of target (shape= (num_samples, num_classes))
    :return:
    """
    intersect = tf.reduce_sum(y_pred * y_true, 0)
    denominator = tf.reduce_sum(y_pred, 0) + tf.reduce_sum(y_true, 0)
    dice_scores = tf.constant(2) * intersect / (denominator + tf.constant(1e-6))
    return dice_scores


def mixed_loss(logits, y):
    return cross_entropy_loss(logits, y) + soft_dice_loss(logits, y)


def asd(predicts, annots):
    """
    distance is calculated as euclidean distnace
    :param predicts: MxN matrix, M = # of batch, N:dimension
    :param annots: MxN matrix, M = # of batch, N:dimension
    :return:
    """
    r = tf.reduce_sum(predicts * annots, 1)
    r = tf.reshape(r, [-1, 1])
    sds = tf.reduce_sum(
        D == r - 2 * tf.matmul(predicts, tf.transpose(predicts)) + tf.transpose(r)
    )
    asd_loss = tf.divide(sds, tf.reduce_sum(tf.sqrt(tf.square(predicts))))
    return asd_loss


def assd_loss(predicts, annots):
    asd1 = asd(predicts, annots)
    asd2 = asd(annots, predicts)
    numerator = tf.add(asd1, asd2)
    cost = tf.divide(numerator, 2.0)
    return cost


def hdhd_loss(predicts, annots):
    """
    Hausdorf distance between two 3D volume
    :param predicts: prediction matrix - predictions [batch,width,height,depth]
    :param annots: ground-truth - annotation [batch, ~]
    :return:cost
    """
    a = tf.reshape(predicts, [-1])
    b = tf.reshape(annots, [-1])
    dota = tf.matmul(a, a, transpose_a=False, transpose_b=True)
    dotb = tf.matmul(b, b, transpose_a=False, transpose_b=True)
    d_mat = tf.sqrt(
        dota + dotb - 2 * tf.matmul(a, b, transpose_a=False, transpose_b=True)
    )
    hd_cost = tf.reduce_max(
        tf.reduce_max(tf.reduce_min(d_mat, axis=0)),
        tf.reduct_max(tf.reduce_min(d_mat, axis=1)),
    )
    return hd_cost
