from functools import reduce
from typing import List, Optional, Tuple

import tensorflow as tf

from src.model.layers.interpolation import resize_nn


LabelsToIgnore = Optional[List[int]]


def cascade_loss(cascade_output_nodes: List[tf.Tensor],
                 y: tf.Tensor,
                 weight_decay: float = 0.0,
                 cascade_loss_weights: Optional[List[float]] = None,
                 labels_to_ignore: LabelsToIgnore = None) -> tf.Operation:
    losses = []
    for output_node in cascade_output_nodes:
        loss = _compute_branch_loss(
            output_node=output_node,
            y=y,
            ignored_labels=labels_to_ignore)
        loss = tf.reduce_mean(loss)
        losses.append(loss)
    if cascade_loss_weights is not None:
        losses = _weight_losses(
            losses=losses,
            cascade_loss_weights=cascade_loss_weights)
    l2_loss = _create_l2_loss(weight_decay=weight_decay)
    return reduce(lambda acc, x: acc + x, losses) + l2_loss


def _compute_branch_loss(output_node: tf.Tensor,
                         y: tf.Tensor,
                         ignored_labels: LabelsToIgnore = None) -> tf.Operation:
    target_size = output_node.shape[1:3]
    resized_y = y
    if target_size != resized_y.shape[1:3]:
        resized_y = _resize_gt(gt=resized_y, target_size=target_size)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=output_node,
        labels=resized_y)
    mask = prepare_loss_mask(resized_y, to_ignore=ignored_labels)
    if mask is not None:
        loss = tf.math.multiply(loss, mask)
    return tf.reduce_mean(loss)


def _resize_gt(gt: tf.Tensor,
               target_size: Tuple[tf.Dimension, tf.Dimension]) -> tf.Tensor:
    expanded_y = tf.expand_dims(gt, axis=-1)
    resized_y = resize_nn(
        x=expanded_y,
        height=target_size[0],
        width=target_size[1])
    resized_y = tf.squeeze(resized_y, axis=-1)
    return tf.cast(resized_y, dtype=tf.int32)


def prepare_loss_mask(resized_y: tf.Tensor,
                      to_ignore: LabelsToIgnore = None) -> Optional[tf.Tensor]:
    mask = None
    for label_to_ignore in to_ignore:
        label_mask = tf.cast(
            tf.math.not_equal(resized_y, label_to_ignore),
            tf.float32)
        if mask is None:
            mask = label_mask
        else:
            mask = tf.math.multiply(mask, label_mask)
    return mask


def _weight_losses(losses: List[tf.Operation],
                   cascade_loss_weights: List[float]) -> List[tf.Operation]:
    assert len(cascade_loss_weights) == len(losses)
    losses_with_weights = list(zip(cascade_loss_weights, losses))
    return list(map(lambda x: x[0] * x[1], losses_with_weights))


def _create_l2_loss(weight_decay: float) -> tf.Operation:
    l2_losses = [
        weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables()
        if 'kernel' in v.name
    ]
    return tf.add_n(l2_losses)
