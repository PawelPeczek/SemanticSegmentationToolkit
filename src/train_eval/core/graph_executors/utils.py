from typing import Tuple, List, Callable, Union, Optional

import tensorflow as tf

from src.model.losses.cascade import prepare_loss_mask
from src.model.network import Network

Gradient = Union[Tuple[tf.Operation, tf.Variable], tf.Operation]
Gradients = List[Gradient]
AvgGradOperations = List[Tuple[tf.Operation, tf.Variable]]
OptimizationStep = Callable[[tf.Operation], Gradient]
LabelsToIgnore = Optional[List[int]]

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']


class ValidationOperation:

    def __init__(self,
                 iterator: tf.data.Iterator,
                 mean_iou: tf.Operation,
                 mean_iou_update: tf.Operation):
        self.__iterator = iterator
        self.__mean_iou = mean_iou
        self.__mean_iou_update = mean_iou_update

    @property
    def iterator(self) -> tf.data.Iterator:
        return self.__iterator

    @property
    def mean_iou(self) -> tf.Operation:
        return self.__mean_iou

    @property
    def mean_iou_update(self) -> tf.Operation:
        return self.__mean_iou_update


class ValidationOperations:

    def __init__(self,
                 training_set_evaluation: ValidationOperation,
                 test_set_evaluation: ValidationOperation):
        self.__training_set_evaluation = training_set_evaluation
        self.__test_set_evaluation = test_set_evaluation

    @property
    def training_set_evaluation(self) -> ValidationOperation:
        return self.__training_set_evaluation

    @property
    def test_set_evaluation(self) -> ValidationOperation:
        return self.__test_set_evaluation


class SessionOperations:

    def __init__(self,
                 iterator: tf.data.Iterator,
                 loss_operations: tf.Operation,
                 gradient_update: tf.Operation,
                 validation_operations: Optional[ValidationOperations]):
        self.__iterator = iterator
        self.__loss_operation = loss_operations
        self.__gradient_update = gradient_update
        self.__validation_operations = validation_operations

    @property
    def iterator(self) -> tf.data.Iterator:
        return self.__iterator

    @property
    def loss_operations(self) -> tf.Operation:
        return self.__loss_operation

    @property
    def gradient_update(self) -> tf.Operation:
        return self.__gradient_update

    @property
    def validation_operations(self) -> ValidationOperations:
        return self.__validation_operations


class MultiGPUOperations:

    def __init__(self, gradient: Gradient, loss_operation: tf.Operation):
        self.__gradient = gradient
        self.__loss_operation = loss_operation

    @property
    def gradient(self) -> Gradient:
        return self.__gradient

    @property
    def loss_operation(self) -> tf.Operation:
        return self.__loss_operation


def assign_to_device(target_device: str,
                     ps_device: str = '/cpu:0') -> tf.NodeDef:

    def assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return target_device

    return assign


def get_validation_operation(iterator: tf.data.Iterator,
                             model: Network,
                             num_classes: int,
                             labels_to_ignore: LabelsToIgnore = None) -> ValidationOperation:
    x, y = iterator.get_next()
    prediction = model.infer(x)
    weights = None
    if labels_to_ignore is not None:
        weights = prepare_loss_mask(y, labels_to_ignore)
    mean_iou, mean_iou_update = tf.metrics.mean_iou(
        labels=y,
        predictions=prediction,
        num_classes=num_classes,
        weights=weights)
    return ValidationOperation(
        iterator=iterator,
        mean_iou=mean_iou,
        mean_iou_update=mean_iou_update)


def evaluate_miou(session: tf.Session,
                  validation_operation: ValidationOperation) -> float:
        session.run(tf.initializers.variables(tf.local_variables()))
        session.run(validation_operation.iterator.initializer)
        try:
            while True:
                session.run(validation_operation.mean_iou_update)
        except tf.errors.OutOfRangeError:
            return session.run(validation_operation.mean_iou)


def get_gradient(optimizer: tf.train.Optimizer,
                 loss_operation: tf.Operation) -> Gradient:
    return _make_optimization_step(optimizer.compute_gradients, loss_operation)


def minimize_loss(optimizer: tf.train.Optimizer,
                  loss_operation: tf.Operation) -> Gradient:
    return _make_optimization_step(optimizer.minimize, loss_operation)


def _make_optimization_step(step: OptimizationStep,
                            loss_operation: tf.Operation) -> Gradient:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gradient = step(loss_operation)
    return gradient


def average_gradients(gradients: Gradients) -> AvgGradOperations:
    """
    The function originally comes from:
    https://github.com/jhui/deep_learning/

    Calculate the average gradient for each shared variable across
    all towers. Notee that this function provides a synchronization
    point across all towers.

    Args:
      gradients: List of lists of (gradient, variable) tuples.
                 The outer list is over individual gradients.
                 The inner list is over the gradient calculation
                 for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been
       averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*gradients):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def average_loss(loss: List[tf.Operation]) -> tf.Operation:
    loss = tf.stack(loss, axis=0)
    return tf.reduce_mean(loss, axis=0)
