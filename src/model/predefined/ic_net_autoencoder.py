from typing import Optional, List

import tensorflow as tf

from src.model.layers.conv import bottleneck_conv2d
from src.model.losses.cascade import reconstruction_loss
from src.model.network import NetworkOutput
from src.model.predefined.ic_net_backbone import ICNetBackbone


class ICNetAutoEncoder(ICNetBackbone):

    def __init__(self,
                 output_classes: int,
                 image_mean: Optional[List[float]] = None,
                 ignore_labels: Optional[List[int]] = None,
                 config: Optional[dict] = None):
        super().__init__(
            output_classes=3,
            image_mean=image_mean,
            ignore_labels=None,
            config=config)

    def training_pass(self, x: tf.Tensor, y: tf.Tensor) -> tf.Operation:
        if self._image_mean is not None:
            x -= self._image_mean
            y -= self._image_mean
        x = tf.math.divide(x, 255.0)
        y = tf.math.divide(y, 255.0)
        nodes_to_return = ['conv_sub4', 'conv_sub2', 'conv6_cls']
        model_output = self.feed_forward(
            x=x,
            is_training=True,
            nodes_to_return=nodes_to_return)
        conv_sub4 = model_output['conv_sub4']
        conv_sub2 = model_output['conv_sub2']
        conv6_cls = model_output['conv6_cls']
        conv_sub4_cls = bottleneck_conv2d(
            x=conv_sub4,
            num_filters=self._output_classes,
            activation=None,
            name='conv_sub4_cls'
        )
        conv_sub2_cls = bottleneck_conv2d(
            x=conv_sub2,
            num_filters=self._output_classes,
            activation=None,
            name='conv_sub2_cls'
        )
        cls_outputs = [conv_sub4_cls, conv_sub2_cls, conv6_cls]
        cls_weights = [self._labmda_1, self._lambda_2, self._lambda_3]
        return reconstruction_loss(
            cascade_output_nodes=cls_outputs,
            y=y,
            loss_weights=cls_weights)

    def infer(self, x: tf.Tensor) -> NetworkOutput:
        y_dash = super().infer(x)
        y_dash = y_dash * 255.0
        if self._image_mean is not None:
            y_dash += self._image_mean
        return tf.cast(y_dash, dtype=tf.uint8)
