from typing import Optional, List

import tensorflow as tf

from src.model.layers.conv import bottleneck_conv2d
from src.model.losses.cascade import cascade_loss
from src.model.network import NetworkOutput, RequiredNodes
from src.model.predefined.ic_net_backbone import ICNetBackbone


class ICNet(ICNetBackbone):

    def __init__(self,
                 output_classes: int,
                 image_mean: Optional[List[float]] = None,
                 ignore_labels: Optional[List[int]] = None,
                 config: Optional[dict] = None):
        super().__init__(
            output_classes=output_classes,
            image_mean=image_mean,
            ignore_labels=ignore_labels,
            config=config)
        self.__weight_decay = 0.0
        if config is not None and 'weight_decay' in config:
            self.__weight_decay = config['weight_decay']

    def feed_forward(self,
                     x: tf.Tensor,
                     is_training: bool = True,
                     nodes_to_return: RequiredNodes = None) -> NetworkOutput:
        conv6_interp = super().feed_forward(
            x=x,
            is_training=is_training
        )
        out = tf.math.argmax(conv6_interp, axis=3, output_type=tf.dtypes.int32)
        return self._construct_output(
            feedforward_output=out,
            nodes_to_return=nodes_to_return)

    def training_pass(self, x: tf.Tensor, y: tf.Tensor) -> tf.Operation:
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
        return cascade_loss(
            cascade_output_nodes=cls_outputs,
            y=y,
            weight_decay=self.__weight_decay,
            loss_weights=cls_weights,
            labels_to_ignore=self._ignore_labels)
