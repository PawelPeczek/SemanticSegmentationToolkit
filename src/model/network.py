import logging
from abc import abstractmethod
from typing import Optional, Union, Dict, List, Set

import tensorflow as tf


logger = logging.getLogger(__file__)


class MissingNodeError(Exception):
    pass


BlockOutput = Union[tf.Tensor, List[tf.Tensor]]
NetworkOutput = Union[BlockOutput, Dict[str, BlockOutput]]
RequiredNodes = Optional[List[str]]


class Network:

    class Block:

        @classmethod
        def output_registered(cls, node_name: str):

            def decorator(layer_fun):

                def register_wrapper(_self, *args, **kwargs):
                    out_node = layer_fun(_self, *args, **kwargs)
                    _self._register_output(
                        node=out_node,
                        node_name=node_name)
                    return out_node

                return register_wrapper

            return decorator

    MAIN_OUTPUT_NAME = 'out'
    _MISSING_NODE_ERROR_MSG = 'Node name(s) required as output not present.'

    def __init__(self,
                 output_classes: int,
                 image_mean: Optional[List[float]] = None,
                 ignore_labels: Optional[List[int]] = None,
                 config: Optional[dict] = None):
        self._output_classes = output_classes
        self._ignore_labels = ignore_labels
        self._output_nodes = {}
        self._config = config
        self._image_mean = None
        if image_mean is not None and len(image_mean) is 3:
            mean_tensor = tf.convert_to_tensor(
                image_mean,
                dtype=tf.float32)
            self._image_mean = tf.expand_dims(mean_tensor, axis=0)

    @abstractmethod
    def feed_forward(self,
                     x: tf.Tensor,
                     is_training: bool = True,
                     nodes_to_return: RequiredNodes = None) -> NetworkOutput:
        """
        This method should allow the caller to get feed-forward network
        result - with possibility to obtain any registered node which may be
        useful when using the model later on. In particular - this method
        should be invoked by training_pass() in order to get nodes
        required to compute training loss.
        """
        raise NotImplementedError('This method must be implemented in '
                                  'derived class.')

    @abstractmethod
    def training_pass(self, x: tf.Tensor, y: tf.Tensor) -> tf.Operation:
        """
        Method invoked while training, should return error operation in
        order to make it possible to train model with chosen optimizer.
        Should use feed_forward() method to obtain desired network output.
        """
        raise NotImplementedError('This method must be implemented in '
                                  'derived class.')

    @abstractmethod
    def infer(self, x: BlockOutput) -> NetworkOutput:
        """
        Method used while inference from trained model.
        """
        raise NotImplementedError('This method must be implemented in '
                                  'derived class.')

    def restore_checkpoint(self,
                           checkpoint_path: str,
                           session: tf.Session) -> None:
        var_to_restore = self._get_variables_to_restore()
        saver = tf.train.Saver(var_list=var_to_restore)
        saver.restore(sess=session, save_path=checkpoint_path)

    def _get_variables_to_restore(self) -> List[tf.Variable]:
        not_to_restore = self._get_variables_name_not_to_restore()
        var_to_restore = [
            v for v in tf.global_variables()
            if v.name.split('/')[0] not in not_to_restore
        ]
        return var_to_restore

    def _get_variables_name_not_to_restore(self) -> Set[str]:
        return set()

    def _register_output(self,
                         node: tf.Tensor,
                         node_name: str) -> None:
        if node_name in self._output_nodes:
            logger.warning(f'Loss node with name {node_name} already exists. '
                           f'It will be overwritten by default.')
        self._output_nodes[node_name] = node

    def _construct_output(self,
                          feedforward_output: BlockOutput,
                          nodes_to_return: RequiredNodes) -> NetworkOutput:
        if nodes_to_return is None:
            return feedforward_output
        if self._output_impossible_to_construct(nodes_to_return):
            raise MissingNodeError(Network._MISSING_NODE_ERROR_MSG)
        output = {Network.MAIN_OUTPUT_NAME: feedforward_output}
        for node_name in nodes_to_return:
            output[node_name] = self._output_nodes[node_name]
        return output

    def _output_impossible_to_construct(self,
                                        nodes_to_return: RequiredNodes) -> bool:
        return any([node not in self._output_nodes for node in nodes_to_return])
