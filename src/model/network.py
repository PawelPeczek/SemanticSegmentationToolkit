import logging
from abc import abstractmethod
from enum import Enum
from typing import Optional, Union, Dict, List

import tensorflow as tf


logger = logging.getLogger(__file__)


class NetworkMode(Enum):
    TRAIN = 0
    INFERENCE = 1


class MissingNodeError(Exception):
    pass


BlockOutput = Union[tf.Tensor, List[tf.Tensor]]
NetworkOutput = Union[BlockOutput, Dict[str, BlockOutput]]
RequiredNodes = Optional[List[str]]


class Network:

    class Block:

        def output_registered(self, node_name: str):

            def decorator(layer_fun):

                def register_wrapper(_self, *args, **kwargs):
                    out_node = layer_fun(_self, *args, **kwargs)
                    _self._register_output(
                        node=out_node,
                        node_name=node_name)

                return register_wrapper

            return decorator

    MAIN_OUTPUT_NAME = 'out'
    _MISSING_NODE_ERROR_MSG = 'Node name(s) required as output not present.'

    def __init__(self, output_classes: int):
        self._output_classes = output_classes
        self._output_nodes = {}

    @abstractmethod
    def feed_forward(self,
                     x: tf.Tensor,
                     is_training: bool = True,
                     nodes_to_return: RequiredNodes = None) -> NetworkOutput:
        raise RuntimeError('This method must be implemented in derived class.')

    @abstractmethod
    def infer(self, x: BlockOutput) -> NetworkOutput:
        raise RuntimeError('This method must be implemented in derived class.')

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
