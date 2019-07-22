from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Tuple, Optional, Union

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.common.CityScapesIteratorFactory import CityScapesIteratorFactory, IteratorType
from src.model.network import Network
from src.model.utils import ModelFactory
from src.train_eval.core.optimizers.OptimizerWrapperFactory import OptimizerWrapperFactory
from src.train_eval.core.persistence.PersistenceManager import PersistenceManager


class GraphExecutor(ABC):

    def __init__(self,
                 descriptive_name: str,
                 config: GraphExecutorConfigReader):
        self._config = config
        self._descriptive_name = descriptive_name
        self._model = self.__construct_model()
        self._iterator_factory = CityScapesIteratorFactory(config)
        self._persistence_manager = self._get_persistence_manager()

    @abstractmethod
    def execute(self) -> None:
        raise NotImplementedError('This method must be implemented.')

    @abstractmethod
    def _get_iterator_type(self) -> IteratorType:
        raise NotImplementedError('This method must be implemented.')

    @abstractmethod
    def _get_persistence_manager(self) -> PersistenceManager:
        raise NotImplementedError('This method must be implemented.')

    def _get_session_config(self) -> tf.ConfigProto:
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        return config

    def _get_iterator(self) -> tf.data.Iterator:
        iterator_type = self._get_iterator_type()
        return self._iterator_factory.get_iterator(iterator_type)

    def _get_data_point(self) -> Tuple[tf.Tensor, tf.Tensor]:
        iterator = self._get_iterator()
        return iterator.get_next()

    def _get_network(self) -> Network:
        return self._model

    def _initialize_optimizer(self) -> tf.train.Optimizer:
        optimizer_wrapper_factory = OptimizerWrapperFactory()
        optimizer_wrapper = optimizer_wrapper_factory.assembly(
            self._config.optimizer_options)
        return optimizer_wrapper.get_optimizer()

    def __construct_model(self) -> Network:
        model_factory = ModelFactory()
        return model_factory.assembly(
            self._config.model_name,
            self._config.num_classes,
            self._config.get_or_else('ignore_labels', None),
            self._config.get_or_else('model_config', None))
