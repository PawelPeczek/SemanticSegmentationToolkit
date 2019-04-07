from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Tuple, Optional, Union
from src.dataset.common.CityScapesIteratorFactory import CityScapesIteratorFactory, IteratorType
from src.model.SegmentationModelFactory import SegmentationModelFactory
from src.model.SemanticSegmentationModel import SemanticSegmentationModel
from src.train_eval.core.config_readers.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.train_eval.core.optimizer_wrappers.OptimizerWrapperFactory import OptimizerWrapperFactory
from src.train_eval.core.persistence.PersistenceManager import PersistenceManager


class GraphExecutor(ABC):

    def __init__(self, descriptive_name: str, config: GraphExecutorConfigReader):
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

    def _get_tf_session_config(self) -> tf.ConfigProto:
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        return config

    def _build_computation_graph(self) -> Tuple[tf.data.Iterator, Union[tf.Tensor, Optional[tf.Tensor]], tf.Tensor, tf.Tensor]:
        iterator = self._iterator_factory.get_iterator(self._get_iterator_type())
        X, y = iterator.get_next()
        model_out = self._model.run(X, self._config.num_classes, is_training=self._config.mode == 'train', y=y)
        return iterator, model_out, X, y

    def _initialize_optimizer(self) -> tf.train.Optimizer:
        optimizer_wrapper_factory = OptimizerWrapperFactory()
        optimizer_wrapper = optimizer_wrapper_factory.assembly(self._config.optimizer_options)
        return optimizer_wrapper.get_optimizer()

    def __construct_model(self) -> SemanticSegmentationModel:
        model_factory = SegmentationModelFactory()
        return model_factory.assembly(self._config.model_name)
