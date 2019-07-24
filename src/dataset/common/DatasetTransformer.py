from typing import Tuple, List

import tensorflow as tf

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.common.transformations.DatasetTransformation import DatasetTransformation
from src.dataset.common.transformations.DatasetTransformationFactory import DatasetTransformationFactory
from src.dataset.common.transformations.TransformationType import TransformationType

from logging import getLogger

logger = getLogger(__file__)


class DatasetTransformer:

    def __init__(self, config: GraphExecutorConfigReader):
        self.__config = config

    def augment_data(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        transformation_chain = self.__get_transformation_chain()
        for operation, operation_probab in transformation_chain:
            operation_type = operation.get_transformation_type()
            operation_parameters_key = '{}_parameters'.format(operation_type.value)
            operation_parameters = None
            if operation_parameters_key in self.__config.data_transoformation_options:
                operation_parameters = self.__config.data_transoformation_options[operation_parameters_key]
            image, label = operation.apply(image, label, operation_probab, operation_parameters)
        return image, label

    def __get_transformation_chain(self) -> List[Tuple[DatasetTransformation, float]]:
        transformations = self.__get_transformations()
        transformation_chain = []
        transformation_factory = DatasetTransformationFactory()
        for transformation in transformations:
            operation = transformation_factory.assembly_transformation(transformation)
            operation_probab = self.__config.data_transoformation_options['{}_probability'.format(transformation.value)]
            transformation_chain.append((operation, operation_probab))
        return transformation_chain

    def __get_transformations(self) -> List[TransformationType]:
        transformations = []
        transformation_names = dict([(item.value, item) for item in TransformationType])
        for transformation_name in self.__config.data_transoformation_options['transformation_chain']:
            if transformation_name not in transformation_names:
                logger.warning('There is no transformation with name: %s', transformation_name)
            else:
                transformations.append(transformation_names[transformation_name])
        return transformations
