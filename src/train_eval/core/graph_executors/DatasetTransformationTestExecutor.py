from typing import Dict, Tuple
import matplotlib.pyplot as plt
import tensorflow as tf

from src.dataset.common.CityScapesIteratorFactory import IteratorType
from src.dataset.utils.mapping_utils import get_id_to_colour_mapping, map_colour
from src.train_eval.core.config_readers.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.train_eval.core.graph_executors.GraphExecutor import GraphExecutor
from src.train_eval.core.persistence.PersistenceManager import PersistenceManager
from src.train_eval.core.persistence.TrainingPersistenceManager import TrainingPersistenceManager


class DatasetTransformationTestExecutor(GraphExecutor):

    def __init__(self, descriptive_name: str, config: GraphExecutorConfigReader):
        super().__init__(descriptive_name, config)
        self._config.data_transoformation_options['application_probability'] = 1.0

    def execute(self) -> None:
        _, _, image, label = self._build_computation_graph()
        image = tf.cast(image, tf.uint8)
        config = self._get_tf_session_config()
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:{}".format(self._config.gpu_to_use)):
                self.__proceed_inference(sess, image, label)

    def _get_iterator_type(self) -> IteratorType:
        return IteratorType.OS_TRAIN_ITERATOR

    def _get_persistence_manager(self) -> PersistenceManager:
        return TrainingPersistenceManager(self._descriptive_name, self._config)

    def __proceed_inference(self, sess: tf.Session, image: tf.Tensor, label: tf.Tensor) -> None:
        mappings = get_id_to_colour_mapping(self._config.mapping_file)
        try:
            while True:
                self.__proceed_inference_on_batch(sess, image, label, mappings)
        except tf.errors.OutOfRangeError:
            print('Inference [DONE]')

    def __proceed_inference_on_batch(self, sess: tf.Session, image: tf.Tensor, label: tf.Tensor,
                                     mappings: Dict[int, Tuple[int, int, int]]) -> None:
        base, gt = sess.run([image, label])
        fig = plt.figure(figsize=(20, 40))
        for i in range(0, base.shape[0]):
            to_show = base[i][..., ::-1]
            fig.add_subplot(base.shape[0], 2, 2 * i + 1)
            plt.imshow(to_show)
            fig.add_subplot(base.shape[0], 2, 2 * i + 2)
            ground_truth = gt[i]
            ground_truth = map_colour(ground_truth, mappings)
            plt.imshow(ground_truth)
        image_path = self._persistence_manager.generate_random_transformation_test_image_path()
        plt.savefig(image_path)
        plt.close()
