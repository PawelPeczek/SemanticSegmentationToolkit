import os
from typing import Tuple
import tensorflow as tf
import cv2 as cv
import numpy as np

from src.common.config_utils import InferenceConfigReader
from src.dataset.utils.mapping_utils import get_id_to_color_mapping
from src.model.network import Network
from src.model.utils import ModelFactory
from src.utils.filesystem_utils import create_directory


class StreamInferenceUtil:

    def __init__(self, config: InferenceConfigReader):
        self.__config = config
        self.__model = self.__construct_model()
        self.__prepare_storage()

    def infer_on_video_stream(self) -> None:
        params = self.__prepare_prediction_to_color_mapping()
        x_placeholder, prediction = self.__build_feedable_graph()
        prediction = tf.gather(params, prediction)
        prediction = tf.cast(prediction, dtype=tf.uint8)
        saver = tf.train.Saver()
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:{}".format(self.__config.gpu_to_use)):
                saver.restore(sess, self.__config.checkpoint_name)
                self.__proceed_inference_on_video_stream(
                    sess, x_placeholder, prediction
                )

    def __prepare_prediction_to_color_mapping(self) -> np.ndarray:
        mappings = get_id_to_color_mapping(self.__config.mapping_file)
        return np.array([(0, 0, 0)] + list(mappings.values()))

    def __construct_model(self) -> Network:
        model_factory = ModelFactory()
        return model_factory.assembly(
            model_name=self.__config.model_name,
            output_classes=self.__config.num_classes,
        )

    def __build_feedable_graph(self) -> Tuple[tf.placeholder, tf.Tensor]:
        inference_size = self.__config.destination_size
        x = tf.placeholder(
            tf.float32, shape=[1, inference_size[1], inference_size[0], 3]
        )
        model_out = self.__model.infer(x)
        return x, model_out

    def __proceed_inference_on_video_stream(self,
                                            sess: tf.Session,
                                            x_placeholder: tf.Tensor,
                                            prediction: tf.Tensor):
        video_file_path = self.__config.input_video
        video = cv.VideoCapture(video_file_path)
        saver = None
        if self.__config.persist_video is True:
            saver = cv.VideoWriter(
                self.__config.output_video_file,
                cv.VideoWriter_fourcc(*'DIVX'),
                25,
                (self.__config.destination_size[0], self.__config.destination_size[1])
            )
        while True:
            success, frame = video.read()
            if not success:
                break
            frame = np.expand_dims(frame, axis=0)
            prediction_eval = sess.run(
                prediction, feed_dict={x_placeholder: frame}
            )
            overlay = cv.addWeighted(frame, 0.1, prediction_eval, 0.9, 0)
            cv.imshow('Infered video', overlay[0, :, :, ::-1].copy())
            if self.__config.persist_video is True:
                saver.write(overlay[0, :, :, ::-1])
            cv.waitKey(1)
        if self.__config.persist_video is True:
            saver.release()

    def __prepare_storage(self):
        if self.__config.persist_video:
            output_directory = os.path.dirname(self.__config.output_video_file)
            create_directory(output_directory)
