import time
import statistics

import tensorflow as tf
import numpy as np


class ProfilingResult:

    def __init__(self, input_size: list, mean_time: float, stddev_time: float):
        self.__input_size = input_size
        self.__mean_time = mean_time
        self.__stddev_time = stddev_time


    @property
    def input_size(self) -> list:
        return self.__input_size


    @property
    def mean_time(self) -> float:
        return self.__mean_time

    @property
    def stddev_time(self) -> float:
        return self.__stddev_time

    def __str__(self) -> str:
        return f'{self.__mean_time},{self.__stddev_time}'


def profile_operation(operation: tf.Tensor,
                      x_placeholder: tf.Tensor,
                      device: str
                      ) -> ProfilingResult:
    time_results = []
    with tf.device(device):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(105):
                shape = x_placeholder.shape.as_list()
                random_input = np.random.normal(0.0, 1.0, shape)
                start_time = time.time()
                sess.run(operation, feed_dict={x_placeholder: random_input})
                end_time = time.time()
                operation_time = end_time - start_time
                time_results.append(operation_time)
    time_results = time_results[5:]
    mean = statistics.mean(time_results)
    std_dev = statistics.stdev(time_results)
    return ProfilingResult(
        input_size=x_placeholder.shape.as_list(),
        mean_time=mean,
        stddev_time=std_dev
    )


