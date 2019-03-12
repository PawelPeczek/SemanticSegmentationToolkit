import tensorflow as tf
import sys
from src.dataset.common.CityScapesDataset import CityScapesDataset
from src.model.SegmentationModelFactory import SegmentationModelFactory
from src.training.core.optimizer_wrappers.OptimizerWrapperFactory import OptimizerWrapperFactory
from src.training.core.persistence.PersistenceManager import PersistenceManager


class Trainer:

    def __init__(self, descriptive_name, config):
        """
        :param descriptive_name: training process descriptive identifier
        :param config: TrainingConfigReader object
        """
        self.__config = config
        self.__descriptive_name = descriptive_name
        self.__model = self.__initialize_model()
        self.__persistence_manager = PersistenceManager(descriptive_name, config)

    def train(self):
        self.__persistence_manager.prepare_storage()
        iterator, error, optimization_op = self.__build_computation_graph()
        if self.__use_multi_gpu():
            self.__train_on_single_gpu(iterator, error, optimization_op)
        else:
            pass

    def __initialize_model(self):
        model_factory = SegmentationModelFactory()
        return model_factory.assembly(self.__config.model_name)

    def __build_computation_graph(self):
        dataset = CityScapesDataset(self.__config)
        iterator = dataset.get_training_iterator(self.__config.batch_size)
        X, y = iterator.get_next()
        model_out = self.__model.run(X, self.__config.num_classes, is_training=True)
        error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_out,
                                                                              labels=y))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = self.__initialize_optimizer()
            train_step = optimizer.minimize(error)
        return iterator, error, train_step

    def __initialize_optimizer(self):
        optimizer_wrapper_factory = OptimizerWrapperFactory()
        optimizer_wrapper = optimizer_wrapper_factory.assembly(self.__config.optimizer_options)
        return optimizer_wrapper.get_optimizer()

    def __get_tf_session_config(self):
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True)
        config.gpu_options.allow_growth = True
        return config

    def __train_on_single_gpu(self, iterator, error, optimization_op):
        sess_config = self.__get_tf_session_config()
        with tf.Session(config=sess_config) as sess:
            with tf.device("/gpu:{}".format(self.__get_gpu_to_use())):
                sess.run(tf.global_variables_initializer())
                self.__single_gpu_train_loop(sess, iterator,  error, optimization_op)

    def __single_gpu_train_loop(self, sess, iterator, error, optimization_op):
        print('TRAINING [STARTED]')
        saving_freq_decreased = False
        acc_measure_after_ith_ep = self.__config.saving_frequency
        for i in range(0, self.__config.epochs):
            sess.run(iterator.initializer)
            variables = [error, optimization_op]
            errors = []
            print("===========================================================================")
            try:
                while True:
                    error_eval, _ = sess.run(variables)
                    self.__dump_stats_on_screen(i + 1, error_eval)
                    errors.append(error_eval)
                    if self.__saving_frequency_should_be_increased(error_eval, saving_freq_decreased):
                        acc_measure_after_ith_ep = int(round(acc_measure_after_ith_ep / 2))
                        saving_freq_decreased = True
            except tf.errors.OutOfRangeError:
                avg_error = sum(errors) / len(errors) if len(errors) > 0 else 'NaN'
                print('\nAverage epoch loss: {}'.format(avg_error))
                print("===========================================================================")
                self.__persistence_manager.log_loss(i + 1, avg_error)
                if (i + 1) % acc_measure_after_ith_ep is 0:
                    self.__persistence_manager.persist_model(sess, i + 1)
        print('TRAINING [FINISHED]')

    def __dump_stats_on_screen(self, epoch, loss):
        sys.stdout.write("\rMid-Epoch #{}\tlast batch loss value: {}\t\t".format(epoch, loss))
        sys.stdout.flush()

    def __saving_frequency_should_be_increased(self, error_val, saving_freq_decreased):
        return error_val < self.__config.increase_saving_frequency_loss_treshold and not saving_freq_decreased

    def __get_gpu_to_use(self):
        if isinstance(list, self.__config.gpu_to_use):
            if self.__use_multi_gpu():
                return self.__config.gpu_to_use
            else:
                return self.__config.gpu_to_use[0]
        return self.__config.gpu_to_use

    def __use_multi_gpu(self):
        return len(self.__config.gpu_to_use) > 0





