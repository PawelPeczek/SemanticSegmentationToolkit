import tensorflow as tf
import sys
from src.dataset.common.CityScapesDataset import CityScapesDataset
from src.model.SegmentationModelFactory import SegmentationModelFactory
from src.training.core.optimizer_wrappers.OptimizerWrapperFactory import OptimizerWrapperFactory
from src.training.core.persistence.PersistenceManager import PersistenceManager


class Trainer:

    PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

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
        if self.__use_multi_gpu():
            self.__train_on_multiple_gpus()
        else:
            self.__train_on_single_gpu()

    def __initialize_model(self):
        model_factory = SegmentationModelFactory()
        return model_factory.assembly(self.__config.model_name)

    def __build_computation_graph(self):
        dataset = CityScapesDataset(self.__config)
        iterator = dataset.get_training_iterator(self.__config.batch_size)
        X, y = iterator.get_next()
        model_out = self.__model.run(X, self.__config.num_classes, is_training=True)
        return iterator, model_out, y

    def __initialize_optimizer(self):
        optimizer_wrapper_factory = OptimizerWrapperFactory()
        optimizer_wrapper = optimizer_wrapper_factory.assembly(self.__config.optimizer_options)
        return optimizer_wrapper.get_optimizer()

    def __get_tf_session_config(self):
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True)
        return config

    def __train_on_single_gpu(self):
        iterator, model_out, ground_truth = self.__build_computation_graph()
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_out,
                                                                             labels=ground_truth))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = self.__initialize_optimizer()
            optimization_op = optimizer.minimize(loss)
        sess_config = self.__get_tf_session_config()
        with tf.Session(config=sess_config) as sess:
            with tf.device("/gpu:{}".format(self.__get_gpu_to_use())):
                self.__train_loop(sess, iterator, loss, optimization_op)

    def __train_on_multiple_gpus(self):
        with tf.device('/cpu:0'):
            grads_acc = []
            loss_acc = []
            gpus_to_use = self.__get_gpu_to_use()
            dataset = CityScapesDataset(self.__config)
            iterator = dataset.get_training_iterator(self.__config.batch_size)
            for gpu_id in gpus_to_use:
                with tf.device(self.__assign_to_device('/gpu:{}'.format(gpu_id), ps_device='/cpu:0')):
                    X, y = iterator.get_next()
                    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                        model_out = self.__model.run(X, self.__config.num_classes, is_training=True)
                    loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_out, labels=y)
                    loss_op = tf.reduce_mean(loss_op)
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        optimizer = self.__initialize_optimizer()
                        grads = optimizer.compute_gradients(loss_op)
                    grads_acc.append(grads)
                    loss_acc.append(loss_op)
            grads_acc = self.__get_average_gradients_op(grads_acc)
            loss_acc = self.__get_average_errors_op(loss_acc)
            train_op = optimizer.apply_gradients(grads_acc)
            sess_config = self.__get_tf_session_config()
            with tf.Session(config=sess_config) as sess:
                self.__train_loop(sess, iterator, loss_acc, train_op)

    def __assign_to_device(self, device, ps_device='/cpu:0'):

        def assign(op):
            node_def = op if isinstance(op, tf.NodeDef) else op.node_def
            if node_def.op in self.PS_OPS:
                return "/" + ps_device
            else:
                return device

        return assign

    def __get_average_gradients_op(self, gardients):
        """

        The function originally comes from: https://github.com/jhui/deep_learning/

        Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          gardients: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*gardients):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def __get_average_errors_op(self, errors):
        errors = tf.stack(errors, axis=0)
        return tf.reduce_mean(errors, axis=0)

    def __train_loop(self, sess, iterator, loss, optimization_op):
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        print('TRAINING [STARTED]')
        saving_freq_decreased = False
        acc_measure_after_ith_ep = self.__config.saving_frequency
        for i in range(0, self.__config.epochs):
            sess.run(iterator.initializer)
            variables = [loss, optimization_op]
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
        if isinstance(self.__config.gpu_to_use, list):
            if self.__use_multi_gpu():
                return self.__config.gpu_to_use
            else:
                return self.__config.gpu_to_use[0]
        return self.__config.gpu_to_use

    def __use_multi_gpu(self):
        return len(self.__config.gpu_to_use) > 1
