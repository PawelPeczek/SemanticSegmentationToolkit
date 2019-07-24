import tensorflow as tf
import sys
from typing import List, Union

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.common.iterators import IteratorType
from src.train_eval.core.graph_executors.graph_executor import GraphExecutor
from src.train_eval.core.graph_executors.utils import ValidationOperation, \
    evaluate_miou, average_gradients, average_loss, SessionOperations, \
    ValidationOperations, MultiGPUOperations, get_gradient, \
    minimize_loss, assign_to_device, get_validation_operation
from src.train_eval.core.persistence.managers import PersistenceManager, \
    TrainingPersistenceManager


class TrainingExecutor(GraphExecutor):

    LoopVariables = List[Union[tf.Tensor, tf.Operation]]

    def __init__(self,
                 descriptive_name: str,
                 config: GraphExecutorConfigReader,
                 iterator_type: IteratorType):
        super().__init__(descriptive_name, config)
        self.__iterator_type = iterator_type
        self.__saving_frequency_increased = False

    def execute(self) -> None:
        if self.__use_multi_gpu():
            self.__train_on_multiple_gpus()
        else:
            self.__train_on_single_gpu()

    def _get_iterator_type(self) -> IteratorType:
        return self.__iterator_type

    def _get_persistence_manager(self) -> PersistenceManager:
        return TrainingPersistenceManager(
            descriptive_name=self._descriptive_name,
            config=self._config)

    def __train_on_single_gpu(self) -> None:
        with tf.device("/gpu:{}".format(self.__get_gpu_to_use())):
            iterator = self._get_iterator()
            optimizer = self._initialize_optimizer()
            x, y = iterator.get_next()
            loss = self._model.training_pass(x, y)
            gradient_update = minimize_loss(optimizer, loss)
        validation_operations = self.__initialize_validation_operations()
        sess_ops = SessionOperations(
            iterator=iterator,
            loss_operation=loss,
            gradient_update=gradient_update,
            validation_operations=validation_operations)
        self.__run_training_session(sess_ops)

    def __train_on_multiple_gpus(self) -> None:
        with tf.device('/cpu:0'):
            iterator = self._iterator_factory.get_iterator(self.__iterator_type)
            optimizer = self._initialize_optimizer()
            loss_operation, gradient_update = self.__distribute_training(
                iterator=iterator,
                optimizer=optimizer)
            validation_operations = self.__initialize_validation_operations()
            sess_ops = SessionOperations(
                iterator=iterator,
                loss_operation=loss_operation,
                gradient_update=gradient_update,
                validation_operations=validation_operations)
            self.__run_training_session(sess_ops)

    def __initialize_validation_operations(self) -> ValidationOperations:
        with tf.device("/gpu:{}".format(self.__get_primary_gpu())):
            training_set_eval_operation = self.__get_validation_operation(
                IteratorType.INITIALIZABLE_TRAIN_SET_ITERATOR)
            test_set_eval_operation = self.__get_validation_operation(
                IteratorType.INITIALIZABLE_VALIDATION_ITERATOR)
        return ValidationOperations(
            training_set_evaluation=training_set_eval_operation,
            test_set_evaluation=test_set_eval_operation)

    def __run_training_session(self,
                               session_operations: SessionOperations) -> None:
        sess_config = self._get_session_config()
        with tf.Session(config=sess_config) as sess:
            self.__train(sess, session_operations)

    def __distribute_training(self,
                              iterator: tf.data.Iterator,
                              optimizer: tf.train.Optimizer):
        gpus_to_use = self.__get_gpu_to_use()
        gradients, loss_operations = [], []
        for gpu_id in gpus_to_use:
            multi_gpu_operations = self.__place_operations(
                target_gpu_id=gpu_id,
                iterator=iterator,
                optimizer=optimizer)
            gradients.append(multi_gpu_operations.gradient)
            loss_operations.append(multi_gpu_operations.loss_operation)
        gradients = average_gradients(gradients)
        loss_operation = average_loss(loss_operations)
        training_step = optimizer.apply_gradients(gradients)
        return loss_operation, training_step

    def __place_operations(self,
                           target_gpu_id: str,
                           iterator: tf.data.Iterator,
                           optimizer: tf.train.Optimizer) -> MultiGPUOperations:
        gpu_device = f'/gpu:{target_gpu_id}'
        with tf.device(assign_to_device(target_device=gpu_device)):
            loss_operation = self.__get_multi_training_loss(iterator)
            gradient = get_gradient(
                optimizer=optimizer,
                loss_operation=loss_operation)
            return MultiGPUOperations(
                gradient=gradient,
                loss_operation=loss_operation)

    def __get_multi_training_loss(self,
                                  iterator: tf.data.Iterator) -> tf.Operation:
        x, y = iterator.get_next()
        variable_scope = tf.get_variable_scope()
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
            loss_operation = self._model.training_pass(x, y)
        return loss_operation

    def __get_validation_operation(self,
                                   iterator_type: IteratorType) -> ValidationOperation:
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            iterator = self._iterator_factory.get_iterator(iterator_type)
            num_classes = self._config.num_classes
            labels_to_ignore = self._config.get_or_else('ignore_labels', None)
            return get_validation_operation(
                iterator=iterator,
                model=self._model,
                num_classes=num_classes,
                labels_to_ignore=labels_to_ignore)

    def __train(self,
                session: tf.Session,
                session_operations: SessionOperations) -> None:
        session.run(tf.global_variables_initializer())
        print('TRAINING [STARTED]')
        for epoch_id in range(0, self._config.epochs):
            self.__make_training_epoch(
                epoch_id=epoch_id,
                session=session,
                session_operations=session_operations)
        print('TRAINING [FINISHED]')

    def __make_training_epoch(self,
                              epoch_id: int,
                              session: tf.Session,
                              session_operations: SessionOperations) -> None:
        session.run(session_operations.iterator.initializer)
        loop_variables = [
            session_operations.loss_operation,
            session_operations.gradient_update
        ]
        errors = self.__proceed_epoch_loop(epoch_id, session, loop_variables)
        avg_error = sum(errors) / len(errors) if len(errors) > 0 else 'NaN'
        self.__finish_epoch(
            session=session,
            epoch_id=epoch_id + 1,
            avg_error=avg_error,
            validation_ops=session_operations.validation_operations)
        if self.__model_should_be_persisted(epoch_id + 1):
            self._persistence_manager.persist_model(session, epoch_id + 1)

    def __proceed_epoch_loop(self,
                             epoch_id: int,
                             session: tf.Session,
                             loop_variables: LoopVariables) -> List[float]:
        errors = []
        try:
            while True:
                step_error, _ = session.run(loop_variables)
                self.__dump_stats(epoch_id + 1, step_error)
                errors.append(step_error)
                if self.__saving_frequency_should_be_increased(step_error):
                    self.__increase_saving_frequency()
        except tf.errors.OutOfRangeError:
            return errors

    def __dump_stats(self, epoch: int, loss: float) -> None:
        sys.stdout.write("\rMid-Epoch #{}\tlast batch loss value: {}\t\t".format(epoch, loss))
        sys.stdout.flush()

    def __saving_frequency_should_be_increased(self, error_val: float) -> bool:
        threshold_passed = \
            error_val < self._config.increase_saving_frequency_loss_treshold
        return threshold_passed and not self.__saving_frequency_increased

    def __increase_saving_frequency(self) -> None:
        saving_frequency = self._config.saving_frequency
        saving_frequency = int(round(saving_frequency / 2))
        self._config.saving_frequency = saving_frequency

    def __model_should_be_persisted(self, epoch_num: int) -> bool:
        return epoch_num % self._config.saving_frequency is 0

    def __finish_epoch(self,
                       session: tf.Session,
                       validation_ops: ValidationOperations,
                       epoch_id: int,
                       avg_error: Union[float, str]) -> None:
        print('\nAverage epoch loss: {}'.format(avg_error))
        train_acc, val_acc = None, None
        if self.__should_measure_train_acc(epoch_id):
            train_acc = self.__measure_miou(
                session=session,
                validation_operation=validation_ops.training_set_evaluation)
            print('\nmIoU acc on training set: {}%'.format(train_acc * 100))
        if self.__should_measure_val_acc(epoch_id):
            val_acc = self.__measure_miou(
                session=session,
                validation_operation=validation_ops.test_set_evaluation)
            print('\nmIoU acc on validation set: {}%'.format(val_acc * 100))
        self._persistence_manager.log_loss(
            epoch=epoch_id,
            loss_value=avg_error,
            train_acc=train_acc,
            val_acc=val_acc)

    def __should_measure_train_acc(self, epoch_num: int) -> bool:
        epoch_number_matches = \
            epoch_num % self._config.measure_train_accuracy_frequency is 0
        return self._config.measure_train_accuracy and epoch_number_matches

    def __should_measure_val_acc(self, epoch_number: int) -> bool:
        epoch_number_matches = \
            epoch_number % self._config.measure_val_accuracy_frequency is 0
        return self._config.measure_val_accuracy and epoch_number_matches

    def __measure_miou(self,
                       session: tf.Session,
                       validation_operation: ValidationOperation) -> float:
        primary_gpu = self.__get_primary_gpu()
        with tf.device(f'/gpu:{primary_gpu}'):
            return evaluate_miou(
                session=session,
                validation_operation=validation_operation)

    def __get_primary_gpu(self) -> str:
        gpu_to_use = self.__get_gpu_to_use()
        if isinstance(gpu_to_use, list):
            return gpu_to_use[0]
        else:
            return gpu_to_use

    def __get_gpu_to_use(self) -> Union[str, List[str]]:
        if isinstance(self._config.gpu_to_use, list):
            if self.__use_multi_gpu():
                return self._config.gpu_to_use
            else:
                return self._config.gpu_to_use[0]
        return self._config.gpu_to_use

    def __use_multi_gpu(self) -> bool:
        return len(self._config.gpu_to_use) > 1
