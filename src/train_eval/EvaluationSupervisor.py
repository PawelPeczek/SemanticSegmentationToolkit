from fire import Fire

from src.dataset.common.CityScapesIteratorFactory import IteratorType
from src.train_eval.core.GraphExecutor import GraphExecutor
from src.train_eval.core.TrainValConfigReader import TrainValConfigReader


class EvaluationSupervisor:

    def evaluate_model(self, descriptive_name=None, config_path=None):
        try:
            config = TrainValConfigReader(config_path, reader_type='val')
            if descriptive_name is None:
                descriptive_name = 'model_evaluation'
            executor = GraphExecutor(descriptive_name, config, IteratorType.VALIDATION_ITERATOR)
            executor.evaluate()
        except Exception as ex:
            print('Failed to proceed model evaluation. {}'.format(ex))

    def get_predictions_from_model(self, descriptive_name=None, config_path=None):
        try:
            config = TrainValConfigReader(config_path, reader_type='val')
            if descriptive_name is None:
                descriptive_name = 'model_predictions'
            executor = GraphExecutor(descriptive_name, config, IteratorType.DUMMY_ITERATOR)
            executor.infer()
        except Exception as ex:
            print('Failed to proceed taking inference from model. {}'.format(ex))

    def measure_inference_speed(self, descriptive_name=None, config_path=None):
        try:
            config = TrainValConfigReader(config_path, reader_type='val')
            if descriptive_name is None:
                descriptive_name = 'model_inference_speed_measurement'
            executor = GraphExecutor(descriptive_name, config, IteratorType.DUMMY_ITERATOR)
            executor.test_inference_speed()
        except Exception as ex:
            print('Failed to proceed speed evaluation. {}'.format(ex))


if __name__ == '__main__':
    Fire(EvaluationSupervisor)
